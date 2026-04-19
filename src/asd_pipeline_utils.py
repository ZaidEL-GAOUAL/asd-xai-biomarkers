from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

GSE_ID = "GSE18123"
RANDOM_STATE = 42

# Base paths (relative to notebooks/ directory)
_DATA_RAW = Path("../data/raw")
_DATA_PROC = Path("../data/processed")
_RESULTS = Path("../results/tables")

DESTDIR = str(_DATA_RAW) + "/"
PARQUET_PATH = _DATA_PROC / "GSE18123_commonGenes_2platforms.parquet"
CSV_PATH = _DATA_PROC / "GSE18123_commonGenes_2platforms.csv.gz"
METADATA_PATH = _DATA_PROC / "GSE18123_sample_metadata.csv"
BASELINE_BINARY_PATH = _RESULTS / "baseline_results_binary.csv"
BASELINE_MULTICLASS_PATH = _RESULTS / "baseline_results_multiclass.csv"
SIGNATURE_STABILITY_PATH = _RESULTS / "signature_stability_binary_top100.csv"
SMOTE_PFI_PATH = _RESULTS / "smote_pfi_results.csv"
XAI_COMPARISON_PATH = _RESULTS / "xai_comparison_results.csv"
ROBUSTNESS_PATH = _RESULTS / "robustness_analysis_results.csv"


def require_geoparse():
    try:
        import GEOparse  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "This workflow requires GEOparse. Install it before running the data-loading notebooks."
        ) from exc
    return GEOparse


def clean_symbol(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip()
    if not value or value.lower() in {"na", "nan", "none", "---"}:
        return np.nan
    if "///" in value:
        value = value.split("///")[0].strip()
    return value


def parse_gene_from_gpl6244_gene_assignment(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip()
    if not value or value.lower() in {"na", "nan", "none", "---"}:
        return np.nan

    parts = [part.strip() for part in value.split(" // ") if part.strip()]
    if not parts:
        return np.nan

    bad_prefixes = ("ENST", "ENSG", "NM_", "NR_", "XM_", "XR_", "NP_", "AP", "AF", "AL", "BC")
    candidates = []
    for part in parts[:5]:
        token = part.split(" /// ")[0].split(" / ")[0].strip()
        if not token or token.startswith(bad_prefixes):
            continue
        if re.match(r"^[A-Za-z0-9][A-Za-z0-9\\-\\._]*$", token):
            candidates.append(token)

    if candidates:
        return clean_symbol(candidates[0])

    if len(parts) >= 2 and re.match(r"^[A-Za-z0-9][A-Za-z0-9\\-\\._]*$", parts[1].strip()):
        return clean_symbol(parts[1].strip())

    return clean_symbol(parts[0])


def build_platform_matrix(expr_probes, gpl_table, sample_ids, platform_id):
    gpl = gpl_table.copy()
    gpl["ID"] = gpl["ID"].astype(str)
    expr_probes.index = expr_probes.index.astype(str)

    platform_probes = [probe for probe in gpl["ID"].unique().tolist() if probe in expr_probes.index]
    if not platform_probes:
        print(f"[{platform_id}] 0 probes found after aligning IDs.")
        return pd.DataFrame(index=sample_ids)

    sub = expr_probes.loc[platform_probes, sample_ids].copy()

    if platform_id == "GPL570":
        gene_col = None
        for column in gpl.columns:
            normalized = column.lower().replace("_", " ").strip()
            if normalized in {"gene symbol", "gene_symbol"}:
                gene_col = column
                break
        if gene_col is None:
            for column in gpl.columns:
                lowered = column.lower()
                if "gene" in lowered and "symbol" in lowered:
                    gene_col = column
                    break
        if gene_col is None:
            raise ValueError(f"GPL570 gene symbol column not found. Available columns: {list(gpl.columns)}")

        ann = gpl[["ID", gene_col]].dropna()
        ann[gene_col] = ann[gene_col].map(clean_symbol)
        ann = ann.dropna()
        probe_to_gene = dict(zip(ann["ID"].astype(str), ann[gene_col].astype(str)))
    elif platform_id == "GPL6244":
        if "gene_assignment" not in gpl.columns:
            raise ValueError(
                f"GPL6244 gene_assignment column not found. Available columns: {list(gpl.columns)}"
            )
        ann = gpl[["ID", "gene_assignment"]].dropna()
        ann["gene_symbol"] = ann["gene_assignment"].map(parse_gene_from_gpl6244_gene_assignment)
        ann = ann.dropna(subset=["gene_symbol"])
        probe_to_gene = dict(zip(ann["ID"].astype(str), ann["gene_symbol"].astype(str)))
    else:
        raise ValueError(f"Unsupported platform: {platform_id}")

    mapped_index = sub.index.map(lambda probe_id: probe_to_gene.get(str(probe_id), np.nan))
    sub.index = mapped_index
    sub = sub.dropna(axis=0, how="any")

    if sub.empty:
        print(f"[{platform_id}] 0 probes remained after probe-to-gene mapping.")
        return pd.DataFrame(index=sample_ids)

    sub = sub.groupby(sub.index).mean()
    return sub.T


def nan_report(df, title=""):
    print(f"\nNaN report — {title}")
    rate = df.isna().mean().sort_values(ascending=False)
    print("Shape:", df.shape)
    print("Columns with any NaN:", int((rate > 0).sum()), "/", df.shape[1])
    print("Top missing-rate columns:")
    print(rate.head(15))


def load_geo_series(gse_id=GSE_ID, destdir=DESTDIR):
    geoparse = require_geoparse()
    print(f"Loading {gse_id} ...")
    gse = geoparse.get_GEO(gse_id, destdir=destdir)
    print("Pivot probes × samples ...")
    expr_probes = gse.pivot_samples("VALUE").apply(pd.to_numeric, errors="coerce")
    return gse, expr_probes


def extract_sample_metadata(gse, expr_probes):
    diagnosis_by_sample = {}
    platform_by_sample = {}

    for gsm_name, gsm in gse.gsms.items():
        platform_by_sample[gsm_name] = gsm.metadata["platform_id"][0]
        diagnosis = None
        for item in gsm.metadata.get("characteristics_ch1", []):
            if ":" not in item:
                continue
            key, value = item.split(":", 1)
            if key.strip().lower() == "diagnosis":
                diagnosis = value.strip()
        diagnosis_by_sample[gsm_name] = diagnosis

    diagnosis_df = pd.DataFrame.from_dict(diagnosis_by_sample, orient="index", columns=["diagnosis"])
    platform_s = pd.Series(platform_by_sample, name="platform_id")

    common_samples = expr_probes.columns.intersection(diagnosis_df.index)
    expr_probes = expr_probes[common_samples]
    diagnosis_df = diagnosis_df.loc[common_samples]
    platform_s = platform_s.loc[common_samples]

    mask = diagnosis_df["diagnosis"].notna()
    expr_probes = expr_probes.loc[:, mask.values]
    diagnosis_df = diagnosis_df.loc[mask]
    platform_s = platform_s.loc[mask]
    return expr_probes, diagnosis_df, platform_s


def save_metadata(diagnosis_df, platform_s, path=METADATA_PATH):
    metadata_df = pd.DataFrame(
        {
            "diagnosis": diagnosis_df["diagnosis"].astype(str).str.strip(),
            "platform_id": platform_s.astype(str),
        },
        index=diagnosis_df.index,
    )
    metadata_df.to_csv(path)
    return metadata_df


def build_common_gene_dataset(gse, expr_probes, diagnosis_df, platform_s, apply_log2=True):
    idx_570 = platform_s[platform_s == "GPL570"].index
    idx_6244 = platform_s[platform_s == "GPL6244"].index

    print("\nBuilding GPL570 gene matrix...")
    x_570 = build_platform_matrix(expr_probes, gse.gpls["GPL570"].table, idx_570, "GPL570")
    print("GPL570 matrix:", x_570.shape)

    print("\nBuilding GPL6244 gene matrix...")
    x_6244 = build_platform_matrix(expr_probes, gse.gpls["GPL6244"].table, idx_6244, "GPL6244")
    print("GPL6244 matrix:", x_6244.shape)

    common_genes = x_570.columns.intersection(x_6244.columns)
    print("\nGPL570 genes:", x_570.shape[1])
    print("GPL6244 genes:", x_6244.shape[1])
    print("Common genes:", len(common_genes))
    print("Example common genes:", list(common_genes[:10]))

    if len(common_genes) == 0:
        raise RuntimeError("No common genes were found across GPL570 and GPL6244.")

    x = pd.concat([x_570[common_genes], x_6244[common_genes]], axis=0)
    if apply_log2:
        max_value = np.nanmax(x.to_numpy())
        if max_value > 50:
            print(f"\nHeuristic: max={max_value:.2f} > 50 → applying log2(x+1)")
            x = np.log2(x + 1)

    meta = pd.DataFrame(
        {
            "diagnosis": diagnosis_df.loc[x.index, "diagnosis"].astype(str).str.strip(),
            "platform_id": platform_s.loc[x.index].astype(str),
        },
        index=x.index,
    )
    final_df = x.join(meta)
    return x, meta, final_df


def save_analysis_frame(final_df, parquet_path=PARQUET_PATH, csv_path=CSV_PATH):
    final_df.to_parquet(parquet_path, index=True)
    final_df.to_csv(csv_path, compression="gzip", index=True)
    return parquet_path, csv_path


def load_analysis_frame(parquet_path=PARQUET_PATH, csv_path=CSV_PATH):
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path, compression="gzip", index_col=0)
    raise FileNotFoundError(
        "Processed analysis matrix not found. Run 02_probe_mapping_and_gene_matrix.ipynb first."
    )


def split_features_and_metadata(final_df):
    meta = final_df[["diagnosis", "platform_id"]].copy()
    x = final_df.drop(columns=["diagnosis", "platform_id"])
    return x, meta


def get_targets(meta):
    y_bin = (meta["diagnosis"].str.upper() != "CONTROL").astype(int)
    y_multi = meta["diagnosis"].astype("category")
    return y_bin, y_multi


def build_baseline_pipeline():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("var", VarianceThreshold(threshold=0.0)),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="saga",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    max_iter=8000,
                ),
            ),
        ]
    )


def baseline_scoring():
    scoring_multi = {
        "acc": "accuracy",
        "bal_acc": "balanced_accuracy",
        "f1_macro": make_scorer(f1_score, average="macro"),
        "f1_weighted": make_scorer(f1_score, average="weighted"),
    }
    scoring_bin = {
        "acc": "accuracy",
        "bal_acc": "balanced_accuracy",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    return scoring_bin, scoring_multi


def run_baseline_models(x, meta, n_splits=5):
    y_bin, y_multi = get_targets(meta)
    pipeline = build_baseline_pipeline()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring_bin, scoring_multi = baseline_scoring()

    results_multi = cross_validate(pipeline, x, y_multi, cv=cv, scoring=scoring_multi, n_jobs=-1)
    results_bin = cross_validate(pipeline, x, y_bin, cv=cv, scoring=scoring_bin, n_jobs=-1)
    return results_bin, results_multi


def cross_validate_summary(results):
    rows = []
    for key, values in results.items():
        if not key.startswith("test_"):
            continue
        metric = key.replace("test_", "")
        rows.append(
            {
                "metric": metric,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
        )
    return pd.DataFrame(rows).sort_values("metric").reset_index(drop=True)


# ── Step 6: SMOTE + Permutation Feature Importance ──────────────────────────

def load_signature_stability(path=SIGNATURE_STABILITY_PATH):
    """Load the signature stability table from Step 5."""
    if path.exists():
        return pd.read_csv(path)
    raise FileNotFoundError(
        "Signature stability table not found. Run 05_signature_extraction.ipynb first."
    )


def build_smote_pipeline():
    """Build an imblearn Pipeline: Imputer -> Scaler -> SMOTE -> LogisticRegression."""
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    return ImbPipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", LogisticRegression(solver="saga", max_iter=8000, random_state=RANDOM_STATE)),
    ])


def compute_pfi(model, X_test, y_test, scoring="balanced_accuracy", n_repeats=30):
    """Compute Permutation Feature Importance and return a ranked DataFrame."""
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats, random_state=RANDOM_STATE,
        scoring=scoring, n_jobs=-1,
    )
    return pd.DataFrame({
        "gene": X_test.columns if hasattr(X_test, "columns") else list(range(X_test.shape[1])),
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)


# ── Step 7: Synthetic Data Generation (SMOTE) ───────────────────────────────

def generate_smote_samples(X, y):
    """Apply SMOTE to generate balanced synthetic dataset. Returns (X_resampled, y_resampled)."""
    from imblearn.over_sampling import SMOTE

    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    X_imp = pd.DataFrame(imp.fit_transform(X), index=X.index, columns=X.columns)
    X_sc = pd.DataFrame(sc.fit_transform(X_imp), index=X_imp.index, columns=X_imp.columns)

    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_sc, y)
    return X_res, y_res, sc, imp


# ── Step 8: XAI Method Comparison ────────────────────────────────────────────

def compute_shap_importance(model, X_scaled):
    """Compute mean absolute SHAP values across all classes. Returns array of shape (n_features,)."""
    import shap

    explainer = shap.LinearExplainer(model, X_scaled)
    shap_values = explainer.shap_values(X_scaled)

    if isinstance(shap_values, list):
        shap_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    elif shap_values.ndim == 3:
        shap_abs = np.abs(shap_values).mean(axis=2)
    else:
        shap_abs = np.abs(shap_values)
    return shap_abs.mean(axis=0).ravel()


def compute_lime_importance(model, X_scaled, feature_names, n_samples=50):
    """Compute aggregated LIME importance over a sample of instances."""
    import lime.lime_tabular

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_scaled.values if hasattr(X_scaled, "values") else X_scaled,
        feature_names=feature_names,
        class_names=model.classes_.tolist(),
        mode="classification",
        random_state=RANDOM_STATE,
    )
    np.random.seed(RANDOM_STATE)
    data = X_scaled.values if hasattr(X_scaled, "values") else X_scaled
    sample_idx = np.random.choice(len(data), size=min(n_samples, len(data)), replace=False)
    importances = np.zeros(len(feature_names))

    for idx in sample_idx:
        exp = explainer.explain_instance(
            data[idx], model.predict_proba,
            num_features=len(feature_names), top_labels=1,
        )
        label = list(exp.local_exp.keys())[0]
        for feat_idx, weight in exp.local_exp[label]:
            importances[feat_idx] += abs(weight)
    return importances / len(sample_idx)


def compute_ablation_importance(model, X_scaled, y):
    """Compute feature ablation importance (zero-out each feature)."""
    from sklearn.metrics import balanced_accuracy_score

    baseline = balanced_accuracy_score(y, model.predict(X_scaled))
    scores = []
    for i in range(X_scaled.shape[1]):
        X_abl = X_scaled.copy()
        if hasattr(X_abl, "iloc"):
            X_abl.iloc[:, i] = 0
        else:
            X_abl[:, i] = 0
        scores.append(baseline - balanced_accuracy_score(y, model.predict(X_abl)))
    return np.array(scores)


def run_xai_comparison(model, X_scaled, y, feature_names, top_k=20, lime_samples=50):
    """Run all 5 XAI methods and return rankings + consensus analysis."""
    from sklearn.inspection import permutation_importance

    # PFI
    pfi_res = permutation_importance(
        model, X_scaled, y,
        n_repeats=30, random_state=RANDOM_STATE,
        scoring="balanced_accuracy", n_jobs=-1,
    )
    r_pfi = pd.DataFrame({"gene": feature_names, "score": pfi_res.importances_mean})

    # SHAP
    shap_scores = compute_shap_importance(model, X_scaled)
    r_shap = pd.DataFrame({"gene": feature_names, "score": shap_scores[:len(feature_names)]})

    # Coefficients
    coef_scores = np.abs(model.coef_).mean(axis=0)
    r_coef = pd.DataFrame({"gene": feature_names, "score": coef_scores})

    # LIME
    lime_scores = compute_lime_importance(model, X_scaled, feature_names, n_samples=lime_samples)
    r_lime = pd.DataFrame({"gene": feature_names, "score": lime_scores})

    # Ablation
    abl_scores = compute_ablation_importance(model, X_scaled, y)
    r_abl = pd.DataFrame({"gene": feature_names, "score": abl_scores})

    rankings = {
        "PFI": r_pfi.sort_values("score", ascending=False).reset_index(drop=True),
        "SHAP": r_shap.sort_values("score", ascending=False).reset_index(drop=True),
        "Coef": r_coef.sort_values("score", ascending=False).reset_index(drop=True),
        "LIME": r_lime.sort_values("score", ascending=False).reset_index(drop=True),
        "Ablation": r_abl.sort_values("score", ascending=False).reset_index(drop=True),
    }

    top_sets = {name: set(df.head(top_k)["gene"]) for name, df in rankings.items()}
    consensus = set.intersection(*top_sets.values())

    return rankings, top_sets, consensus


# ── Step 9: Robustness Analysis ──────────────────────────────────────────────

def run_robustness_for_k(X, y_multi, stability_df, k_val, random_state=RANDOM_STATE):
    """Run the full SMOTE+PFI+XAI pipeline for a given k value. Returns consensus gene set."""
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import train_test_split

    genes_k = stability_df.head(k_val)["gene"].tolist()
    X_k = X[genes_k].copy()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_k, y_multi, test_size=0.2, stratify=y_multi, random_state=random_state,
    )

    pipe = build_smote_pipeline()
    pipe.fit(X_tr, y_tr)
    test_acc = pipe.score(X_te, y_te)

    # PFI to select top genes
    pfi_df = compute_pfi(pipe, X_te, y_te)
    xai_genes = pfi_df.head(min(50, len(genes_k)))["gene"].tolist()

    # Preprocess for XAI
    X_xai = X[xai_genes].copy()
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    X_imp = pd.DataFrame(imp.fit_transform(X_xai), index=X_xai.index, columns=X_xai.columns)
    X_sc = pd.DataFrame(sc.fit_transform(X_imp), index=X_imp.index, columns=X_imp.columns)

    smote = SMOTE(random_state=random_state)
    X_r, y_r = smote.fit_resample(X_sc, y_multi)
    clf = LogisticRegression(solver="saga", max_iter=8000, random_state=random_state)
    clf.fit(X_r, y_r)

    _, _, consensus = run_xai_comparison(clf, X_sc, y_multi, xai_genes)
    return consensus, test_acc
