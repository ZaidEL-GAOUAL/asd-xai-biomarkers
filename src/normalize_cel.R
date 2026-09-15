# Frozen RMA: each CEL is processed on its own against fixed package reference vectors.
# Entry point: Rscript scripts/pipeline/01_normalize_arrays.R GPL570
# or GPL6244. Optional second argument limits samples; third is worker count.
args <- commandArgs(trailingOnly = TRUE)
platform <- if (length(args)) args[[1]] else stop("Specify GPL570 or GPL6244")
stopifnot(platform %in% c("GPL570", "GPL6244"))
suppressPackageStartupMessages(library(frma))
manifest <- read.csv("phase1_outputs/split_manifest.csv", stringsAsFactors = FALSE)
manifest <- manifest[manifest$platform == platform, ]
if (length(args) > 1) manifest <- head(manifest, as.integer(args[[2]]))
cores <- if (length(args) > 2) as.integer(args[[3]]) else 1L
stopifnot(cores >= 1L, cores <= 4L)
cache <- file.path("data/whole_blood/frma", platform)
dir.create(cache, recursive = TRUE, showWarnings = FALSE)
packages <- c("frma", "affy", "oligo", "hgu133plus2cdf", "pd.hugene.1.0.st.v1",
              "hgu133plus2frmavecs", "hugene.1.0.st.v1frmavecs")
recipe <- c("fRMA; one CEL per call; summarize=robust_weighted_average; default fixed vectors",
            vapply(packages, function(p) paste(p, packageVersion(p)), character(1)))
if (platform == "GPL6244") recipe <- c(recipe, "target=core (transcript-cluster IDs, not exon probesets)")
recipe <- unname(recipe)
recipe_file <- file.path(cache, "recipe.txt")
if (file.exists(recipe_file)) stopifnot(identical(readLines(recipe_file), recipe))
writeLines(recipe, recipe_file)
vecname <- if (platform == "GPL570") "hgu133plus2frmavecs" else "hugene.1.0.st.v1frmavecs"
data(list = vecname, package = vecname)
frozen <- get(vecname)
normalize_one <- function(i) {
    gsm <- manifest$sample_id[[i]]
    outfile <- file.path(cache, paste0(gsm, ".tsv.gz"))
    qcfile <- file.path(cache, paste0(gsm, "_qc.csv"))
    if (file.exists(outfile) && file.exists(qcfile)) return(TRUE)
    cel <- file.path("data/whole_blood/cel", platform, paste0(gsm, ".CEL.gz"))
    stopifnot(file.exists(cel))
    message(sprintf("[%s %d/%d] %s", platform, i, nrow(manifest), gsm))
    raw <- if (platform == "GPL570") affy::ReadAffy(filenames = cel) else
        oligo::read.celfiles(cel, verbose = FALSE)
    expected <- if (platform == "GPL570") "hgu133plus2" else "hugene.1.0.st.v1"
    stopifnot(grepl(expected, Biobase::annotation(raw), fixed = TRUE))
    raw_values <- Biobase::exprs(raw)
    stopifnot(all(is.finite(raw_values)))
    normalized <- frma::frma(raw, summarize = "robust_weighted_average",
                             target = if (platform == "GPL6244") "core" else "probeset",
                             input.vecs = frozen)
    values <- Biobase::exprs(normalized)
    stopifnot(ncol(values) == 1, all(is.finite(values)), !anyDuplicated(rownames(values)))
    # GNUSE is a review flag only; no silent sample deletion based on a threshold.
    # Published medianSE vectors are probeset-level, not GPL6244 core-cluster level.
    gnuse <- if (platform == "GPL570") as.numeric(frma::GNUSE(
        normalized, medianSE = frozen$medianSE, type = "stats")["median", 1]) else NA_real_
    qc <- data.frame(sample_id = gsm, platform = platform, n_features = nrow(values),
                     raw_min = min(raw_values), raw_median = median(raw_values),
                     raw_max = max(raw_values), frma_min = min(values),
                     frma_median = median(values), frma_max = max(values),
                     frma_iqr = IQR(values), gnuse_median = gnuse)
    con <- gzfile(paste0(outfile, ".part"), "wt")
    write.table(data.frame(probe_id = rownames(values), log2_expression = values[, 1]),
                con, sep = "\t", row.names = FALSE, quote = FALSE)
    close(con)
    stopifnot(file.rename(paste0(outfile, ".part"), outfile))
    write.csv(qc, qcfile, row.names = FALSE)
    rm(raw, raw_values, normalized, values)
    gc(verbose = FALSE)
    TRUE
}
results <- parallel::mclapply(seq_len(nrow(manifest)), normalize_one, mc.cores = cores)
stopifnot(all(vapply(results, identical, logical(1), TRUE)))
capture.output(sessionInfo(), file = file.path(cache, "sessionInfo.txt"))
message("Finished ", platform, ": ", nrow(manifest), " selected samples")
