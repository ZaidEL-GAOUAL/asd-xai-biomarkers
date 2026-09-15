# Run in the project-local R 4.5 environment. No model packages are installed.
options(repos = c(CRAN = "https://cloud.r-project.org"), timeout = 1200,
        Ncpus = 4)
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install(version = "3.22", ask = FALSE, update = FALSE)
BiocManager::install(c("frma", "affy", "oligo", "hgu133plus2cdf",
                      "pd.hugene.1.0.st.v1", "hgu133plus2frmavecs",
                      "hugene.1.0.st.v1frmavecs"),
                     ask = FALSE, update = FALSE)
needed <- c("frma", "affy", "oligo", "hgu133plus2cdf",
            "pd.hugene.1.0.st.v1", "hgu133plus2frmavecs", "hugene.1.0.st.v1frmavecs")
stopifnot(all(vapply(needed, requireNamespace, logical(1), quietly = TRUE)))
sessionInfo()
