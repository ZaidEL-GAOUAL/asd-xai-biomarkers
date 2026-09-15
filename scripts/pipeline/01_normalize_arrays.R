# Run once for GPL570 and once for GPL6244, after installing the R dependencies.
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
stopifnot(length(script_arg) == 1L)
script_path <- normalizePath(sub("^--file=", "", script_arg))
root <- normalizePath(file.path(dirname(script_path), "../.."))
setwd(root)
source(file.path(root, "src/normalize_cel.R"))
