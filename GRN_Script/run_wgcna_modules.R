options(repos = c(CRAN = "https://cloud.r-project.org"))

options(stringsAsFactors = FALSE)

quiet_install_cran <- function(pkgs) {
  for (p in pkgs) {
    if (!requireNamespace(p, quietly = TRUE)) {
      message(sprintf("[Install CRAN] %s", p))
      install.packages(p, dependencies = TRUE)
    }
  }
}

quiet_install_bioc <- function(pkgs) {
  if (!requireNamespace("BiocManager", quietly = TRUE)) {
    message("[Install CRAN] BiocManager")
    install.packages("BiocManager")
  }
  for (p in pkgs) {
    if (!requireNamespace(p, quietly = TRUE)) {
      message(sprintf("[Install Bioconductor] %s", p))
      BiocManager::install(p, ask = FALSE, update = FALSE)
    }
  }
}

quiet_install_cran(c("dynamicTreeCut", "fastcluster", "WGCNA"))
quiet_install_bioc(c("impute", "preprocessCore", "GO.db"))

suppressMessages(library(WGCNA))
allowWGCNAThreads()

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript run_wgcna_modules.R <expr.tsv> <out_gene_modules.tsv> [sep] [do_log1p TRUE/FALSE]")
}

expr_file <- args[1]
out_file  <- args[2]
sep       <- ifelse(length(args) >= 3, args[3], "\t")
do_log1p  <- ifelse(length(args) >= 4, args[4], "TRUE")

message(paste("Input :", expr_file))
message(paste("Output:", out_file))
message(paste("Sep   :", sep))
message(paste("log1p :", do_log1p))

dat <- read.table(expr_file, header = TRUE, sep = sep, row.names = 1, check.names = FALSE)
datExpr <- as.data.frame(dat)

colnames(datExpr) <- gsub("^(gene[:：]|Gene[:：]|transcript[:：]|Transcript[:：])", "", colnames(datExpr))

if (toupper(do_log1p) == "TRUE") {
  mat <- as.matrix(datExpr)
  if (min(mat, na.rm = TRUE) < 0) {
    message("[WARN] 检测到负值，跳过 log1p。")
  } else {
    datExpr <- log1p(datExpr)
  }
}

gsg <- goodSamplesGenes(datExpr, verbose = 3)
if (!gsg$allOK) {
  if (sum(!gsg$goodSamples) > 0) {
    message(paste("Removing samples:", paste(rownames(datExpr)[!gsg$goodSamples], collapse = ",")))
  }
  if (sum(!gsg$goodGenes) > 0) {
    message(paste("Removing genes:", paste(colnames(datExpr)[!gsg$goodGenes], collapse = ",")))
  }
  datExpr <- datExpr[gsg$goodSamples, gsg$goodGenes]
}

message(sprintf("After QC: samples=%d genes=%d", nrow(datExpr), ncol(datExpr)))

powers <- c(1:20)
sft <- pickSoftThreshold(datExpr, powerVector = powers, verbose = 5)
fit <- sft$fitIndices

if (is.null(fit) || nrow(fit) == 0) {
  softPower <- 6
} else {
  ok <- fit$SFT.R.sq >= 0.85
  softPower <- if (any(ok)) fit$Power[min(which(ok))] else 6
}
message(paste("Using softPower =", softPower))

net <- blockwiseModules(
  datExpr,
  power = softPower,
  TOMType = "unsigned",
  minModuleSize = 30,
  reassignThreshold = 0,
  mergeCutHeight = 0.25,
  numericLabels = FALSE,
  pamRespectsDendro = FALSE,
  saveTOMs = FALSE,
  verbose = 3
)

moduleColors <- net$colors
genes <- colnames(datExpr)

out <- data.frame(GeneID = genes, Module = moduleColors, stringsAsFactors = FALSE)
write.table(out, file = out_file, sep = "\t", quote = FALSE, row.names = FALSE)

message(paste("Done. Wrote:", out_file))