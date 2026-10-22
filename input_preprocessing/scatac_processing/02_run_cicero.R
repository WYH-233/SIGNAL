library(cicero)
library(monocle3)


args <- commandArgs(trailingOnly = TRUE)
sample_name <- if (length(args) >= 1) args[1] else "A549_C3"

message("Running Cicero for sample: ", sample_name)


indata <- Matrix::readMM(paste0(sample_name, "_matrix.mtx"))

indata@x[indata@x > 0] <- 1


barcode_path <- paste0(sample_name, "_barcodes.tsv")
cellinfo <- read.delim(
  barcode_path,
  header = FALSE,
  sep = "\t",
  colClasses = "character",
  quote = "",
  comment.char = "",
  na.strings = NULL,
  blank.lines.skip = FALSE,
  fill = FALSE,
  check.names = FALSE,
  stringsAsFactors = FALSE
)
if (ncol(cellinfo) != 1L) {
  stop("Expected exactly one barcode column in ", barcode_path)
}
barcodes <- cellinfo[[1L]]
if (anyNA(barcodes) || any(!nzchar(trimws(barcodes)))) {
  stop("Missing or empty barcodes in ", barcode_path)
}
if (anyDuplicated(barcodes)) {
  stop("Duplicate barcodes in ", barcode_path)
}
if (length(barcodes) != ncol(indata)) {
  stop(
    "Barcode count (", length(barcodes), ") does not match matrix columns (",
    ncol(indata), ") for ", sample_name
  )
}
cellinfo <- data.frame(cells = barcodes, row.names = barcodes, stringsAsFactors = FALSE)


peakinfo <- read.table(paste0(sample_name, "_peaks.bed"))
names(peakinfo) <- c("chr", "bp1", "bp2")
peakinfo$site_name <- paste(peakinfo$chr, peakinfo$bp1, peakinfo$bp2, sep = "_")
row.names(peakinfo) <- peakinfo$site_name

row.names(indata) <- row.names(peakinfo)
colnames(indata) <- barcodes


input_cds <- suppressWarnings(new_cell_data_set(indata,
                                                 cell_metadata = cellinfo,
                                                 gene_metadata = peakinfo))

input_cds <- monocle3::detect_genes(input_cds)


input_cds <- input_cds[Matrix::rowSums(exprs(input_cds)) != 0, ]
set.seed(2017)
input_cds <- detect_genes(input_cds)
input_cds <- estimate_size_factors(input_cds)
input_cds <- preprocess_cds(input_cds, method = "LSI")
input_cds <- reduce_dimension(input_cds, reduction_method = 'UMAP',
                              preprocess_method = "LSI")
umap_coords <- reducedDims(input_cds)$UMAP
cicero_cds <- make_cicero_cds(input_cds, reduced_coordinates = umap_coords)

human.hg38.genome <- data.frame(
  chromosome = c(
    "chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10",
    "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19",
    "chr20", "chr21", "chr22", "chrX", "chrY"
  ),
  length = c(
    248956422, 242193529, 198295559, 190214555, 181538259, 170805979, 159345973,
    145138636, 138394717, 133797422, 135086622, 133275309, 114364328, 107043718,
    101991189, 90338345, 83257441, 80373285, 58617616, 64444167, 46709983,
    50818468, 156040895, 57227415
  )
)
sample_genome <- subset(human.hg38.genome, chromosome %in% c(paste0("chr", 1:22), "chrX"))


conns <- run_cicero(cicero_cds, sample_genome, sample_num = 100, window = 1000000)


out_dir <- "./cicero_output"
if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}
out_file <- file.path(out_dir, "cicero_connections_1mb.csv")
write.csv(conns, file = out_file, row.names = FALSE)

message("Results saved to: ", out_file)
