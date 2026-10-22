library(ArchR)
library(Matrix)
library(SummarizedExperiment)
library(GenomicRanges)





addArchRThreads(threads = 40)
addArchRGenome("hg38")

outdir <- "./ArchR_joint_peak_project"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)


input_files <- c(
  A549_N1 = "./fragments/A549_N1_fragments/fragments.tsv.gz",
  A549_C3 = "./fragments/A549_C3_fragments/fragments.tsv.gz"
)





ArrowFiles <- createArrowFiles(
  inputFiles = input_files,
  sampleNames = names(input_files),
  geneAnnotation = getGeneAnnotation(),
  genomeAnnotation = getGenomeAnnotation(),
  minTSS = 4,
  minFrags = 1000,
  addTileMat = TRUE,
  addGeneScoreMat = TRUE
)





proj <- ArchRProject(
  ArrowFiles = ArrowFiles,
  outputDirectory = outdir,
  copyArrows = TRUE
)


pdf(file.path(outdir, "QC_TSS_vs_Frags.pdf"), width = 6, height = 5)
plotGroups(
  ArchRProj = proj,
  groupBy = "Sample",
  colorBy = "cellColData",
  name = "TSSEnrichment",
  plotAs = "ridges"
)
dev.off()





proj <- addIterativeLSI(
  ArchRProj = proj,
  useMatrix = "TileMatrix",
  name = "IterativeLSI",
  iterations = 2,
  clusterParams = list(
    resolution = c(0.2),
    sampleCells = 10000,
    n.start = 10
  ),
  varFeatures = 25000,
  dimsToUse = 1:30
)

proj <- addClusters(
  input = proj,
  reducedDims = "IterativeLSI",
  method = "Seurat",
  name = "Clusters",
  resolution = 0.8
)





proj <- addGroupCoverages(
  ArchRProj = proj,
  groupBy = "Sample"
)

proj <- addReproduciblePeakSet(
  ArchRProj = proj,
  groupBy = "Sample",
  pathToMacs2 = "macs2",
  plot = FALSE
)

proj <- addPeakMatrix(proj)


proj <- saveArchRProject(
  ArchRProj = proj,
  outputDirectory = outdir,
  load = TRUE
)





export_sample_results <- function(proj, sample_name, base_outdir) {
  
  message("Processing sample: ", sample_name)
  
  sample_outdir <- file.path(base_outdir, sample_name)
  dir.create(sample_outdir, recursive = TRUE, showWarnings = FALSE)
  
  
  
  
  
  cells_use <- proj$cellNames[proj$Sample == sample_name]
  
  proj_sub <- subsetArchRProject(
    ArchRProj = proj,
    cells = cells_use,
    outputDirectory = file.path(sample_outdir, paste0(sample_name, "_ArchRProject")),
    dropCells = TRUE,
    force = TRUE
  )
  
  
  
  
  
  
  proj_sub <- addIterativeLSI(
    ArchRProj = proj_sub,
    useMatrix = "TileMatrix",
    name = "IterativeLSI",
    iterations = 2,
    clusterParams = list(
      resolution = c(0.2),
      sampleCells = min(10000, nCells(proj_sub)),
      n.start = 10
    ),
    varFeatures = 25000,
    dimsToUse = 1:30,
    force = TRUE
  )
  
  
  
  
  
  proj_sub <- addCoAccessibility(
    ArchRProj = proj_sub,
    reducedDims = "IterativeLSI",
    maxDist = 1000000
  )
  
  coacc <- getCoAccessibility(
    ArchRProj = proj_sub,
    corCutOff = 0
  )
  
  write.csv(
    as.data.frame(coacc),
    file = file.path(sample_outdir, paste0(sample_name, "_archr_coaccessibility_0_1mb.csv")),
    row.names = FALSE
  )
  
  
  
  
  
  
  peak_matrix <- getMatrixFromProject(
    ArchRProj = proj_sub,
    useMatrix = "PeakMatrix"
  )
  
  mat <- assay(peak_matrix)
  peaks <- rowRanges(peak_matrix)
  barcodes <- colnames(peak_matrix)
  
  peak_names <- paste0(
    as.character(seqnames(peaks)), ":",
    start(peaks), "-",
    end(peaks)
  )
  
  rownames(mat) <- peak_names
  colnames(mat) <- barcodes
  
  
  
  
  
  writeMM(
    mat,
    file = file.path(sample_outdir, paste0(sample_name, "_matrix.mtx"))
  )
  
  peak_bed <- data.frame(
    chrom = as.character(seqnames(peaks)),
    chromStart = start(peaks),
    chromEnd = end(peaks)
  )
  
  write.table(
    peak_bed,
    file = file.path(sample_outdir, paste0(sample_name, "_peaks.bed")),
    sep = "\t",
    row.names = FALSE,
    col.names = FALSE,
    quote = FALSE
  )
  
  write.table(
    barcodes,
    file = file.path(sample_outdir, paste0(sample_name, "_barcodes.tsv")),
    sep = "\t",
    row.names = FALSE,
    col.names = FALSE,
    quote = FALSE
  )
  
  saveArchRProject(
    ArchRProj = proj_sub,
    outputDirectory = file.path(sample_outdir, paste0(sample_name, "_ArchRProject_saved")),
    load = FALSE
  )
  
  message("Finished sample: ", sample_name)
}





result_outdir <- file.path(outdir, "sample_specific_outputs")
dir.create(result_outdir, recursive = TRUE, showWarnings = FALSE)

for (sample_name in names(input_files)) {
  export_sample_results(
    proj = proj,
    sample_name = sample_name,
    base_outdir = result_outdir
  )
}

message("All done!")