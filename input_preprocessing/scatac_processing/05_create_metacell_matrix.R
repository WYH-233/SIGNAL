
library(Signac)
library(Seurat)
library(Matrix)
library(RANN)


count_matrix <- read.csv("peak_cell_matrix.csv", 
                         row.names = 1, 
                         header = TRUE, 
                         check.names = FALSE)


head(rownames(count_matrix), 3) 


count_matrix <- as(as.matrix(count_matrix), "dgCMatrix")


atac <- CreateSeuratObject(counts = count_matrix, 
                          assay = "peaks", 
                          project = "scATAC")



atac <- RunTFIDF(atac)


atac <- FindTopFeatures(atac, min.cutoff = 'q0')
atac <- RunSVD(atac)


atac <- RunUMAP(atac, reduction = "lsi", dims = 2:30)


set.seed(123)  
selected_cells <- sample(colnames(atac), size = 256)


lsi_coords <- Embeddings(atac, reduction = "lsi")[, 2:30]  


knn_result <- nn2(data = lsi_coords, 
                  query = lsi_coords[selected_cells, ], 
                  k = 101)  


metacell_matrix <- matrix(0, 
                         nrow = nrow(count_matrix), 
                         ncol = length(selected_cells),
                         dimnames = list(rownames(count_matrix),  
                                       paste0("metacell_", 1:length(selected_cells))))


for (i in seq_along(selected_cells)) {
  
  neighbors <- knn_result$nn.idx[i, -1]
  
  
  neighbor_cells <- rownames(lsi_coords)[neighbors]
  
  
  metacell_matrix[, i] <- rowMeans(count_matrix[, neighbor_cells])
}


metacell_matrix <- as(metacell_matrix, "dgCMatrix")


identical(rownames(metacell_matrix), rownames(count_matrix)) 


write.csv(as.matrix(metacell_matrix), 
          file = "metacell_accessibility_matrix_0_1mb.csv",
          row.names = TRUE)  


cat("元细胞矩阵已生成，包含", nrow(metacell_matrix), "个peaks和", 
    ncol(metacell_matrix), "个元细胞\n")
cat("前3个peak名称:", head(rownames(metacell_matrix), 3), "\n")

