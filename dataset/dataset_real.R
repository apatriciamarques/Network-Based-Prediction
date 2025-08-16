if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}

BiocManager::install(c(
  "TCGAbiolinks",   # Access TCGA data
  "maftools",       # Mutation analysis
  "OmnipathR",      # Protein interaction network
  "dplyr",          # Data manipulation
  "readr"           # Reading CSV files
))

library(OmnipathR)
library(TCGAbiolinks)
library(dplyr)
library(readr)
library(igraph)

print("Let's start our dataset gathering.")

########################### TCGA pan-cancer study of 9 423 tumor exomes ###########################
########################### (comprising all 33 of TCGA projects; Bailey et al., 2018) ###########################

# query <- GDCquery(
#   project = "TCGA-BRCA",
#   data.category = "Simple Nucleotide Variation",
#   data.type = "Masked Somatic Mutation",
#   workflow.type = "Aliquot Ensemble Somatic Variant Merging and Masking"
# )
# GDCdownload(query)
# maf_data <- GDCprepare(query)

########################### InBio Map PPI network ###########################

ppi_network <- import_omnipath_interactions() # curated PPI from OmniPath (Downloaded 81529 interactions.)
# not inbiomap_download() because the domain inbio-discover.com,
# which the inbiomap_download() function in OmnipathR tries to contact
# now redirects to ZS Discovery, which is completely unrelated to the original InBio Map database

########################### -log10 MutSig p-value (Lawrence et al., 2014) ###########################

if (!file.exists("MutSig_results.csv")) {
  stop("MutSig_results.csv not found. Please provide this file.")
}

genes_with_pvalues <- read_csv("MutSig_results.csv")  # with columns: gene, pval

# remove all nodes from the network that cannot be represented with a MutSig P-value
ppi_filtered <- ppi_network %>%
  filter(source_genesymbol %in% genes_with_pvalues$gene,
         target_genesymbol %in% genes_with_pvalues$gene)

g <- graph_from_data_frame(ppi_filtered, directed = FALSE) # igraph network object
g <- delete.vertices(g, degree(g) == 0) # remove isolated nodes

# add features to nodes (-log10 p-values)
V(g)$mutsig_score <- -log10(genes_with_pvalues$pval[match(V(g)$name, genes_with_pvalues$gene)])

########################### CGC data from the COSMIC database (Sondka et al., 2018) ###########################

if (!file.exists("CGC.csv")) {
  stop("CGC.csv not found. Please provide this file.")
}

cgc_genes <- read_csv("CGC.csv")  # with column: gene (723 genes causally implicated in cancer)

V(g)$label <- ifelse(V(g)$name %in% cgc_genes$gene, 1, 0) # 1 if in CGC, 0 otherwise
