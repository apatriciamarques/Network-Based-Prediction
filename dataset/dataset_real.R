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
library(ggplot2)

print("Let's start our dataset gathering.")

# ---------------------------------------------------------
# Load OmniPath PPI network
# ---------------------------------------------------------

ppi_network <- import_omnipath_interactions() # 81529 interactions
write.csv(ppi_network, "OmniPath_PPI.csv", row.names = FALSE)

# Gene-gene edge list
gene_edges <- ppi_network %>%
  select(source_genesymbol, target_genesymbol) %>%
  filter(!is.na(source_genesymbol) & !is.na(target_genesymbol)) %>%
  mutate(
    gene1 = pmin(source_genesymbol, target_genesymbol),  
    gene2 = pmax(source_genesymbol, target_genesymbol)
  ) %>%
  select(gene1, gene2) %>%
  distinct()

# Gene degrees
gene_degrees <- c(gene_edges$gene1, gene_edges$gene2) %>%
  table() %>%
  as.data.frame() %>%
  rename(gene = '.', Degree = 'Freq') %>%
  arrange(desc(Degree))

# ------------------------------------------------------------------------------
# -log10 MutSig p-value (Lawrence et al., 2014)
# GitHub/NetSig/Input: Q and P values from the Lawrence et al. 2014 paper.
# ------------------------------------------------------------------------------

mut_sig <- read.delim("pvals.txt", header = TRUE, stringsAsFactors = FALSE) %>%
  select(gene, pmax) %>%
  rename(pval = pmax) %>%
  mutate(mutsig_score = -log10(pval))

# ------------------------------------------------------------------------------
# Filter PPI network to genes with MutSig p-values
# ------------------------------------------------------------------------------

# 1. Filter MutSig genes by those in the network
mut_sig_filtered <- mut_sig %>%
  filter(gene %in% unique(c(gene_edges$gene1, gene_edges$gene2)))

# 2. Keep only network edges where both ends are MutSig genes
gene_edges_filtered <- gene_edges %>%
  filter(gene1 %in% mut_sig_filtered$gene & gene2 %in% mut_sig_filtered$gene)

# 3. Recompute degrees on filtered network
gene_degrees_filtered <- c(gene_edges_filtered$gene1, gene_edges_filtered$gene2) %>%
  table() %>%
  as.data.frame() %>%
  rename(gene = '.', Degree = 'Freq') %>%
  arrange(desc(Degree))

# --------------------------
# Add compact indices
# --------------------------

all_genes_filtered <- data.frame(gene = sort(unique(c(gene_edges_filtered$gene1, gene_edges_filtered$gene2))))
all_genes_filtered$index <- seq_len(nrow(all_genes_filtered))

gene_edges_filtered <- gene_edges_filtered %>%
  left_join(all_genes_filtered, by=c("gene1"="gene")) %>% rename(index1=index) %>%
  left_join(all_genes_filtered, by=c("gene2"="gene")) %>% rename(index2=index) %>%
  select(index1, gene1, index2, gene2)  %>%
  arrange(index1, index2)

gene_degrees_filtered <- gene_degrees_filtered %>% left_join(all_genes_filtered, by="gene") %>%
  select(index, gene, Degree) %>% arrange(index)

mut_sig_filtered <- mut_sig_filtered %>% left_join(all_genes_filtered, by="gene") %>%
  select(index, gene, pval, mutsig_score) %>% arrange(index)

write_csv(mut_sig_filtered, "MutSig_gene_pvalues_filtered_with_index.csv")
write_csv(gene_edges_filtered, "OmniPath_gene_edges_filtered_with_index.csv")
write_csv(gene_degrees_filtered, "OmniPath_gene_degrees_filtered_with_index.csv")

# ------------------------------------------
# Create igraph object from filtered network
# Add vertex attributes
# ------------------------------------------

vertices_df <- all_genes_filtered %>% rename(name = gene)
g_filtered <- graph_from_data_frame(
  d = gene_edges_filtered %>% select(gene1, gene2),
  directed = FALSE,
  vertices = vertices_df
)

V(g_filtered)$index <- vertices_df$index[match(V(g_filtered)$name, vertices_df$name)]
V(g_filtered)$mutsig_score <- mut_sig_filtered$mutsig_score[match(V(g_filtered)$name, mut_sig_filtered$gene)]

# ------------------------------------------------------------------------------
# Degree distribution (before and after filtering)
# ------------------------------------------------------------------------------

p_before <- ggplot(gene_degrees, aes(x = Degree)) +
  geom_histogram(binwidth = 0.1, fill = "steelblue", color = "white") +
  theme_minimal() +
  labs(title = "Histogram of Gene Degrees (Before Filtering)",
       x = "log10 Degree (Number of connections)", y = "Number of genes") +
  scale_x_log10()
ggsave("Degree_histogram_before.png", p_before, width = 7, height = 5, dpi = 300)
cat("s BEFORE:", max(gene_degrees$Degree),
    "N:", nrow(gene_degrees),
    "interactions:", nrow(gene_edges), "\n") # s=641, N=8352 (2^13=8192), 79940 interactions

p_after <- ggplot(gene_degrees_filtered, aes(x = Degree)) +
  geom_histogram(binwidth = 0.1, fill = "darkorange", color = "white") +
  theme_minimal() +
  labs(title = "Histogram of Gene Degrees (After Filtering)",
       x = "log10 Degree (Number of connections)", y = "Number of genes") +
  scale_x_log10()
ggsave("Degree_histogram_after.png", p_after, width = 7, height = 5, dpi = 300)
cat("s AFTER:", max(gene_degrees_filtered$Degree),
    "N:", nrow(gene_degrees_filtered),
    "interactions:", nrow(gene_edges_filtered), "\n") # s=601, N=6850, 27075 interactions

# ------------------------------------------------------------------------------
# CGC data from the COSMIC database (Sondka et al., 2018)
# Cancer Gene Census (CGC) - Tier 1 and 2 - 17/08/2025
# genes which contain mutations that have been causally implicated in cancer
# and explain how dysfunction of these genes drives cancer.
# ------------------------------------------------------------------------------

if (!file.exists("Census_all.csv")) {
  stop("Census_all.csv not found. Please provide this file.")
}

cgc_genes <- read_csv("Census_all.csv") %>%
  rename(gene = `Gene Symbol`)

V(g_filtered)$label <- ifelse(V(g_filtered)$name %in% cgc_genes$gene, 1, 0) # 1 if in CGC, 0 otherwise

sum(V(g_filtered)$label == 1)   # 590 CGC genes in filtered network
sum(V(g_filtered)$label == 0)   # 6260 non-CGC genes

# ---------------------------------------------------------
# Build filtered_nodes (feature, degree, label)
# ---------------------------------------------------------

reduced_x_k_y <- all_genes_filtered %>%
  left_join(mut_sig_filtered %>% select(index, gene, mutsig_score), by = c("index", "gene")) %>%
  left_join(gene_degrees_filtered %>% select(index, Degree), by = "index") %>%
  mutate(
    label = ifelse(gene %in% cgc_genes$gene, 1, 0)
  ) %>%
  select(index, gene, label, mutsig_score, Degree)

# Save to CSV
write_csv(reduced_x_k_y, "filtered_nodes.csv")

cat("✅ rfiltered_nodes created with", nrow(reduced_x_k_y), "genes\n")
