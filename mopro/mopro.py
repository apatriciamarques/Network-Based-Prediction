import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

DTYPE = np.float32
KERNEL_FILE = f"mopro_kernel_matrix.npy"
show = False  # plotting toggle

# -------------------------
# Load Data
# -------------------------

def input_data(show=False):
    print("\nImporting dataset (edges, degrees, features)...")
    edgesRaw = pd.read_csv("../dataset/OmniPath_gene_edges_filtered_with_index.csv")
    degreesRaw = pd.read_csv("../dataset/OmniPath_gene_degrees_filtered_with_index.csv")
    featuresRaw = pd.read_csv("../dataset/MutSig_gene_pvalues_filtered_with_index.csv")
    genes_all = pd.read_excel("../dataset/Census_all.xlsx")
    print("Finished.")

    # --- Extract degrees ---
    nodeDegrees = degreesRaw.iloc[:, 2].astype(int).to_numpy()
    nrNodes = len(nodeDegrees)

    # --- Extract features ---
    features = featuresRaw.iloc[:nrNodes, 3].astype(float).to_numpy()
    features = (features - np.mean(features)) / np.std(features) # added
    featuresGenes = featuresRaw.iloc[:nrNodes, 1].astype(str).to_numpy()

    if show:
        plt.hist(features, bins=100, log=True)
        plt.xlabel("-log10 Mutsig P-values"); plt.ylabel("Frequency")
        plt.title("Distribution of features")
        plt.show()

    # --- Build adjacency list ---
    adjacencyList = [[] for _ in range(nrNodes)]
    edges = edgesRaw.iloc[:, [0, 2]].astype(int).to_numpy()
    for v, u in edges:
        adjacencyList[v-1].append(u-1)
        adjacencyList[u-1].append(v-1)

    # --- Labels ---
    genes_all_set = set(genes_all['Gene Symbol'].values)
    classLabels = np.array([1 if g in genes_all_set else 0 for g in featuresGenes], dtype=DTYPE)

    print(f"Nodes: {nrNodes}")
    print(f"Positives: {classLabels.sum()} / {len(classLabels)}")

    return nrNodes, adjacencyList, features, classLabels

# -------------------------
# Moment-Based Embeddings
# -------------------------
def safe_stats(vals):
    """Compute mean, std, skew, kurtosis with NaN protection."""
    if len(vals) < 2 or np.allclose(vals, vals[0]):
        return np.mean(vals), np.std(vals), 0.0, 0.0
    return np.mean(vals), np.std(vals), skew(vals), kurtosis(vals)

def build_node_embeddings(adjacencyList, features, unique_second_neighbors=False):
    """
    Each node gets an 8-dim embedding:
    [mean1, std1, skew1, kurt1, mean2, std2, skew2, kurt2]

    Now each half (1-hop, 2-hop) is normalized to unit norm before concatenation.
    """
    nrNodes = len(adjacencyList)
    embeddings = np.zeros((nrNodes, 8), dtype=np.float32)

    for v in range(nrNodes):
        # 1-hop neighbors
        n1 = adjacencyList[v]
        vals1 = features[n1] if n1 else [0]
        mean1, std1, skew1, kurt1 = safe_stats(vals1)
        vec1 = np.array([mean1, std1, skew1, kurt1], dtype=np.float32)
        norm1 = np.linalg.norm(vec1)
        if norm1 > 0:
            vec1 /= norm1
        else:
            vec1 = np.ones_like(vec1, dtype=np.float32) / np.sqrt(len(vec1))

        # 2-hop neighbors
        if unique_second_neighbors:
            n2 = set()
            for u in n1:
                n2.update(adjacencyList[u])
            n2.discard(v)
            vals2 = features[list(n2)] if n2 else [0]
        else:
            n2 = []
            for u in n1:
                n2.extend(adjacencyList[u])
            vals2 = features[n2] if n2 else [0]

        mean2, std2, skew2, kurt2 = safe_stats(vals2)
        vec2 = np.array([mean2, std2, skew2, kurt2], dtype=np.float32)
        norm2 = np.linalg.norm(vec2)
        if norm2 > 0:
            vec2 /= norm2
        else:
            vec2 = np.ones_like(vec2, dtype=np.float32) / np.sqrt(len(vec2))

        # Concatenate normalized halves
        embeddings[v] = np.concatenate([vec1, vec2])

    return embeddings

# -------------------------
# Kernel Matrix
# -------------------------

def build_kernel_matrix(embeddings, power=2):
    """
    Kernel K_ij = (x_i dot x_j)^power
    """
    dot_matrix = embeddings @ embeddings.T
    kernel = np.power(dot_matrix, power)
    return kernel.astype(DTYPE)

# -------------------------
# Usage
# -------------------------

nrNodes, adjacencyList, features, classLabels = input_data(show)

print("\nComputing moment-based embeddings...")
embeddings = build_node_embeddings(adjacencyList, features)
print("NaNs in embeddings:", np.isnan(embeddings).sum())  # should be 0

#-------------------------------------------------------
# Force embeddings to have norm 1
#-------------------------------------------------------

for v in range(nrNodes):
    vec = embeddings[v]
    norm = np.linalg.norm(vec)
    if norm > 0:
        embeddings[v] = vec / norm
    else:
        # If the vector is all zeros, assign uniform values to get norm 1
        embeddings[v] = np.ones_like(vec, dtype=np.float32) / np.sqrt(len(vec))

#-------------------------------------------------------
# Kernel Matrix
#-------------------------------------------------------

embedding_norms = np.linalg.norm(embeddings, axis=1)
print("Min embedding norm:", embedding_norms.min())
print("Max embedding norm:", embedding_norms.max())
print("Number of zero embeddings:", np.sum(embedding_norms == 0))

print("Building kernel matrix...")
kernelMatrix = build_kernel_matrix(embeddings, power=2)
kernelMatrix = np.nan_to_num(kernelMatrix, nan=0.0)  # final safeguard

np.save(KERNEL_FILE, kernelMatrix)
print(f"Kernel matrix saved to {KERNEL_FILE}")
print("Done.")