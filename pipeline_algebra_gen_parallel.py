import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from itertools import groupby
from operator import itemgetter
from joblib import Parallel, delayed
import os
os.environ["OMP_NUM_THREADS"] = "1"        # generic OpenMP
os.environ["OPENBLAS_NUM_THREADS"] = "1"   # OpenBLAS
os.environ["MKL_NUM_THREADS"] = "1"        # Intel MKL
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # Apple vecLib (macOS)
os.environ["NUMEXPR_NUM_THREADS"] = "1"    # numexpr
# ---------- caching / dtype config ----------
DTYPE = np.float32
n_jobs = 1
nodeCheck = 0 # 0-based
show = False # True
graph_type = "synthetic" # "PPI" # 
KERNEL_FILE = f"{graph_type}_kernel_matrix.npy"

def input_data(graph_type = "synthetic", n = 7, m = 2, show = False):
    ''' 
    n = 7 # Number of qubits for |v⟩ # 13 # 7
    m = 2 # Number of qubits for |l⟩ # 10 # 7
    Returns parameters from the network.
    graph_type = "synthetic" or "PPI"
    '''
    if graph_type == "synthetic":
        # Derived quantities
        nrNodes = 2 ** n  # Number of nodes
        s = 2 ** m        # Sparsity level (2**m)
        maxD = s          # Maximum degree 
        print("Number of Nodes:", nrNodes)

        # Initialize adjacency matrix
        print("Creating the Adjacency Matrix and List...")
        A = np.zeros((nrNodes, nrNodes), dtype=int)

        # Circulant graph shifts
        if s <= 2:
            shifts = np.array([1, -1], dtype=DTYPE)
        else:
            half_s = s // 2
            shifts = np.concatenate([np.arange(1, half_s + 1), -np.arange(1, half_s + 1)])

        # Build adjacency list and adjacency matrix
        adjacencyList = []
        for i in range(nrNodes):
            vList = (i + shifts) % nrNodes  # 0-based indexing
            for v in vList:
                # Convert back to 0-based for the adjacency matrix
                A[i, v - 1] = 1
                A[v - 1, i] = 1  # Ensure symmetry
            adjacencyList.append([int(v) for v in vList])

        # --- Make degrees unequal by removing some edges ---
        # for i in [0, nrNodes-1]:  # just first and last nodes as example
        #     if adjacencyList[i]:  # remove one neighbor
        #         v = adjacencyList[i].pop()  # remove last connection
        #         A[i, v-1] = 0
        #         A[v-1, i] = 0

        print("Finished.")

        # Node degrees
        nodeDegrees = np.array([len(adj) for adj in adjacencyList], dtype=DTYPE)
        # Second-order node degrees (approximation)
        print("Creating the A2...")
        nodeDegreesC1 = nodeDegrees ** 2
        print("Finished.")

        # Feature vector
        p = n  # Precision level
        P = 2 ** p  # Scaling factor
        features = np.linspace(1 - 1/P, 0, nrNodes)  # descending
        print("features.min():", features.min())
        print("features.max():", features.max())
        featuresNorm = features
        # featuresNorm = (features - features.min()) / (features.max() - features.min())
        print("features: ", features)
        print("P:", P)
        featuresInt = np.round(featuresNorm * P).astype(int)

        # --- Parameters ---
        thClass = 0.5
        # Class labels: 0 if < thClass, else 1
        classLabels = np.where(featuresNorm < thClass, 0, 1).astype(np.int8)

    if graph_type == "PPI":
        print("\nImporting dataset (edges, degrees, features)...")
        # Paths
        edgesRaw = pd.read_csv(f"dataset/OmniPath_gene_edges_filtered_with_index.csv")
        degreesRaw = pd.read_csv(f"dataset/OmniPath_gene_degrees_filtered_with_index.csv")
        featuresRaw = pd.read_csv(f"dataset/MutSig_gene_pvalues_filtered_with_index.csv")
        genes_all = pd.read_excel(f"dataset/Census_all.xlsx")
        print("Finished.")

        # --- Extracting the degrees ---
        print("Extracting the degrees...")
        nodeDegrees = degreesRaw.iloc[0:, 2].astype(int).to_numpy()  # 2nd row onward, 3rd col
        print(nodeDegrees)
        if show: 
            plt.hist(nodeDegrees, bins=50, log=True)
            plt.xlabel("Degree"); plt.ylabel("Frequency"); plt.title("Degree Distribution")
            plt.show()

        # --- Extracting the features ---
        print("Extracting the features...")
        features = featuresRaw.iloc[0:6850, 3].astype(float).to_numpy()  # 2nd row onward, 4th col
        featuresGenes = featuresRaw.iloc[0:6850, 1].astype(str).to_numpy()  # 2nd column has names
        print(features)
        if show:
            plt.hist(features, bins=100, log=True)
            plt.xlabel("-log10 Mutsig P-values"); plt.ylabel("Frequency")
            plt.title("Distribution of -log10 Mutsig P-values")
            plt.show()

        # --- Create adjacency list ---
        print("Creating the adjacency list...")
        nrNodes = len(nodeDegrees)
        n = int(np.ceil(np.log2(nrNodes)))

        adjacencyList = [[] for _ in range(nrNodes)]
        edges = edgesRaw.iloc[0:27075, [0, 2]].astype(int).to_numpy()

        print("Max index in edges:", edges.max())
        print("Number of nodes:", nrNodes)

        for v, u in edges:
            adjacencyList[v-1].append(int(u-1))  # Convert to 0-based
            adjacencyList[u-1].append(int(v-1))  # 0...nrNodes - 1

        for i in range(nrNodes):
            if nodeDegrees[i] != len(adjacencyList[i]):
                print(f"Mismatch at node {i}: degree file={nodeDegrees[i]}, adjacency={len(adjacencyList[i])}")

        print("Finished.")

        # --- Number of second-hop neighbors (C1) ---
        print("Creating the 'number of second-hop neighbors' table...")
        nodeDegreesC1 = [sum(nodeDegrees[adj]) for adj in adjacencyList]

        # --- Feature scaling ---
        print("Compute scaled features for binary encoding...")
        p = 7
        P = 2**p
        featuresNorm = (features - features.min()) / (features.max() - features.min())
        featuresInt = np.round(featuresNorm * P).astype(int)
        featuresCheck = featuresInt / P * (features.max() - features.min()) + features.min()
        print("Error after quantization:", np.max(np.abs(features - featuresCheck)))

        if show:
            plt.hist(featuresNorm[featuresNorm > 0], bins=50)
            plt.yscale("log"); plt.xlabel("-log10 Mutsig P-values"); plt.ylabel("Frequency")
            plt.title("Distribution of featuresNorm (non-zero only)")
            plt.show()

            plt.hist(featuresInt[featuresInt > 0], bins=50)
            plt.yscale("log"); plt.xlabel("-log10 Mutsig P-values"); plt.ylabel("Frequency")
            plt.title("Distribution of featuresInt (non-zero only)")
            plt.show()

        # --- Compute m_c1 and s_c1 ---
        maxD = np.max(nodeDegrees)
        m = int(np.ceil(np.log2(maxD)))
        s = 2**m

        # --- Import class labels ---
        print("Importing class labels...")
        genes_all_set = set(genes_all['Gene Symbol'].values)  # Set of genes with label 1
        # Assign labels: 1 if gene in Genes_all, else 0
        classLabels = np.array([1 if gene in genes_all_set else 0 for gene in featuresGenes], dtype=DTYPE)
        print("Number of positive labels:", np.sum(classLabels), "/", len(classLabels))
        print("Number of negative labels:", len(classLabels) - np.sum(classLabels), "/", len(classLabels))
        
    # Bits needed to represent max second-order degree
    maxC1 = np.max(nodeDegreesC1)
    mC1 = int(np.ceil(np.log2(maxC1)))
    sC1 = 2 ** mC1 # Sparsity level for c1

    print("  Number of nodes:", nrNodes, ", n =", n, ", N =", 2 ** n)
    print("  Immediate neighbors: max=", maxD, ", m =", m, ", s =", s)
    print("  Second-hop neighbors: max=", maxC1, ", m_c1 =", mC1, ", s_c1 =", sC1)
    print("  Features scaling: maxFeat=", np.max(features), ", p =", p, ", P =", P)

    # --------------------------------------- Dimensions for each register ---------------------------------------
    swapDim = 2**5 * 2**3 * 2**m * 2**m
    fullDim = swapDim * 2**p * 2**mC1 * 2**n * 2**n * 2**n
    print(f"Full dimension : {int(fullDim):,}")
    print(f"Swap dimension: {int(swapDim):,}")
    print(f"Full dim (qubits): {int(np.log2(float(fullDim))):,}")
    print(f"Swap dim (qubits): {int(np.log2(swapDim)):,}")

    # Sanity (potential MemoryError)
    print(f"[Diagnostics] Rough upper bounds (might exhaust RAM):")
    print(f"  firstHopStates entries ~ {int(nrNodes * s * 4):,}")
    print(f"  secondHopStates entries ~ {int(nrNodes * s ** 2 * 4):,}")

    return n, nrNodes, m, maxD, s, adjacencyList, nodeDegrees, nodeDegreesC1, mC1, sC1, p, P, featuresNorm, classLabels

def check_node(nodeCheck):
    ''' nodeCheck = 0 or 2145 or etc. (0-based) '''
    neighborsCheck = adjacencyList[nodeCheck]
    print("\nNode:", nodeCheck)
    print("  Degree:", nodeDegrees[nodeCheck])
    print("  Neighbors:", neighborsCheck)
    print("  Neighbors' Degrees:", nodeDegrees[neighborsCheck])
    print("  C1 Degree (sum of neighbors' degrees):", nodeDegreesC1[nodeCheck])
    return neighborsCheck

def get_r(v_idx, l_idx):
    ''' Complete the graph.
        Input: v_idx, get l_idx.
        Returns 0..nrNodes-1 if valid, otherwise nrNodes '''
    # print(f"v_idx={v_idx}, l_idx={l_idx}, degree={nodeDegrees[v_idx]}")
    if l_idx < nodeDegrees[v_idx]:
        return adjacencyList[v_idx][l_idx]
    else:
        return nrNodes

def hadamard(k):
    """Generate a Hadamard matrix of size 2^k."""
    if k == 1:
        return (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=DTYPE)
    H = hadamard(k-1)
    return (1/np.sqrt(2)) * np.block([[H, H], [H, -H]])

def Wtheta(theta):
    ''' Feature rotation matrix. Input is theta. '''
    return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                     [np.sin(theta/2),  np.cos(theta/2)]], dtype=DTYPE)

def Wval(val):
    ''' Feature rotation matrix. Input is the amplitude (<1) of the zero-state.'''
    sqrt_term = np.sqrt(1 - val**2)
    return np.array([[val, -sqrt_term],
                     [sqrt_term, val]], dtype=DTYPE)

def get_feature_rotation(v, l):
    """
    OL: Feature Rotation Gate
    Returns the Wval matrix for the feature corresponding to adjacency list entry.
    Input: v, l are 1-based indices to match Mathematica.
    """
    v_idx = v - 1
    l_idx = l - 1
    
    if 0 <= v_idx < nrNodes and 0 <= l_idx < maxD: # s:
        u = get_r(v_idx, l_idx)           # 0..nrNodes, where nrNodes = dummy
        if u < nrNodes:                   # if real node
            # print("u: ", u)
            featVal = featuresNorm[u]         # with |featVal| <= 1
            return Wval(featVal)      # return proper feature
    # fallback (dummy u1 or u2)
    return Wval(0)

def check_feat_rotation(nodeCheck, neighborsCheck):
    print("Node:", nodeCheck)
    print("Neighbor:", neighborsCheck[0])
    print("Neighbor's FeatureNorm:", featuresNorm[neighborsCheck[0]])
    print(get_feature_rotation(nodeCheck + 1, 1))

def oracleX_kronecker(feat_vec, i, q=4):
    """
    Equivalent to OX as in the paper.
    Build a Kronecker product of length n where:
    - first i slots = feat_vec
    - remaining slots = [1,0]
    """

    final_len = 2 ** q
    result = np.zeros(final_len, dtype=DTYPE)

    # iterate over indices 0..15
    for idx in range(final_len):
        val = 1.0
        for slot in range(q):
            # which bit of idx corresponds to this slot
            bit = (idx >> (q - 1 - slot)) & 1
            # decide if this slot uses feat_vec or [1,0]
            if slot < i:
                val *= feat_vec[bit]
            else:
                val *= 1.0 if bit == 0 else 0.0
        result[idx] = val

    return result

#--------------------------------------------------------------------------------------
# First-Hop and Second-Hop Embeddings
#--------------------------------------------------------------------------------------

def get_firstHopStates():
    ''' Output: indexed dict: firstHopStates[v][i] = list of states (with l info too) '''
    def process_one(v, l_idx, i):
        print(f"firstHopStates: v: {v}/{nrNodes-1}")
        feat_vec = get_feature_rotation(v + 1, l_idx + 1) @ np.array([1, 0], dtype=DTYPE)
        combined_vec = oracleX_kronecker(feat_vec, i)
        return v + 1, i, l_idx + 1, combined_vec  # return tuple instead of dict

    tasks = [(v, l_idx, i) for v in range(nrNodes + 1) for l_idx in range(s) for i in range(1, 5)] # did + 1 to deal with non-regular graphs
    results = Parallel(n_jobs=-1, prefer="threads")(delayed(process_one)(*args) for args in tasks)

    # Pre-index
    firstHopStates = defaultdict(lambda: defaultdict(list))
    for v, i, l, state in results:
        firstHopStates[v][i].append({"l": l, "state": state})
    return firstHopStates

def generate_node_state(v, c, i, s, firstHopStates=None):
    """
    Generate the first- or second-hop state for node v, for given c,i
    Returns the vector for this block only (no generator sharing needed)
    """
    if c == 1:
        # First-hop state
        states_v_i = firstHopStates.get(v, {}).get(i, [])
        stateDim = 16
        degRotVec = Wtheta(2 * np.arccos(1 / nodeDegrees[v - 1])) @ np.array([1, 0], dtype=DTYPE)
        finalState = np.zeros(s * stateDim * 2, dtype=DTYPE)
        for idx_l, state in enumerate(states_v_i):
            for idx_bit, val in enumerate(state["state"]):
                val_h = val / np.sqrt(s)
                final_idx = (idx_l * stateDim + idx_bit) * 2
                finalState[final_idx:final_idx+2] = val_h * degRotVec
        finalState /= np.linalg.norm(finalState)
        return finalState
    else:
        # Second-hop state
        # Gather first-hop states for neighbors
        state_list = []
        for l0 in range(1, s + 1):
            u0 = get_r(v - 1, l0 - 1) + 1  # neighbor node
            # Collect all first-hop states of neighbor u0 with same i
            states_u0_i = firstHopStates.get(u0, {}).get(i, [])
            for st in states_u0_i:
                state_list.append(st["state"])
        if not state_list:
            return None  # skip if no states (note here!!!)
        
        stateDim = len(state_list[0])
        degRotVec = Wtheta(2 * np.arccos(1 / nodeDegreesC1[v-1])) @ np.array([1, 0], dtype=DTYPE)
        final_len = s**2 * stateDim * 2
        finalState = np.zeros(final_len, dtype=DTYPE)
        for idx_l, vec in enumerate(state_list):
            for idx_bit, val in enumerate(vec):
                val_h = val / s
                final_idx = (idx_l * stateDim + idx_bit) * 2
                finalState[final_idx:final_idx+2] = val_h * degRotVec
        finalState /= np.linalg.norm(finalState)
        return finalState

# -------------------------------------------------------------------------------
# KernelMatrix (parallel, memory-safe)
# -------------------------------------------------------------------------------

def compute_kernel_block_batched(c, i, s, firstHopStates, node_ids, nPower=1, batch_size=None):
    """
    Compute kernel contribution for fixed (c,i) over all nodes using memory-efficient batches.
    Skips nodes with None vectors and prints debug info for missing states.
    """
    # Adaptive batch size
    if batch_size is None:
        batch_size = 500 if c == 1 else 50 # 5  # first-hop small, second-hop huge # works for nrNodes = 100

    print(f"compute_kernel_block_batched: c={c}, i={i}, batch_size={batch_size}")
    n_nodes = len(node_ids)
    K_block = np.zeros((n_nodes, n_nodes), dtype=DTYPE)
    missing_nodes = set()  # track missing nodes to avoid repeated debug prints

    # Process nodes in batches
    for start_i in range(0, n_nodes, batch_size):
        end_i = min(start_i + batch_size, n_nodes)
        batch_i_ids = node_ids[start_i:end_i]

        # Generate batch_i vectors safely
        batch_i_vectors = {}
        for v in batch_i_ids:
            print(f"    Node i: {v}/{nrNodes - 1}") # Takes veryyyy long
            vec = generate_node_state(v, c, i, s, firstHopStates)
            print("     VecNorm: ", np.linalg.norm(vec))
            if vec is None:
                if v not in missing_nodes:
                    print(f"[DEBUG] Node {v} has no states for c={c}, i={i}")
                    missing_nodes.add(v)
            else:
                batch_i_vectors[v] = vec

        for start_j in range(start_i, n_nodes, batch_size):
            end_j = min(start_j + batch_size, n_nodes)
            batch_j_ids = node_ids[start_j:end_j]

            # Generate batch_j vectors safely
            batch_j_vectors = {}
            for v in batch_j_ids:
                print(f"    Node j: {v}/{nrNodes - 1}")
                vec = generate_node_state(v, c, i, s, firstHopStates)
                print("     VecNorm: ", np.linalg.norm(vec))
                if vec is None:
                    if v not in missing_nodes:
                        print(f"[DEBUG] Node {v} has no states for c={c}, i={i}")
                        missing_nodes.add(v)
                else:
                    batch_j_vectors[v] = vec

            # Compute kernel between batches
            for idx_i, v_i in enumerate(batch_i_ids):
                vec_i = batch_i_vectors.get(v_i)
                if vec_i is None:
                    continue
                for idx_j, v_j in enumerate(batch_j_ids):
                    if start_i == start_j and idx_j < idx_i:  # upper triangle only
                        continue
                    vec_j = batch_j_vectors.get(v_j)
                    if vec_j is None:
                        continue
                    dot_val = np.dot(vec_i, vec_j) / np.sqrt(vec_i.size)
                    K_block[start_i + idx_i, start_j + idx_j] = dot_val ** (2 * nPower)
                    K_block[start_j + idx_j, start_i + idx_i] = K_block[start_i + idx_i, start_j + idx_j]

    return K_block

def compute_kernel_matrix_blockwise(s, firstHopStates, nrNodes, nPower=1, n_jobs=n_jobs):
    node_ids = list(range(1, nrNodes+1))
    K_total = np.zeros((nrNodes, nrNodes), dtype=DTYPE)

    # Parallel over (c,i) blocks
    blocks = [(c, i) for c in [1,2] for i in range(1,5)]
    # Run sequential accumulation instead of storing all results
    for K_block in Parallel(n_jobs=n_jobs)(
        delayed(compute_kernel_block_batched)(c, i, s, firstHopStates, node_ids, nPower) 
        for c, i in blocks
    ):
        K_total += K_block   # add one block at a time, not after collecting all

    return K_total, node_ids

# -------------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------------

def plot_kernel_matrix(kernelMatrix, show=True):
    """Plot kernel matrix heatmap."""
    print("Creating Kernel Matrix Heatmap...")
    if show:
        plt.figure(figsize=(10, 8))
        sns.heatmap(kernelMatrix, cmap="coolwarm", cbar_kws={'label': 'Kernel Value'})
        plt.title("Kernel Matrix Heatmap")
        plt.xlabel("Training Vector Index")
        plt.ylabel("Training Vector Index")
        plt.show()

def compute_expectation_values(kernelMatrix, trainLabels, featuresNorm, testIsTrain=True, dtype=np.float32):
    """Compute expectation values based on kernel matrix."""
    print("features:", featuresNorm)
    print("Class labels:", trainLabels)

    nrNodes = len(featuresNorm)
    wV = np.ones(nrNodes, dtype=dtype) / nrNodes
    expectationValsAll = []

    if testIsTrain:
        for v in range(nrNodes):
            val = np.sum(((-1) ** trainLabels) * wV * kernelMatrix[v])
            expectationValsAll.append(val)
    else:
        for v in range(nrNodes):
            idxs = np.delete(np.arange(nrNodes), v)  # leave-one-out
            val = np.sum(((-1) ** trainLabels[idxs]) * wV[idxs] * kernelMatrix[v, idxs])
            expectationValsAll.append(val)

    return np.array(expectationValsAll, dtype=dtype)

def predict_labels(expectationValsAll, dtype=np.float32):
    """Predict labels from expectation values."""
    predictedLabels = 0.5 * (1 - np.sign(expectationValsAll))
    return predictedLabels.astype(int)

def evaluate_predictions(trainLabels, predictedLabels, expectationValsAll, show=True, dtype=np.float32):
    """Evaluate classification performance and plot results."""
    nrNodes = len(trainLabels)
    results = [{
        "Node": v,
        "ExpectationValue": float(expectationValsAll[v]),
        "TrueLabel": int(trainLabels[v]),
        "PredictedLabel": int(predictedLabels[v]),
        "Correct": int(predictedLabels[v] == trainLabels[v])
    } for v in range(nrNodes)]

    if show:
        colors = ['green' if r['Correct'] else 'red' for r in results]
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(nrNodes), expectationValsAll, color=colors)
        for bar, r in zip(bars, results):
            plt.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height(),
                     f"T:{r['TrueLabel']}\nP:{r['PredictedLabel']}",
                     ha='center', va='bottom', fontsize=9)
        plt.xlabel("Node Index")
        plt.ylabel("Expectation Value")
        plt.title(f"Expectation Values per Node with Predictions (N={nrNodes})")
        plt.show()

    trueLabels = np.array([r['TrueLabel'] for r in results], dtype=dtype)
    predLabels = np.array([r['PredictedLabel'] for r in results], dtype=dtype)

    posClass, negClass = 1, 0
    tp = np.sum((predLabels == posClass) & (trueLabels == posClass))
    fp = np.sum((predLabels == posClass) & (trueLabels == negClass))
    fn = np.sum((predLabels == negClass) & (trueLabels == posClass))

    accuracy = np.mean(predLabels == trueLabels)
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    metrics = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1}
    return metrics, results

# -------------------------
# Usage
# -------------------------

n, nrNodes, m, maxD, s, adjacencyList, nodeDegrees, nodeDegreesC1, mC1, sC1, p, P, featuresNorm, classLabels = input_data(graph_type)
# nrNodes = 6850
# s = 2^10
# nrNodes = nrNodes + 1
# nrNodes = 100

if os.path.exists(KERNEL_FILE):
    print(f"Loading precomputed kernel matrix from {KERNEL_FILE}...")
    kernelMatrix = np.load(KERNEL_FILE)
    node_ids = np.arange(kernelMatrix.shape[0])  # fallback if not saved separately
else:
    print("Computing kernel matrix from scratch...")

    neighborsCheck = check_node(nodeCheck)
    check_feat_rotation(nodeCheck, neighborsCheck)

    print("Computing H, H2, H3, H4, H10...")
    H = hadamard(1)
    H2 = hadamard(2)
    H3 = hadamard(3)
    H4 = hadamard(4)
    H10 = hadamard(10)
    Hl = {1: H, 2: H2, 3: H3, 4: H4, 10: H10}.get(m, None)
    Hl = Hl.astype(DTYPE) # cast Hl to float32
    if Hl is None:
        raise ValueError("Only m=1..4 or 10 supported")
    print("Finished.")

    random_factory = None
    print("Creating firstHopStates...")
    firstHopStates = get_firstHopStates()
    print(firstHopStates)
    print("Finished.")

    print("\nGot the data (embeddings). Start classification...") # ISSUE
    kernelMatrix, node_ids = compute_kernel_matrix_blockwise(s, firstHopStates, nrNodes, nPower=1, n_jobs=n_jobs)
    np.save(KERNEL_FILE, kernelMatrix)
    print(f"Kernel matrix saved to {KERNEL_FILE}")

plot_kernel_matrix(kernelMatrix, show=show)
expectationValsAll = compute_expectation_values(kernelMatrix, classLabels, featuresNorm)
predictedLabels = predict_labels(expectationValsAll)
metrics, results = evaluate_predictions(classLabels, predictedLabels, expectationValsAll, show=show)
print("\nMetrics:")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")