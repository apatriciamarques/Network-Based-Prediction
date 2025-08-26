import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from itertools import groupby
from operator import itemgetter
from joblib import Parallel, delayed
import os
# ---------- caching / dtype config ----------
DTYPE = np.float32
nodeCheck = 0 # 0-based
show = True
graph_type = "PPI" # "synthetic" # 
KERNEL_FILE = f"output/{graph_type}_kernel_matrix.npy"

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
        featuresNorm = features
        featuresInt = np.round(featuresNorm * P).astype(int)

        # --- Parameters ---
        thClass = 0.5
        # Class labels: 0 if < thClass, else 1
        classLabels = np.where(featuresNorm < thClass, 0, 1).astype(np.int8)

    if graph_type == "PPI":
        print("\nImporting dataset (edges, degrees, features)...")
        # Paths
        base_dir = r"C:\Users\mpatr\Documents\Thesis\ThesisWork"
        edgesRaw = pd.read_csv(f"{base_dir}/dataset/OmniPath_gene_edges_filtered_with_index.csv")
        degreesRaw = pd.read_csv(f"{base_dir}/dataset/OmniPath_gene_degrees_filtered_with_index.csv")
        featuresRaw = pd.read_csv(f"{base_dir}/dataset/MutSig_gene_pvalues_filtered_with_index.csv")
        genes_all = pd.read_excel(f"{base_dir}/dataset/Census_all.xlsx")
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
    print(f"[Diagnostics] Rough upper bounds:")
    print(f"  firstHopStates entries ~ {int(nrNodes * s * 4):,}")
    print(f"  secondHopStates entries ~ {int(np.sum(nodeDegreesC1)):,}")
    print(f"  (Storing these as Python dicts with arrays will exhaust RAM)")

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
    # base = np.array([1,0], dtype=DTYPE)
    # vecs = [feat_vec if j < i else base for j in range(n)]

    # result = vecs[0]
    # for vec in vecs[1:]:
    #     result = np.kron(result, vec).astype(DTYPE)

        # length of final vector = 2^q

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

def get_firstHopStates():
    def process_one(v, l_idx, i):
        # print(f"firstHopStates: v: {v}/{nrNodes}")
        feat_vec = get_feature_rotation(v + 1, l_idx + 1) @ np.array([1, 0], dtype=DTYPE)
        combined_vec = oracleX_kronecker(feat_vec, i)
        return {
            "v": v + 1,
            "i": i,
            "l": l_idx + 1,
            "state": combined_vec
        }

    tasks = [(v, l_idx, i) for v in range(nrNodes) for l_idx in range(s) for i in range(1, 5)]
    results = Parallel(n_jobs=-1, prefer="threads")(delayed(process_one)(*args) for args in tasks)
    return results

def process_group(v, i, states):
    # print(f"firstHopSuperposedStates: v: {v}/{nrNodes}")
    s_eff = len(states)
    stateDim = 16  # dim of individual feat_vec

    degRotVec = Wtheta(2 * np.arccos(1 / nodeDegrees[v - 1])) @ np.array([1, 0], dtype=DTYPE)
    finalState = np.zeros(s_eff * stateDim * 2, dtype=DTYPE)

    for idx_l, state in enumerate(states):
        for idx_bit, val in enumerate(state["state"]):
            val_h = val / np.sqrt(s_eff)
            finalState_idx = (idx_l * stateDim + idx_bit) * 2
            finalState[finalState_idx:finalState_idx+2] = val_h * degRotVec

    finalState /= np.linalg.norm(finalState)

    return {
        "c": 1,
        "v": v,
        "i": i,
        "stateDegreeRotated": finalState
    }

def get_firstHopSuperposedStates(firstHopStates, n_jobs=-1):
    firstHopStates_sorted = sorted(firstHopStates, key=lambda x: (x["v"], x["i"]))
    grouped = [(v, i, list(group)) for (v, i), group in groupby(firstHopStates_sorted, key=lambda x: (x["v"], x["i"]))]

    print(f"firstHopSuperposedStates: {len(grouped)} groups to process")

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_group)(v, i, states) for v, i, states in grouped
    )

    return results

def process_secondHop_block(v0, l0, firstHopByV):
    # print(f"secondHopStates: v: {v0}/{nrNodes}")
    u0 = get_r(v0 - 1, l0 - 1) + 1
    results = []
    for state in firstHopByV.get(u0, []):
        results.append({
            "v": v0,
            "i": state["i"],
            "state": state["state"]
        })
    return results

def get_secondHopStates(firstHopStates, n_jobs=-1):
    firstHopByV = defaultdict(list)
    for state in firstHopStates:
        firstHopByV[state["v"]].append(state)

    tasks = [(v0, l0) for v0 in range(1, nrNodes + 1) for l0 in range(1, s + 1)]
    print(f"secondHopStates: {len(tasks)} tasks to process")

    all_results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_secondHop_block)(v0, l0, firstHopByV) for v0, l0 in tasks
    )

    # Flatten the results and yield one by one
    for block in all_results:
        for item in block:
            yield item

def process_secondHopSuperposed_block(v, i, state_list):
    # print(f"secondHopSuperposedStates: v: {v}/{nrNodes}")
    c = 2
    stateDim = state_list[0].size
    s_eff = len(state_list)
    degRotVec = Wtheta(2 * np.arccos(1 / nodeDegreesC1[v-1])) @ np.array([1, 0], dtype=DTYPE)

    final_len = s_eff * stateDim * 2
    finalState = np.zeros(final_len, dtype=DTYPE)

    for idx_l, vec in enumerate(state_list):
        for idx_bit, val in enumerate(vec):
            val_h = val / np.sqrt(s_eff)
            finalState_idx = (idx_l * stateDim + idx_bit) * 2
            finalState[finalState_idx : finalState_idx + 2] = val_h * degRotVec

    finalState /= np.linalg.norm(finalState)

    return {
        "c": c,
        "v": v,
        "i": i,
        "stateDegreeRotated": finalState
    }

def get_secondHopSuperposedStates(firstHopStates, n_jobs=-1):
    """Stream second-hop states, group by (v,i), and superpose in parallel."""

    grouped = defaultdict(list)

    # stream instead of precomputing
    for v, i, state in iter_secondHopStates(firstHopStates):
        grouped[(v, i)].append(state)

    print(f"Processing {len(grouped)} (v,i) groups in parallel...")

    tasks = [(v, i, states) for (v, i), states in grouped.items()]

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_secondHopSuperposed_block)(v, i, states) for v, i, states in tasks
    )

    return results

def iter_secondHopStates(firstHopStates):
    """Generator that yields (v, i, state) triples without exploding into 7M tasks."""
    firstHopByV = defaultdict(list)
    for state in firstHopStates:
        firstHopByV[state["v"]].append(state)

    # iterate over v0,l0 pairs *lazily*
    for v0 in range(1, nrNodes + 1):
        for l0 in range(1, s + 1):
            u0 = get_r(v0 - 1, l0 - 1) + 1
            for state in firstHopByV.get(u0, []):
                yield (v0, state["i"], state["state"])

def get_superLongVectorNode(v, firstHopSuperposedStates, secondHopSuperposedStates):
    """Compute superLongVector for a single node v on-demand."""
    first_hop_v = [assoc for assoc in firstHopSuperposedStates if assoc["v"] == v]
    second_hop_v = [assoc for assoc in secondHopSuperposedStates if assoc["v"] == v]

    block_len = 2**5 * s**2
    target_len = 2 * 4 * block_len

    states_for_v = []

    for assoc in first_hop_v + second_hop_v:
        state_rot = assoc["stateDegreeRotated"]
        padded = np.zeros(block_len, dtype=DTYPE)
        padded[:state_rot.size] = state_rot
        padded /= np.linalg.norm(padded)
        assoc_copy = assoc.copy()
        assoc_copy["statePadded"] = padded
        states_for_v.append(assoc_copy)

    superVec = np.zeros(target_len, dtype=DTYPE)
    idx = 0
    for c in [1, 2]:
        for i in range(1, 5):
            match = next((assoc for assoc in states_for_v if assoc["c"] == c and assoc["i"] == i), None)
            if match:
                superVec[idx:idx + block_len] = match["statePadded"] / np.sqrt(8)
            idx += block_len

    return superVec

def compute_kernel_matrix(firstHopSuperposedStates, secondHopSuperposedStates, nPower=1, dtype=DTYPE, n_jobs=-1):
    """Compute kernel matrix in RAM-safe parallel chunks."""
    node_set = {st["v"] for st in firstHopSuperposedStates} | {st["v"] for st in secondHopSuperposedStates}
    node_ids = sorted(node_set)
    nrNodes = len(node_ids)

    vector_size_gb = (2 * 4 * (2**5 * s**2) * 4) / (1024**3)  # float32
    max_vectors_in_mem = int(8 / vector_size_gb)
    chunk_size = min(max_vectors_in_mem, 10)  # 10 is conservative for 10GB RAM

    print(f"Vector size: {vector_size_gb:.2f} GB | Chunk size: {chunk_size}")

    kernelMatrix = np.zeros((nrNodes, nrNodes), dtype=dtype)
    total_chunks = (nrNodes + chunk_size - 1) // chunk_size

    for i in range(0, nrNodes, chunk_size):
        end_i = min(i + chunk_size, nrNodes)
        print(f"\n=== Chunk {i // chunk_size + 1}/{total_chunks} | Nodes {i}-{end_i-1} ===")

        # Compute super vectors in parallel for this chunk
        chunk_vecs = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(get_superLongVectorNode)(node_ids[idx], firstHopSuperposedStates, secondHopSuperposedStates)
            for idx in range(i, end_i)
        )

        # Intra-chunk kernel values
        for k1, vec1 in enumerate(chunk_vecs):
            for k2 in range(k1, len(chunk_vecs)):
                vec2 = chunk_vecs[k2]
                val = np.abs(np.dot(vec1, vec2)) ** (2 * nPower)
                kernelMatrix[i + k1, i + k2] = val
                kernelMatrix[i + k2, i + k1] = val

        # Inter-chunk kernel values with previous nodes
        for j in range(0, i):
            vec_j = get_superLongVectorNode(node_ids[j], firstHopSuperposedStates, secondHopSuperposedStates)
            for k, vec_i in enumerate(chunk_vecs):
                val = np.abs(np.dot(vec_i, vec_j)) ** (2 * nPower)
                kernelMatrix[i + k, j] = val
                kernelMatrix[j, i + k] = val
            del vec_j

        del chunk_vecs
        print(f"Progress: {min(100, (i + chunk_size)/nrNodes*100):.1f}%")

    print(f"\nKernel matrix computation complete! Shape: {kernelMatrix.shape}")
    return kernelMatrix, node_ids

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

    print("Creating firstHopStates and firstHopSuperposedStates...")
    firstHopStates = get_firstHopStates()
    firstHopSuperposedStates = get_firstHopSuperposedStates(firstHopStates)
    print("Finished.")
    print("Creating secondHopStates and secondHopSuperposedStates...")
    secondHopStatesGen = get_secondHopStates(firstHopStates)
    secondHopSuperposedStates = get_secondHopSuperposedStates(secondHopStatesGen)
    print("Finished.")

    print("\nGot the data (embeddings). Start classification...")
    kernelMatrix, node_ids = compute_kernel_matrix(firstHopSuperposedStates, secondHopSuperposedStates, nPower=1)
    np.save(KERNEL_FILE, kernelMatrix)
    print(f"Kernel matrix saved to {KERNEL_FILE}")

plot_kernel_matrix(kernelMatrix, show=show)
expectationValsAll = compute_expectation_values(kernelMatrix, classLabels, featuresNorm)
predictedLabels = predict_labels(expectationValsAll)
metrics, results = evaluate_predictions(classLabels, predictedLabels, expectationValsAll, show=show)
print("\nMetrics:")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")
