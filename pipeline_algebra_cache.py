import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from itertools import groupby
from operator import itemgetter
import os, sys, gc

# ---------- caching / dtype config ----------
DTYPE = np.float32
SAVE_EVERY = 50  # print progress every N nodes

base_dir = r"C:\Users\mpatr\Documents\Thesis\ThesisWork"
cache_dir = os.path.join(base_dir, "mid_files/22082025-2")
os.makedirs(cache_dir, exist_ok=True)

def cp(name):  # cache path
    return os.path.join(cache_dir, name)

def save_npy(path, arr):
    np.save(path, arr)

def load_npy(path, mmap=True):
    return np.load(path, mmap_mode='r' if mmap else None)

def log_progress(i, total, tag=""):
    if (i+1) % SAVE_EVERY == 0 or (i+1) == total:
        pct = 100.0 * (i+1) / total
        print(f"{tag}{i+1}/{total} ({pct:.2f}%)")

# --------------------------------------------------------------------------------------------------------------
# Network Parameters
# --------------------------------------------------------------------------------------------------------------

def input_data(graph_type = "synthetic", show = False):
    ''' 
    Returns parameters from the network.
    '''
    if graph_type == "synthetic":
        # Parameters
        n = 13 # Number of qubits for |j⟩
        m = 2 # Number of qubits for |k⟩

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
            shifts = np.array([1, -1])
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
        nodeDegrees = np.array([len(adj) for adj in adjacencyList])
        # Second-order node degrees (approximation)
        print("Creating the A2...")
        nodeDegreesC1 = nodeDegrees ** 2
        print("Finished.")

        # Feature vector
        p = n  # Precision level
        P = 2 ** p  # Scaling factor
        features = np.linspace(1 - 1/P, 0, nrNodes)  # descending
        featuresNorm = features
        # features = np.array([round(x, 10) for x in np.linspace(1 - 1/P, 0, nrNodes)])
        featuresInt = np.round(featuresNorm * P).astype(int)

        # --- Parameters ---
        thClass = 0.5
        # Class labels: 0 if < thClass, else 1
        classLabels = np.where(featuresNorm < thClass, 0, 1).astype(np.int8)

    if graph_type == "PPI":
        print("Importing dataset (edges, degrees, features)...")
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
        classLabels = np.array([1 if gene in genes_all_set else 0 for gene in featuresGenes])
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

n, nrNodes, m, maxD, s, adjacencyList, nodeDegrees, nodeDegreesC1, mC1, sC1, p, P, featuresNorm, classLabels = input_data()

nodeCheck = 0 # 2145 # 0-based
neighborsCheck = adjacencyList[nodeCheck]
print("\nNode:", nodeCheck)
print("  Degree:", nodeDegrees[nodeCheck])
print("  Neighbors:", neighborsCheck)
print("  Neighbors' Degrees:", nodeDegrees[neighborsCheck])
print("  C1 Degree (sum of neighbors' degrees):", nodeDegreesC1[nodeCheck])

# Complete the graph
def get_r(v_idx, l_idx):
    ''' Input: v_idx, get l_idx.
        Returns 0..nrNodes-1 if valid, otherwise nrNodes '''
    # print(f"v_idx={v_idx}, l_idx={l_idx}, degree={nodeDegrees[v_idx]}")
    if l_idx < nodeDegrees[v_idx]:
        return adjacencyList[v_idx][l_idx]
    else:
        return nrNodes

# --------------------------------------------------------------------------------------------------------------
# Unitary Gates
# --------------------------------------------------------------------------------------------------------------

# --------------------------------------- Hadamard matrices ---------------------------------------
def hadamard(k):
    """Generate a Hadamard matrix of size 2^k."""
    if k == 1:
        return (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]])
    H = hadamard(k-1)
    return (1/np.sqrt(2)) * np.block([[H, H], [H, -H]])

H = hadamard(1)
H2 = hadamard(2)
H3 = hadamard(3)
H4 = hadamard(4)

# Example H10 (careful: large!)
print("Computing H10...")
H10 = hadamard(10)
print("Finished.")

Hl = {1: H, 2: H2, 3: H3, 4: H4, 10: H10}.get(m, None)
Hl = Hl.astype(DTYPE) # cast Hl to float32
if Hl is None:
    raise ValueError("Only m=1..4 or 10 supported")

# Feature rotation matrices
def Wtheta(theta):
    return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                     [np.sin(theta/2),  np.cos(theta/2)]])

def Wval(val):
    sqrt_term = np.sqrt(1 - val**2)
    return np.array([[val, -sqrt_term],
                     [sqrt_term, val]])

print("Wtheta[0.5] =\n", Wtheta(0.5))
print("Wval[0.8] =\n", Wval(0.8))

# --------------------------------------------------------------------------------------------------------------
# OL: Feature Rotation Gate
# --------------------------------------------------------------------------------------------------------------

print("Creating FeatureRotation gate...")

def get_feature_rotation(v, l):
    """
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

print("Node:", nodeCheck)
print("Neighbor:", neighborsCheck[0])
print("Neighbor's FeatureNorm:", featuresNorm[neighborsCheck[0]])
print(get_feature_rotation(nodeCheck + 1, 1))

print("Finished.")

# --------------------------------------------------------------------------------------------------------------
# OK^-1: Feature Rotation Gate
# --------------------------------------------------------------------------------------------------------------

print("Creating DegreeRotation gate...")

def get_degree_rotation(v, c):
    """
    Returns the 2x2 degree rotation matrix for node v and hop index c.
    v, c are 1-based to match Mathematica convention.
    """
    # Adjust for Python 0-based indexing
    v_idx = v - 1
    
    # Determine degree based on hop index
    if c == 1:
        deg = nodeDegrees[v_idx]
    elif c == 2:
        deg = nodeDegreesC1[v_idx]
    else:
        return np.identity(2)
    
    if deg < 1:
        return np.identity(2)
    
    theta = 2 * np.arccos(1 / deg)
    return Wtheta(theta)

print("GetDegreeRotation[2, 1] =\n", get_degree_rotation(2, 1))
print("GetDegreeRotation[2, 2] =\n", get_degree_rotation(2, 2))

print("Finished.")

# --------------------------------------------------------------------------------------------------------------
# QMME Circuit: c=1: Feature Embeddings for (v, l, i)
# First-hop neighborhood: For each (v, l), get the 8 feature vectors (for each i, l) of dimension 2^4 = 16 (corresponding to 4 ancilla rotations).
# --------------------------------------------------------------------------------------------------------------

print("Creating firstHopStates...")

# firstHopStates = []

# for v in range(nrNodes):  # 0-based indexing
#     print(f"v: {v}/{nrNodes-1}")
#     for l_idx in range(s):  # 0-based indexing for neighbors
#         feat_vec = get_feature_rotation(v + 1, l_idx + 1) @ np.array([1, 0])  # column vector
        
#         # Precompute all Kronecker combinations for 4 ancilla rotations
#         vect_list_base = [np.array([1, 0]) for _ in range(4)]
        
#         for i in range(1, 5):  # i = 1 to 4
#             vect_list = vect_list_base.copy()
#             # Replace first i entries with feat_vec
#             for k in range(i):
#                 vect_list[k] = feat_vec
#             # Compute Kronecker product
#             combined_vec = vect_list[0]
#             for vec in vect_list[1:]:
#                 combined_vec = np.kron(combined_vec, vec)

#             firstHopStates.append({
#                 "v": v + 1,  # 1-based to match Mathematica
#                 "i": i,
#                 "l": l_idx + 1,  # 1-based
#                 "u": get_r(v, l_idx) + 1,  # 1-based
#                 "featVec": feat_vec,
#                 "state": combined_vec
#             })

# --------------------------------------------------------------------------------------------------------------
# Stage A (replaces "firstHopStates" + "firstHopSuperposedStates"):
# Build directly the c=1 superposed, degree-rotated vectors per (v, i),
# store as mid-files: mid_files/c1_v{v:05d}.npy with shape (4, s*16*2)
# --------------------------------------------------------------------------------------------------------------

firsthop_done_flag = cp("_FIRSTHOP_DONE.flag")
if not os.path.exists(firsthop_done_flag):
    print("Stage A: Building c=1 per-v superposed blocks (streamed to disk)…")
    for v in range(nrNodes):
        deg_v = nodeDegrees[v]
        stateDim = 16  # 4 ancilla qubits => 2^4

        # degree rotation for c=1
        if deg_v > 0:
            degRotVec = Wtheta(2 * np.arccos(1.0 / deg_v)) @ np.array([1.0, 0.0], dtype=DTYPE)
        else:
            degRotVec = np.array([1.0, 0.0], dtype=DTYPE)

        c1_vi = []  # will hold i=1..4 vectors

        for i in range(1, 5):  # i = 1..4
            # buffer for (l register size s) ⊗ (ancilla state 16)
            combined = np.zeros((s, stateDim), dtype=DTYPE)

            # fill only real neighbors 0..deg_v-1
            for l_idx in range(deg_v):
                u = adjacencyList[v][l_idx]
                feat_val = featuresNorm[u]
                R = Wval(feat_val).astype(DTYPE)                      # 2x2
                vec = R @ np.array([1.0, 0.0], dtype=DTYPE)           # len 2

                # build 4-ancilla tensor with first i entries = vec, rest = |0>
                anc = [vec if k < i else np.array([1.0, 0.0], dtype=DTYPE) for k in range(4)]
                s01 = anc[0]
                for a in anc[1:]:
                    s01 = np.kron(s01, a).astype(DTYPE)               # len 16

                combined[l_idx, :] = s01

            # Hadamard on l register: (Hl @ combined) / sqrt(s)
            had = (Hl @ combined) / np.sqrt(s)                        # (s, 16)
            had = had.reshape(-1).astype(DTYPE)                        # len = s*16

            # tensor with degree rotation (len 2)
            final = np.kron(had, degRotVec).astype(DTYPE)              # len = s*16*2
            c1_vi.append(final)

            del combined, had, final

        c1_vi = np.stack(c1_vi, axis=0)                                # shape (4, s*16*2)
        save_npy(cp(f"c1_v{v:05d}.npy"), c1_vi)
        log_progress(v, nrNodes, tag="  Stage A: ")
        del c1_vi
        gc.collect()

    open(firsthop_done_flag, "w").close()
    print("Stage A: Done (cached).")
else:
    print("Stage A: Using cached c1 files")

print("Finished.")

# Norm of each state should be 1
# for fs in firstHopStates:
#     print(np.linalg.norm(fs["state"]))

# --------------------------------------------------------------------------------------------------------------
# QMME Circuit: c=2: Feature Embeddings for (v, l1, l2, i) 
# Second-hop neighborhood: For each (v, l1, l2), get the 16 feature vectors (for each i, l1, l2) of dimension 2^4 = 16 (corresponding to 4 ancilla rotations).
# --------------------------------------------------------------------------------------------------------------

print("Creating secondHopStates...")

# # Preprocess: group firstHopStates by "v"
# firstHopByV = defaultdict(list)
# for state in firstHopStates:
#     firstHopByV[state["v"]].append(state)

# secondHopStates = []

# for v0 in range(1, nrNodes + 1):  # 1-based indexing
#     print(f"v: {v0}/{nrNodes-1}")
#     for l0 in range(1, s + 1):    # 1-based indexing
#         u0 = get_r(v0 - 1, l0 - 1) + 1 # 1-based adjacency
#         neighborStates = firstHopByV.get(u0, [])
        
#         for state in neighborStates:
#             secondHopStates.append({
#                 "v": v0,
#                 "i": state["i"],
#                 "l0": l0,
#                 "u0": u0,
#                 "l1": state["l"],
#                 "u1": state["u"],
#                 "state": state["state"]
#             })

# --------------------------------------------------------------------------------------------------------------
# Stage B (replaces secondHopStates + secondHopSuperposedStates):
# Build c=2 per (v,i) directly, with 2 Hadamards over l0 and l1.
# Write mid-files: mid_files/c2_v{v:05d}.npy with shape (4, s*s*16*2)
# --------------------------------------------------------------------------------------------------------------

secondhop_done_flag = cp("_SECONDHOP_DONE.flag")
if not os.path.exists(secondhop_done_flag):
    print("Stage B: Building c=2 per-v superposed blocks (streamed to disk)…")

    for v in range(nrNodes):
        deg_v = nodeDegrees[v]
        stateDim = 16

        # degree rotation for c=2
        deg2_v = nodeDegreesC1[v]
        if deg2_v > 0:
            degRotVec2 = Wtheta(2 * np.arccos(1.0 / deg2_v)) @ np.array([1.0, 0.0], dtype=DTYPE)
        else:
            degRotVec2 = np.array([1.0, 0.0], dtype=DTYPE)

        c2_vi = []

        for i in range(1, 5):
            # 3D buffer: (s, s, 16); fill only real (l0,l1)
            buffer = np.zeros((s, s, stateDim), dtype=DTYPE)

            for l0 in range(deg_v):
                u0 = adjacencyList[v][l0]
                deg_u0 = nodeDegrees[u0]
                for l1 in range(deg_u0):
                    u1 = adjacencyList[u0][l1]
                    feat_val = featuresNorm[u1]
                    R = Wval(feat_val).astype(DTYPE)
                    vec = R @ np.array([1.0, 0.0], dtype=DTYPE)

                    anc = [vec if k < i else np.array([1.0, 0.0], dtype=DTYPE) for k in range(4)]
                    s01 = anc[0]
                    for a in anc[1:]:
                        s01 = np.kron(s01, a).astype(DTYPE)           # len 16

                    buffer[l0, l1, :] = s01

            # Apply Hadamard on l0 and l1 axes via two tensordots
            tmp = np.tensordot(Hl, buffer, axes=(1, 0)) / np.sqrt(s)   # (s, s, 16)
            had = np.tensordot(tmp, Hl.T, axes=(1, 0)) / np.sqrt(s)    # (s, s, 16)
            had = had.reshape(-1).astype(DTYPE)                         # len = s*s*16

            final = np.kron(had, degRotVec2).astype(DTYPE)              # len = s*s*16*2
            # normalize like your original c=2
            nrm = np.linalg.norm(final)
            if nrm > 0:
                final /= nrm
            c2_vi.append(final)

            del buffer, tmp, had, final

        c2_vi = np.stack(c2_vi, axis=0)
        save_npy(cp(f"c2_v{v:05d}.npy"), c2_vi)
        log_progress(v, nrNodes, tag="  Stage B: ")
        del c2_vi
        gc.collect()

    open(secondhop_done_flag, "w").close()
    print("Stage B: Done (cached).")
else:
    print("Stage B: Using cached c2 files")

print("Finished.")

# Optional check: Norm of each state
# for sh in secondHopStates:
#     print(np.linalg.norm(sh["state"]))

# --------------------------------------------------------------------------------------------------------------
# QMME Circuit: Superposition from firstHopsStates
# First-hop neighborhood: For each (v, i), get the one feature vectors of dimension 2^5 s = 64 (corresponding to 5 ancilla rotations and the l register).
# --------------------------------------------------------------------------------------------------------------

print("Creating firstHopSuperposedStates...")

# Group firstHopStates by (v, i)
# firstHopStates_sorted = sorted(firstHopStates, key=lambda x: (x["v"], x["i"]))
# groupedByVI = {}
# for key, group in groupby(firstHopStates_sorted, key=lambda x: (x["v"], x["i"])):
#     groupedByVI[key] = list(group)

# firstHopSuperposedStates = []

# for (v, i), states in groupedByVI.items():
#     c = 1
#     stateDim = 16  # dimension of individual featVec states

#     # Combine the states along l (length s * stateDim)
#     combinedState = np.concatenate([state["state"] for state in states])

#     # Hadamard on l register
#     HOnL = np.kron(Hl, np.eye(stateDim))
#     hadState = HOnL @ combinedState / np.sqrt(s)

#     # Degree rotation
#     degRotVec = Wtheta(2 * np.arccos(1 / nodeDegrees[v - 1])) @ np.array([1, 0])  # 1-based v
#     finalState = np.kron(hadState, degRotVec).flatten()

#     firstHopSuperposedStates.append({
#         "c": c,
#         "v": v,
#         "i": i,
#         "stateCombined": combinedState,
#         "stateHadamardL": hadState,
#         "degRotVec": degRotVec,
#         "stateDegreeRotated": finalState
#     })

print("Finished.")

# Optional check: Norm of each state
# for sh in firstHopSuperposedStates:
#     print(np.linalg.norm(sh["stateDegreeRotated"]))

# --------------------------------------------------------------------------------------------------------------
# QMME Circuit: Superposition from secondHopsStates
# Second-hop neighborhood: For each (v, i), get the one feature vectors of dimension 2^5 s^2 = 128 (corresponding to 5 ancilla rotations and the l registers).
# --------------------------------------------------------------------------------------------------------------

print("Creating secondHopSuperposedStates...")

# # Group secondHopStates by (v, i)
# secondHopStates_sorted = sorted(secondHopStates, key=lambda x: (x["v"], x["i"]))
# groupedByVI2 = {}
# for key, group in groupby(secondHopStates_sorted, key=lambda x: (x["v"], x["i"])):
#     groupedByVI2[key] = list(group)

# secondHopSuperposedStates = []

# for (v, i), states in groupedByVI2.items():
#     c = 2

#     stateDim = states[0]["state"].size

#     # Flatten all states in the group (column-major to match Mathematica)
#     combinedState = np.stack([s["state"] for s in states]).flatten(order='F')

#     # Full size according to sparsity level sC1
#     full_length = s**2 * stateDim
#     paddedCombined = np.zeros(full_length)
#     paddedCombined[:combinedState.size] = combinedState

#     # Kronecker product for Hadamard on l0 and l1 registers
#     HOnL = np.kron(np.kron(Hl, Hl), np.eye(stateDim))

#     # Multiply by HOnL using the padded state
#     hadState = (HOnL @ paddedCombined) / s

#     degRotVec = get_degree_rotation(v, c) @ np.array([1, 0])
#     finalState = np.kron(hadState, degRotVec).flatten()
#     finalState /= np.linalg.norm(finalState)

#     secondHopSuperposedStates.append({
#         "c": c,
#         "v": v,
#         "i": i,
#         "stateCombined": combinedState,
#         "stateHadamardL": hadState,
#         "degRotVec": degRotVec,
#         "stateDegreeRotated": finalState
#     })

print("Finished.")

# Norm of each state should be 1
# for sh in secondHopSuperposedStates:
#     print(np.linalg.norm(sh["stateDegreeRotated"]))

# --------------------------------------------------------------------------------------------------------------
# QMME Circuit: Feature Embeddings Subscript[f, x](v,c,i)
# Both neighborhoods: For each (v, c, i), get the one feature vectors of dimension 2^5 s^2 = 128 (padded and unit-length).
# --------------------------------------------------------------------------------------------------------------

# allSuperposedStates = []

# target_len_c1 = 2**5 * s**2  # equivalent to 2^5 * s^2

# # c = 1: pad to target length
# for assoc in firstHopSuperposedStates:
#     state_rot = assoc["stateDegreeRotated"]
#     padded = np.pad(state_rot, (0, target_len_c1 - len(state_rot)))  # pad with zeros
#     # optionally normalize to unit norm
#     padded /= np.linalg.norm(padded)
    
#     assoc_copy = assoc.copy()
#     assoc_copy["statePadded"] = padded
#     allSuperposedStates.append(assoc_copy)

# for assoc in secondHopSuperposedStates:
#     assoc_copy = assoc.copy()
#     assoc_copy["statePadded"] = assoc_copy["stateDegreeRotated"]
#     allSuperposedStates.append(assoc_copy)

# # for assoc in filter(lambda a: a["v"] == 2, allSuperposedStates):
# #     print({
# #         "c": assoc["c"],
# #         "i": assoc["i"],
# #         "stateLengthPadded": len(assoc["statePadded"]) # 512
# #     })

# # for assoc in allSuperposedStates:
# #     print(np.linalg.norm(assoc["statePadded"]))

print("Finished.")

# --------------------------------------------------------------------------------------------------------------
# QMME Circuit: Feature Embeddings Subscript[f, x](v)
# Both neighborhoods: For each v, get the one feature vectors of dimension 2^(5+3) s^2 = 128x8=1024 (unit-length).
# --------------------------------------------------------------------------------------------------------------

# # Group all states by v
# vectorsByV = defaultdict(list)
# for assoc in allSuperposedStates:
#     vectorsByV[assoc["v"]].append(assoc)

# print("Creating superLongVectorsByV...")
# superLongVectorsByV = {}

# for v, assocList in vectorsByV.items():
#     # Create a lookup {(c,i): assoc}
#     lookup = {(assoc["c"], assoc["i"]): assoc for assoc in assocList}

#     concatenated = []

#     for c in [1, 2]:
#         for i in range(1, 5):  # i = 1..4
#             if (c, i) in lookup:
#                 vec = lookup[(c, i)]["statePadded"] / np.sqrt(8)
#                 concatenated.append(vec)
#             else:
#                 # skip if missing (Mathematica uses empty list)
#                 pass

#     # flatten into a single long vector
#     superLongVectorsByV[v] = np.concatenate(concatenated)

# # for v, vec in superLongVectorsByV.items():
# #     print(f"v={v}, norm={np.linalg.norm(vec)}")
# #     print(f"v={v}, length={vec.shape[0]}") # 4096

# --------------------------------------------------------------------------------------------------------------
# Stage C: Assemble final per-v vector by concatenating c=1 (i=1..4) and c=2 (i=1..4)
# Write mid-files: final_v{v:05d}.npy and an index file final_index.npy
# --------------------------------------------------------------------------------------------------------------

final_done_flag = cp("_FINAL_VECTORS_DONE.flag")

if not os.path.exists(final_done_flag):
    print("Stage C: Assembling final per-v vectors…")
    final_paths = []
    for v in range(nrNodes):
        c1 = load_npy(cp(f"c1_v{v:05d}.npy"))
        c2 = load_npy(cp(f"c2_v{v:05d}.npy"))
        pieces = [c1[0], c1[1], c1[2], c1[3], c2[0], c2[1], c2[2], c2[3]]
        final_v = np.concatenate(pieces).astype(DTYPE)
        outp = cp(f"final_v{v:05d}.npy")
        save_npy(outp, final_v)
        final_paths.append(outp)
        log_progress(v, nrNodes, tag="  Stage C: ")
        del c1, c2, final_v
        gc.collect()

    np.save(cp("final_index.npy"), np.array(final_paths, dtype=object))
    open(final_done_flag, "w").close()
    print("Stage C: Done (cached).")
else:
    print("Stage C: Using cached final vectors")

print("Finished.")

# --------------------------------------------------------------------------------------------------------------
# Swap-Test Classifier
# Expectation value of the product of two Pauli-Z operators acting on the ancilla qubits of the quantum classifier's state.
# It measures their correlation and captures the similarity between the test data and training data. 
# Kernel Matrix on Training Data
# --------------------------------------------------------------------------------------------------------------

# trainVecs = np.array(list(superLongVectorsByV.values()))
# # trainVecs = trainVecs / np.linalg.norm(trainVecs, axis=1, keepdims=True) # Normalize vectors (not needed, if done correctly)

# nPower = 1
# nrNodes = len(trainVecs)

# print("Creating Kernel Matrix Table...")
# # Compute kernel matrix: |<v_i|v_j>|^(2 * nPower)
# kernelMatrix = np.abs(trainVecs @ trainVecs.T) ** (2 * nPower)

# print(f"Kernel matrix shape: {kernelMatrix.shape}")  # should be (nrNodes, nrNodes)

# print("Creating Kernel Matrix Heatmap...")
# plt.figure(figsize=(10, 8))
# sns.heatmap(kernelMatrix, cmap="coolwarm", cbar_kws={'label': 'Kernel Value'})
# plt.title("Kernel Matrix Heatmap")
# plt.xlabel("Training Vector Index")
# plt.ylabel("Training Vector Index")
# plt.show()

# # Weight vector (uniform)
# nrNodes = len(featuresNorm)
# wV = np.ones(nrNodes) / nrNodes

# print("featuresNorm:", featuresNorm)
# print("Class labels:", classLabels)

# # Train/Test setup
# trainLabels = classLabels
# testIsTrain = True

# # --- Expectation Values ---
# expectationValsAll = []

# if testIsTrain:
#     for v in range(nrNodes):
#         val = np.sum(((-1) ** trainLabels) * wV * kernelMatrix[v])
#         expectationValsAll.append(val)
# else:
#     for v in range(nrNodes):
#         idxs = np.delete(np.arange(nrNodes), v)  # exclude self
#         val = np.sum(((-1) ** trainLabels[idxs]) * wV[idxs] * kernelMatrix[v, idxs])
#         expectationValsAll.append(val)

# expectationValsAll = np.array(expectationValsAll)

# # --- Predicted Labels ---
# predictedLabels = 0.5 * (1 - np.sign(expectationValsAll))
# predictedLabels = predictedLabels.astype(int)

# # --- Results Table ---
# results = []
# for v in range(nrNodes):
#     results.append({
#         "Node": v,
#         "ExpectationValue": expectationValsAll[v],
#         "TrueLabel": trainLabels[v],
#         "PredictedLabel": predictedLabels[v],
#         "Correct": predictedLabels[v] == trainLabels[v]
#     })

# # --- Bar Chart Visualization ---
# showLabels = True
# colors = ['green' if r['Correct'] else 'red' for r in results]

# plt.figure(figsize=(12,6))
# bars = plt.bar(range(nrNodes), expectationValsAll, color=colors)

# if showLabels:
#     for bar, r in zip(bars, results):
#         plt.text(bar.get_x() + bar.get_width()/2,
#                  bar.get_height(),
#                  f"T:{r['TrueLabel']}\nP:{r['PredictedLabel']}",
#                  ha='center', va='bottom', fontsize=9)

# plt.xlabel("Node Index")
# plt.ylabel("Expectation Value")
# plt.title(f"Expectation Values per Node with Predictions (N={nrNodes})")
# plt.show()

# # --- Metrics ---
# trueLabels = np.array([r['TrueLabel'] for r in results])
# predLabels = np.array([r['PredictedLabel'] for r in results])

# posClass = 1
# negClass = 0

# tp = np.sum((predLabels == posClass) & (trueLabels == posClass))
# fp = np.sum((predLabels == posClass) & (trueLabels == negClass))
# fn = np.sum((predLabels == negClass) & (trueLabels == posClass))

# accuracy = np.mean(predLabels == trueLabels)
# precision = tp / (tp + fp) if (tp + fp) != 0 else 0
# recall = tp / (tp + fn) if (tp + fn) != 0 else 0
# f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

# # --- Print Metrics ---
# metrics = {
#     "Accuracy": accuracy,
#     "Precision": precision,
#     "Recall": recall,
#     "F1 Score": f1
# }

# print("\nMetrics:")
# for k, v in metrics.items():
#     print(f"{k}: {v:.4f}")

# --------------------------------------------------------------------------------------------------------------
# Swap-Test Classifier (streamed)
# --------------------------------------------------------------------------------------------------------------

final_paths = np.load(cp("final_index.npy"), allow_pickle=True)

print("Print final_paths: ", final_paths)

N = len(final_paths)

nPower = 1
thClass = 0.5

# labels from your features
wV = np.ones(N, dtype=DTYPE) / N

def expectation_row(v_idx, batch=128):
    """Compute expectation value for one v without building full K."""
    v_vec = load_npy(final_paths[v_idx]).astype(DTYPE, copy=False)
    total = 0.0
    for start in range(0, N, batch):
        stop = min(N, start+batch)
        M = np.stack([load_npy(final_paths[j]).astype(DTYPE, copy=False) for j in range(start, stop)], axis=0)
        sims = np.abs(M @ v_vec) ** (2 * nPower)          # (stop-start,)
        total += np.sum(((-1) ** classLabels[start:stop]).astype(DTYPE) * wV[start:stop] * sims.astype(DTYPE))
        del M, sims
    return float(total)

print("Kernel: computing expectation values (streaming)…")
expectationValsAll = np.zeros(N, dtype=DTYPE)
for v in range(N):
    expectationValsAll[v] = expectation_row(v, batch=128)
    log_progress(v, N, tag="  Kernel: ")
print("Kernel: done.")

predictedLabels = (0.5 * (1 - np.sign(expectationValsAll)) > 0.5).astype(np.int8)
trueLabels = classLabels
print("sizePredicted: ", predictedLabels)
print("sizeTrue: ", trueLabels)

tp = np.sum((predictedLabels == 1) & (trueLabels == 1))
fp = np.sum((predictedLabels == 1) & (trueLabels == 0))
fn = np.sum((predictedLabels == 0) & (trueLabels == 1))
accuracy  = np.mean(predictedLabels == trueLabels)
precision = tp / (tp + fp) if (tp + fp) else 0.0
recall    = tp / (tp + fn) if (tp + fn) else 0.0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

print("\nMetrics:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")