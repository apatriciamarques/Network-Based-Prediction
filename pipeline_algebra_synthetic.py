import numpy as np

# --------------------------------------------------------------------------------------------------------------
# Network Parameters
# --------------------------------------------------------------------------------------------------------------

# Parameters
n = 6  # Number of qubits for |j⟩
m = 2  # Number of qubits for |k⟩

# Derived quantities
nrNodes = 2 ** n  # Number of nodes
s = 2 ** m        # Sparsity level
print("Number of Nodes:", nrNodes)

# Initialize adjacency matrix
print("Creating the Adjacency Matrix...")
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
    vList = (i + shifts) % nrNodes + 1  # 1-based indexing to match Mathematica
    for v in vList:
        # Convert back to 0-based for the adjacency matrix
        A[i, v - 1] = 1
        A[v - 1, i] = 1  # Ensure symmetry
    adjacencyList.append(list(vList))

print("Finished.")

# Node degrees
nodeDegrees = np.array([len(adj) for adj in adjacencyList])

# Second-order node degrees (approximation)
print("Creating the A2...")
nodeDegreesC1 = nodeDegrees ** 2
print("Finished.")

# Bits needed to represent max second-order degree
mC1 = int(np.ceil(np.log2(np.max(nodeDegreesC1))))
print("m_c1 (bits for 2nd order degrees):", mC1)

# Sparsity level for c1
sC1 = 2 ** mC1
print("s_c1 (2^m_c1):", sC1)

# Feature vector
p = n  # Precision level
P = 2 ** p  # Scaling factor
features = np.linspace(1 - 1/P, 0, nrNodes)  # descending
# features = np.array([round(x, 10) for x in np.linspace(1 - 1/P, 0, nrNodes)])
featuresInt = np.round(features * P).astype(int)

# Dimensions
swapDim = 2**5 * 2**3 * 2**m * 2**m
fullDim = swapDim * 2**p * 2**mC1 * 2**n * 2**n * 2**n

print("Full dimension:", fullDim)
print("Swap dimension:", swapDim)
print("Full dim (qubits):", int(np.log2(fullDim)))
print("Swap dim (qubits):", int(np.log2(swapDim)))

# --------------------------------------------------------------------------------------------------------------
# Unitary Gates
# --------------------------------------------------------------------------------------------------------------

# Hadamard matrices
H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]])
H2 = (1/2) * np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]])
H3 = (1/np.sqrt(8)) * np.array([
    [1,1,1,1,1,1,1,1],
    [1,-1,1,-1,1,-1,1,-1],
    [1,1,-1,-1,1,1,-1,-1],
    [1,-1,-1,1,1,-1,-1,1],
    [1,1,1,1,-1,-1,-1,-1],
    [1,-1,1,-1,-1,1,-1,1],
    [1,1,-1,-1,-1,-1,1,1],
    [1,-1,-1,1,-1,1,1,-1]
])
H4 = (1/4) * np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1],
    [1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1],
    [1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1],
    [1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1],
    [1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1],
    [1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1],
    [1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1],
    [1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,-1,1,-1,1],
    [1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1],
    [1,-1,-1,1,1,-1,-1,1,-1,1,1,-1,-1,1,1,-1],
    [1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1],
    [1,-1,1,-1,-1,1,-1,1,-1,1,-1,1,1,-1,1,-1],
    [1,1,-1,-1,-1,-1,1,1,-1,-1,1,1,1,1,-1,-1],
    [1,-1,-1,1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1]
])

# Select Hadamard based on m
Hl = {1:H, 2:H2, 3:H3, 4:H4}.get(m, None)
if Hl is None:
    raise ValueError("Only m = 1 to 4 supported")

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
    v, l are 1-based indices to match Mathematica.
    """
    v_idx = v - 1
    l_idx = l - 1
    
    if 0 <= v_idx < len(adjacencyList) and 0 <= l_idx < len(adjacencyList[v_idx]):
        u = adjacencyList[v_idx][l_idx]  # adjacency node (1-based)
        feat_val = features[u - 1]       # convert to 0-based for Python array
        return Wval(feat_val)
    else:
        return np.identity(2)

print("GetFeatureRotation[1, 1] =\n", get_feature_rotation(1, 1))
print("GetFeatureRotation[2, 1] =\n", get_feature_rotation(2, 1))
print("GetFeatureRotation[2, 2] =\n", get_feature_rotation(2, 2))

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

firstHopStates = []

for v in range(nrNodes):  # 0-based indexing
    for l_idx in range(s):  # 0-based indexing for neighbors
        feat_vec = get_feature_rotation(v + 1, l_idx + 1) @ np.array([1, 0])  # column vector
        
        # Precompute all Kronecker combinations for 4 ancilla rotations
        vect_list_base = [np.array([1, 0]) for _ in range(4)]
        
        for i in range(1, 5):  # i = 1 to 4
            vect_list = vect_list_base.copy()
            # Replace first i entries with feat_vec
            for k in range(i):
                vect_list[k] = feat_vec
            # Compute Kronecker product
            combined_vec = vect_list[0]
            for vec in vect_list[1:]:
                combined_vec = np.kron(combined_vec, vec)
            
            firstHopStates.append({
                "v": v + 1,  # 1-based to match Mathematica
                "i": i,
                "l": l_idx + 1,  # 1-based
                "u": adjacencyList[v][l_idx] + 1,  # 1-based
                "featVec": feat_vec,
                "state": combined_vec
            })

print("Finished.")

# Norm of each state should be 1
# for fs in firstHopStates:
#     print(np.linalg.norm(fs["state"]))

# --------------------------------------------------------------------------------------------------------------
# QMME Circuit: c=2: Feature Embeddings for (v, l1, l2, i) 
# Second-hop neighborhood: For each (v, l1, l2), get the 16 feature vectors (for each i, l1, l2) of dimension 2^4 = 16 (corresponding to 4 ancilla rotations).
# --------------------------------------------------------------------------------------------------------------

print("Creating secondHopStates...")

# Preprocess: group firstHopStates by "v"
from collections import defaultdict

firstHopByV = defaultdict(list)
for state in firstHopStates:
    firstHopByV[state["v"]].append(state)

secondHopStates = []

for v0 in range(1, nrNodes + 1):  # 1-based indexing
    for l0 in range(1, s + 1):    # 1-based indexing
        u0 = adjacencyList[v0 - 1][l0 - 1] + 1  # 1-based adjacency
        neighborStates = firstHopByV.get(u0, [])
        
        for state in neighborStates:
            secondHopStates.append({
                "v": v0,
                "i": state["i"],
                "l0": l0,
                "u0": u0,
                "l1": state["l"],
                "u1": state["u"],
                "state": state["state"]
            })

print("Finished.")

# Optional check: Norm of each state
# for sh in secondHopStates:
#     print(np.linalg.norm(sh["state"]))

# --------------------------------------------------------------------------------------------------------------
# QMME Circuit: Superposition from firstHopsStates
# First-hop neighborhood: For each (v, i), get the one feature vectors of dimension 2^5 s = 64 (corresponding to 5 ancilla rotations and the l register).
# --------------------------------------------------------------------------------------------------------------

print("Creating firstHopSuperposedStates...")

from itertools import groupby
from operator import itemgetter

# Group firstHopStates by (v, i)
firstHopStates_sorted = sorted(firstHopStates, key=lambda x: (x["v"], x["i"]))
groupedByVI = {}
for key, group in groupby(firstHopStates_sorted, key=lambda x: (x["v"], x["i"])):
    groupedByVI[key] = list(group)

firstHopSuperposedStates = []

for (v, i), states in groupedByVI.items():
    c = 1
    stateDim = 16  # dimension of individual featVec states

    # Combine the states along l (length s * stateDim)
    combinedState = np.concatenate([state["state"] for state in states])

    # Hadamard on l register
    HOnL = np.kron(Hl, np.eye(stateDim))
    hadState = HOnL @ combinedState / np.sqrt(s)

    # Degree rotation
    degRotVec = Wtheta(2 * np.arccos(1 / nodeDegrees[v - 1])) @ np.array([1, 0])  # 1-based v
    finalState = np.kron(hadState, degRotVec).flatten()

    firstHopSuperposedStates.append({
        "c": c,
        "v": v,
        "i": i,
        "stateCombined": combinedState,
        "stateHadamardL": hadState,
        "degRotVec": degRotVec,
        "stateDegreeRotated": finalState
    })

print("Finished.")

# Optional check: Norm of each state
# for sh in firstHopSuperposedStates:
#     print(np.linalg.norm(sh["stateDegreeRotated"]))

# --------------------------------------------------------------------------------------------------------------
# QMME Circuit: Superposition from secondHopsStates
# Second-hop neighborhood: For each (v, i), get the one feature vectors of dimension 2^5 s^2 = 128 (corresponding to 5 ancilla rotations and the l registers).
# --------------------------------------------------------------------------------------------------------------

print("Creating secondHopSuperposedStates...")

# Group secondHopStates by (v, i)
secondHopStates_sorted = sorted(secondHopStates, key=lambda x: (x["v"], x["i"]))
groupedByVI2 = {}
for key, group in groupby(secondHopStates_sorted, key=lambda x: (x["v"], x["i"])):
    groupedByVI2[key] = list(group)

secondHopSuperposedStates = []

for (v, i), states in groupedByVI2.items():
    c = 2

    stateDim = states[0]["state"].size

    # Flatten all states in the group (column-major to match Mathematica)
    combinedState = np.stack([s["state"] for s in states]).flatten(order='F')

    # Full size according to sparsity level sC1
    full_length = s**2 * stateDim
    paddedCombined = np.zeros(full_length)
    paddedCombined[:combinedState.size] = combinedState

    # Kronecker product for Hadamard on l0 and l1 registers
    HOnL = np.kron(np.kron(Hl, Hl), np.eye(stateDim))

    # Multiply by HOnL using the padded state
    hadState = (HOnL @ paddedCombined) / s

    degRotVec = get_degree_rotation(v, c) @ np.array([1, 0])
    finalState = np.kron(hadState, degRotVec).flatten()
    finalState /= np.linalg.norm(finalState)

    secondHopSuperposedStates.append({
        "c": c,
        "v": v,
        "i": i,
        "stateCombined": combinedState,
        "stateHadamardL": hadState,
        "degRotVec": degRotVec,
        "stateDegreeRotated": finalState
    })

print("Finished.")

# Norm of each state should be 1
# for sh in secondHopSuperposedStates:
#     print(np.linalg.norm(sh["stateDegreeRotated"]))

# --------------------------------------------------------------------------------------------------------------
# QMME Circuit: Feature Embeddings Subscript[f, x](v,c,i)
# Both neighborhoods: For each (v, c, i), get the one feature vectors of dimension 2^5 s^2 = 128 (padded and unit-length).
# --------------------------------------------------------------------------------------------------------------

allSuperposedStates = []

target_len_c1 = 2**5 * s**2  # equivalent to 2^5 * s^2

# c = 1: pad to target length
for assoc in firstHopSuperposedStates:
    state_rot = assoc["stateDegreeRotated"]
    padded = np.pad(state_rot, (0, target_len_c1 - len(state_rot)))  # pad with zeros
    # optionally normalize to unit norm
    padded /= np.linalg.norm(padded)
    
    assoc_copy = assoc.copy()
    assoc_copy["statePadded"] = padded
    allSuperposedStates.append(assoc_copy)

for assoc in secondHopSuperposedStates:
    assoc_copy = assoc.copy()
    assoc_copy["statePadded"] = assoc_copy["stateDegreeRotated"]
    allSuperposedStates.append(assoc_copy)

# for assoc in filter(lambda a: a["v"] == 2, allSuperposedStates):
#     print({
#         "c": assoc["c"],
#         "i": assoc["i"],
#         "stateLengthPadded": len(assoc["statePadded"]) # 512
#     })

# for assoc in allSuperposedStates:
#     print(np.linalg.norm(assoc["statePadded"]))

print("Finished.")

# --------------------------------------------------------------------------------------------------------------
# QMME Circuit: Feature Embeddings Subscript[f, x](v)
# Both neighborhoods: For each v, get the one feature vectors of dimension 2^(5+3) s^2 = 128x8=1024 (unit-length).
# --------------------------------------------------------------------------------------------------------------

from collections import defaultdict

# Group all states by v
vectorsByV = defaultdict(list)
for assoc in allSuperposedStates:
    vectorsByV[assoc["v"]].append(assoc)

print("Creating superLongVectorsByV...")
superLongVectorsByV = {}

for v, assocList in vectorsByV.items():
    # Create a lookup {(c,i): assoc}
    lookup = {(assoc["c"], assoc["i"]): assoc for assoc in assocList}

    concatenated = []

    for c in [1, 2]:
        for i in range(1, 5):  # i = 1..4
            if (c, i) in lookup:
                vec = lookup[(c, i)]["statePadded"] / np.sqrt(8)
                concatenated.append(vec)
            else:
                # skip if missing (Mathematica uses empty list)
                pass

    # flatten into a single long vector
    superLongVectorsByV[v] = np.concatenate(concatenated)

# for v, vec in superLongVectorsByV.items():
#     print(f"v={v}, norm={np.linalg.norm(vec)}")
#     print(f"v={v}, length={vec.shape[0]}") # 4096

print("Finished.")

# --------------------------------------------------------------------------------------------------------------
# Swap-Test Classifier
# Expectation value of the product of two Pauli-Z operators acting on the ancilla qubits of the quantum classifier's state.
# It measures their correlation and captures the similarity between the test data and training data. 
# Kernel Matrix on Training Data
# --------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns

trainVecs = np.array(list(superLongVectorsByV.values()))
# trainVecs = trainVecs / np.linalg.norm(trainVecs, axis=1, keepdims=True) # Normalize vectors (not needed, if done correctly)

nPower = 1
nrNodes = len(trainVecs)

print("Creating Kernel Matrix Table...")
# Compute kernel matrix: |<v_i|v_j>|^(2 * nPower)
kernelMatrix = np.abs(trainVecs @ trainVecs.T) ** (2 * nPower)

print(f"Kernel matrix shape: {kernelMatrix.shape}")  # should be (nrNodes, nrNodes)

print("Creating Kernel Matrix Heatmap...")
plt.figure(figsize=(10, 8))
sns.heatmap(kernelMatrix, cmap="coolwarm", cbar_kws={'label': 'Kernel Value'})
plt.title("Kernel Matrix Heatmap")
plt.xlabel("Training Vector Index")
plt.ylabel("Training Vector Index")
plt.show()

# --- Parameters ---
thClass = 0.5
nrNodes = len(features)

# Class labels: 0 if < thClass, else 1
classLabels = np.where(features < thClass, 0, 1)

# Weight vector (uniform)
wV = np.ones(nrNodes) / nrNodes

print("features:", features)
print("Class labels:", classLabels)

# Train/Test setup
trainLabels = classLabels
testIsTrain = True

# --- Expectation Values ---
expectationValsAll = []

if testIsTrain:
    for v in range(nrNodes):
        val = np.sum(((-1) ** trainLabels) * wV * kernelMatrix[v])
        expectationValsAll.append(val)
else:
    for v in range(nrNodes):
        idxs = np.delete(np.arange(nrNodes), v)  # exclude self
        val = np.sum(((-1) ** trainLabels[idxs]) * wV[idxs] * kernelMatrix[v, idxs])
        expectationValsAll.append(val)

expectationValsAll = np.array(expectationValsAll)

# --- Predicted Labels ---
predictedLabels = 0.5 * (1 - np.sign(expectationValsAll))
predictedLabels = predictedLabels.astype(int)

# --- Results Table ---
results = []
for v in range(nrNodes):
    results.append({
        "Node": v,
        "ExpectationValue": expectationValsAll[v],
        "TrueLabel": trainLabels[v],
        "PredictedLabel": predictedLabels[v],
        "Correct": predictedLabels[v] == trainLabels[v]
    })

# --- Bar Chart Visualization ---
showLabels = True
colors = ['green' if r['Correct'] else 'red' for r in results]

plt.figure(figsize=(12,6))
bars = plt.bar(range(nrNodes), expectationValsAll, color=colors)

if showLabels:
    for bar, r in zip(bars, results):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height(),
                 f"T:{r['TrueLabel']}\nP:{r['PredictedLabel']}",
                 ha='center', va='bottom', fontsize=9)

plt.xlabel("Node Index")
plt.ylabel("Expectation Value")
plt.title(f"Expectation Values per Node with Predictions (N={nrNodes})")
plt.show()

# --- Metrics ---
trueLabels = np.array([r['TrueLabel'] for r in results])
predLabels = np.array([r['PredictedLabel'] for r in results])

posClass = 1
negClass = 0

tp = np.sum((predLabels == posClass) & (trueLabels == posClass))
fp = np.sum((predLabels == posClass) & (trueLabels == negClass))
fn = np.sum((predLabels == negClass) & (trueLabels == posClass))

accuracy = np.mean(predLabels == trueLabels)
precision = tp / (tp + fp) if (tp + fp) != 0 else 0
recall = tp / (tp + fn) if (tp + fn) != 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

# --- Print Metrics ---
metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
}

print("\nMetrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")