### Import modules
#from qiskit import Aer, execute  
from qiskit.visualization import circuit_drawer
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RYGate, UnitaryGate
from qiskit.quantum_info import Statevector
import numpy as np
import math

### Parameters (4,2,16) (3,1,8)
n = 4 # Number of qubits for |j⟩
# Be careful, we will encode, for now, having 1 neighbor with |k⟩=|0⟩, and having 2 neighbors as |k⟩=|1⟩
m = 2  # Number of qubits for |k⟩ (m << n)
# Be careful: if P =/= 2**p (and even =/= 2**n), issues may arise
P = 16 # Scaling factor (float * P = int)

### Network characteristics
N = 2**n  # Number of nodes
s = 2**m  # Sparsity level

### Adjacency matrix
A = np.zeros((N, N), dtype=int)

### Matrix option 1: random with less or equal to s neighbors

# np.random.seed(42)  # For reproducibility
# for i in range(N):
#     neighbors = np.random.choice(np.delete(np.arange(N), i), size=min(s, N-1), replace=False)
#     A[i, neighbors] = 1
# A[-1, :] = 0  # Force the last node to have no neighbors
# A = np.minimum(A, A.T)  # Ensure symmetry

### E structure (like an edge list)
# E_list = [np.nonzero(row)[0].tolist() for row in A]
# neighbors = [E_list[v] + [N-1] * (s - len(E_list[v])) for v in range(N)] # Fill missing neighbors with N-1 (last node)
# print("\nE_list: ", E_list)
# print("\nneighbors: ", neighbors)

#### Matrix Option 2: Circulant graph (subtype of Cayley graph) is defined by a set of shift operations that dictate node connection in a cyclic manner 

E_list = []
shifts = [1, -1]  # Node is connect to the previous and next one
if s > 2: # For s=4, shifts = [1, 2, -1, -2]
    shifts = list(range(1, s//2 + 1)) + [-i for i in range(1, s//2 + 1)]
for i in range(N):
    neighbors = []
    for shift in shifts:
        j = (i + shift) % N  # Wrap around (cycle structure)
        neighbors.append(j)
        A[i, j] = 1
        A[j, i] = 1  # Ensure symmetry
    # Attention: this is working for the case where s=2
    E_list.append(neighbors) #E_list.append([neighbors[-1], neighbors[0]])

### E structure (like an edge list)
neighbors = E_list
print("\nE_list: ", E_list)
print("\nneighbors: ", neighbors)

print("\n -------------------------------------------------------------------------------\n")
print("\nAdjacency Matrix (Sparse): \n", A)

### Neighbors 'inverse' (for O_L 'inverse')
neighbors_inverse = [[] for _ in range(N)]
for u, neighbors_u in enumerate(neighbors):
    for l, v in enumerate(neighbors_u):  # l is the index (position) of v in u's neighbors list
        neighbors_inverse[v].insert(l, u)  # Insert u at the correct position in v's inverse list
print("\nneighbors inverse: \n", neighbors_inverse)

### K (degree of each node)
K = [len(e) for e in E_list]
print("\nK: ", K)

### Feature vector
p = int(np.ceil(np.log2(P))) # Number of qubits for |x⟩
# Be careful: if the features are not these, issues may arise
# In particular, if features[N] is not 0, oracle O_L shall be adjusted
features = np.round(np.linspace(1 - 1/P, 0, N), 10)
print("\nFeature Vector: ", features)
print("\nFeature Vector (scaled by P): \n", features*P)

#################################### f_x_binary and inverse ########################################

def f_x_binary(float):
    ''' Multiply by the scaling factor (should be integer, and int forces that)
        Convert to p-bit binary representation
        Slicing [2:] removes the '0b' prefix
        zfill(p) ensures exactly p bits (pad with 0s if needed).'''
    return bin(int(float * P))[2:].zfill(p) 

def f_x_binary_inverse(binary, P):
    """ Decode a binary value in qr_z_state to its floating-point representation.
        qr_z_state is a string representing the binary state of qr_z."""
    return int(binary, 2) / P

def check_f_x_binary(vec, P):
    ''' Check if the functions f_x_binary and f_x_binary_inverse are working properly.'''
    vec_bin = [f_x_binary(x) for x in vec]
    vec_bin_inv = [f_x_binary_inverse(x_bin, P) for x_bin in vec_bin]
    # print("features: ", vec), print("f_x_binary: ", vec_bin), print("f_x_binary_inverse: ", vec_bin_inv)
    # Check if all elements match
    all_match = all(np.isclose(vec[i], vec_bin_inv[i]) for i in range(len(vec)))
    print("\nDo all elements (between vec and vec_bin_inv) match? ", "Yes" if all_match else "No")
    if not all_match:
        for i in range(len(vec)):
            if not np.isclose(vec[i], vec_bin_inv[i]):
                print(f"Mismatch at index {i}: Original = {vec[i]}, Decoded = {vec_bin_inv[i]}")

########################################## Simulation ############################################

def access_statevector(qc, flag_a=False, flag_l=True):
    ''' If flag=True, shows all substates, including non-zero qubits. '''
    # Generates the quantum statevector resulting from executing the quantum circuit qc
    statevector = Statevector.from_instruction(qc)

    # Get the "useful" states
    num_qubits = qc.num_qubits
    useful_states = []
    for idx, amplitude in enumerate(statevector):
        if not np.isclose(amplitude, 0):
            state_bin = bin(idx)[2:].zfill(num_qubits)

            # Check if state meets the required criteria
            if (flag_a or (state_bin[0] == '0' and state_bin[1] == '0')) and \
               (flag_l or (state_bin[-1] == '0')): # and state_bin[-2] == '0')): # Attention: depends!
                useful_states.append((state_bin, amplitude))

    # Extracted information from the states
    info_states = []
    for state_bin, amplitude in useful_states:#[-4:]: # Change
        info_states.append({
            'state_bin': state_bin,
            'amplitude': amplitude,
            'u': int(state_bin[2 + p + m + n: 2 + p + m + 2*n], 2), 
            'v': int(state_bin[2 + p + m : 2 + p + m + n], 2), 
            'k_v': int(state_bin[2 + p : 2 + p + m], 2), 
            'x_v': f_x_binary_inverse(state_bin[2 : 2 + p], P)
        })

    # Display
    print("\nExtracted Information from Useful States:")
    for state_info in info_states:
        print(f"State: |{state_info['state_bin']}>, Amplitude: {state_info['amplitude']}, "
            f"v: {state_info['v']}, x_v: {state_info['x_v']}, k_v: {state_info['k_v']}, u: {state_info['u']}")

    '''
    Remember we had:

    qr_l = QuantumRegister(m, 'l')  # Basis-encode non-zero entries of each row of A
    qr_u = QuantumRegister(n, 'u')  # Basis-encode node's l-neighbor index
    qr_v = QuantumRegister(n, 'v')  # Basis-encode node index
    qr_k = QuantumRegister(m, 'k')  # Basis-encode degree value (k)
    qr_z = QuantumRegister(p, 'z')  # Basis-encode feature value (x)
    qr_a1 = QuantumRegister(1, 'a1')  # Ancilla qubit for rotation (feature value)
    qr_a2 = QuantumRegister(1, 'a2')  # Ancilla qubit for rotation (degree inverse)

    qc = QuantumCircuit(qr_l, qr_u, qr_v, qr_k, qr_z, qr_a1, qr_a2)

    state_bin: |000000001111000000> = 00 1001 0000 00 1111 0000 00 = a1 a2 [qr_z][qr_k][q_v][q_u][qr_l]

    '''

###################################### Quantum Circuit ######################################
################################### 0. Quantum registers ####################################

qr_l = QuantumRegister(m, 'l')  # Basis-encode non-zero entries of each row of A
qr_u = QuantumRegister(n, 'u')  # Basis-encode node's l-neighbor index
qr_v = QuantumRegister(n, 'v')  # Basis-encode node index
qr_k = QuantumRegister(m, 'k')  # Basis-encode degree value (k)
qr_z = QuantumRegister(p, 'z')  # Basis-encode feature value (x)
qr_a1 = QuantumRegister(1, 'a1')  # Ancilla qubit for rotation (feature value)
qr_a2 = QuantumRegister(1, 'a2')  # Ancilla qubit for rotation (degree inverse)

########### 1. Create the quantum circuit (Start with all qubits initialized to |0⟩) #########

qc = QuantumCircuit(qr_l, qr_u, qr_v, qr_k, qr_z, qr_a1, qr_a2)
print("\n1. Initial:"), access_statevector(qc)

########## 2. Apply Hadamard gates to the `qr_v` qubits to create superposition #############

qc.h(qr_v)
print("\n2. Superposition of 'q_v' qubits:"), access_statevector(qc)

#################################### 3. Apply oracle O_X ####################################

def oracle_O_X(qc, qr_v, qr_z, features, N):
    """ Apply the oracle O_X to entangle qr_v (v node index) and qr_z (encode x)."""
    bin_v = [bin(v)[2:].zfill(len(qr_v)) for v in range(N)] 
    bin_x = [f_x_binary(features[v]) for v in range(N)]
    print(""), [print(v, bin_v[v], features[v], bin_x[v]) for v in range(N)]

    for v in range(N): # Doing [j] instead of [-j-1], after 0, it's doing 8 (1000), not 1
        #print(f"\nProcessing v={v} with bin_x[v] = {bin_x[v]} and bin_v[v] = {bin_v[v]}")
        # Flip necessary qubits in qr_v where needed
        [qc.x(qr_v[j]) for j, b in enumerate(bin_v[v]) if b == '0']
        #print(f"\nStatevector after applying X on qr_v for v={v}:"), access_statevector(qc)

        # Apply MCX gates where x_v[i] is 1
        [qc.mcx(qr_v, qr_z[i]) for i, x in enumerate(bin_x[v]) if x == '1']
        #print(f"\nStatevector after MCX on qr_z for v={v}:"), access_statevector(qc)

        # Restore flipped qubits in qr_v
        [qc.x(qr_v[j]) for j, b in enumerate(bin_v[v]) if b == '0']
        #print(f"\nStatevector after applying oracle O_X for v={v}:"), access_statevector(qc)

    return qc
        
qc = oracle_O_X(qc, qr_v, qr_z, features, N)
print("\n3. Apply oracle O_X:"), access_statevector(qc)

#print(""), [print(v, 1/math.sqrt(N)) for v in range(N)]

######################## 4. Controlled Rotation of Ancilla a1 by F_X ########################

def F_X(P):
    ''' Controlled rotation of one qubit based on a p-qubits register.
        We only need to know P (scaling factor) to create this unitary gate. '''
    thetas = [float(2 * np.arccos(x_int / P)) for x_int in range(P)] # turn x_int into x from features
    print("\nThetas (F_X): ", thetas)
    unitary = np.eye(2 * P, dtype=complex)  # Identity matrix of size 2P

    for x, theta in enumerate(thetas):
        W = np.array([
            [np.cos(theta / 2), -np.sin(theta / 2)],
            [np.sin(theta / 2), np.cos(theta / 2)]
        ])
        id = 2 * x  # Pair of rows/cols for each x
        unitary[id:(id + 2), id:(id + 2)] = W
        '''
            Building the unitary like this leads to having:
            x_v values in elements [v*2, v*2] 
            (instead of [v,v])
            That is why we apply on [qr_a1] + qr_z[:]
            (instead of qr_z[:] + [qr_a1])
        '''

    return UnitaryGate(unitary, label=f'F_X(P={P})')

# Append adds gates (or operations) to the quantum registers already defined
qc.append(F_X(P), [qr_a1] + qr_z[:])
print("\n4. Apply F_X:"), access_statevector(qc)
print("\nExpected amplitudes (x_v normalized):"), [print(v, features[v], features[v]/math.sqrt(N)) for v in range(N)]

############################ 5. Superposition of the l-register qubits ###########################

qc.h(qr_l)
print("\n5. Superposition of 'q_l' qubits:"), access_statevector(qc)
print("\nExpected amplitudes (x_v normalized):"), [print(v, features[v]/math.sqrt(N*s)) for v in range(N)]

#################################### 6. Apply oracle O_L #########################################

def f_int_binary(int, b):
    ''' Convert int to b-bit binary representation
        Slicing [2:] removes the '0b' prefix
        zfill(b) ensures exactly b bits (pad with 0s if needed).'''
    #return bin(int)[2:].zfill(b) 
    return format(int, f'0{b}b')

def oracle_O_L(qc, qr_l, qr_v, qr_u, neighbors, s, N):
    """ Apply the oracle O_L to entangle qr_l (l index) and qr_v (encode r(v,l)). """
    
    bin_l = [bin(l)[2:].zfill(len(qr_l)) for l in range(s)] 
    bin_v = [bin(v)[2:].zfill(len(qr_v)) for v in range(N)] 
    print("\n Neighbors: ", neighbors)

    ### Try to do these calculations with matrices on Mathematica (before-hand)
    ### THIS IS NOT WORKING AS EXPECTED (PROBLEM)
    ### Salvou-me o reversed no controlo em bin_v e no encoding de bin_u?
    
    # Apply multi-controlled X gates based on neighbors
    for v in range(N):        
        #print(f"\nProcessing v={v}...")
        # Reason: Just like we control in qr_l, we also do in qr_v
        [qc.x(qr_v[j]) for j, b in enumerate(reversed(bin_v[v])) if b == '0']
        
        for l in range(s):
            # Flip necessary qubits in qr_l where needed
            [qc.x(qr_l[j]) for j, b in enumerate(reversed(bin_l[l])) if b == '0']
            #print(f"\nStatevector after flipping qr_l and qr_v for v={v}, l={l}:"), access_statevector(qc)

            bin_u = f_int_binary(neighbors[v][l], len(qr_v))
            #print(f"  Processing v={v}, neighbors[v={v}][l={l}] = {neighbors[v][l]}, bin_u = {bin_u}")

            # Apply MCX gates where qubit_i is '1'
            # Update/Main difference here: qc.mcx(qr_l, qr_v[i]) -> qc.mcx(qr_l, qr_u[i])
            # Because I want to use the already-existing mcx
            [qc.mcx(qr_v[:] + qr_l[:], qr_u[i]) for i, qubit_i in enumerate(reversed(bin_u)) if qubit_i == '1']
            #print(f"    Applied MCX gates for l={l}")

            # Restore flipped qubits in qr_l
            [qc.x(qr_l[j]) for j, b in enumerate(reversed(bin_l[l])) if b == '0']
            #print(f"\nStatevector after restoring qr_l and qr_v for v={v}, l={l}:"), access_statevector(qc)

        # Restore flipped qubits in qr_v
        [qc.x(qr_v[j]) for j, b in enumerate(reversed(bin_v[v])) if b == '0']

    return qc

qc = oracle_O_L(qc, qr_l, qr_v, qr_u, neighbors, s, N)
print("\n6. Apply oracle O_L:"), access_statevector(qc)
print("\nExpected amplitudes:", *[f"k_{i+1}(v={N-v-1}) = {neighbors[N-v-1][i]} | {features[N-v-1]/math.sqrt(N*s)}" for v in range(N) for i in range(s)], sep='\n')
#print("\n", ",".join(f" k({v+1})={K[v]}" for v in range(N)))
#print("\nAdjacency Matrix (Sparse): \n", A)

########################################### Intermediate #########################################

# Put qr_z to zeros (do a good justification)
qc = oracle_O_X(qc, qr_v, qr_z, features, N)
print("\nIntermediate 0. Reapply oracle O_X:"), access_statevector(qc)

def swap_registers(qc, qr_u, qr_v):
    """ Perform the swap of quantum registers qr_u and qr_v. """
    
    n = len(qr_u)
    
    for i in range(n):
        qc.cx(qr_u[i], qr_v[i]) 
        qc.cx(qr_v[i], qr_u[i]) 
        qc.cx(qr_u[i], qr_v[i])  

    return qc

qc = swap_registers(qc, qr_u, qr_v)
print("\nIntermediate 1. Swap qr_v with qr_u:"), access_statevector(qc)

def oracle_O_L_inverse(qc, qr_l, qr_v, qr_u, neighbors, s, N):
    """ Apply the oracle O_L_inverse to put qr_u (v) to zeros, based to qr_l and qr_v (r(v,l)). """
    
    bin_l = [bin(l)[2:].zfill(len(qr_l)) for l in range(s)] 
    bin_v = [bin(v)[2:].zfill(len(qr_v)) for v in range(N)] 
    print("\n Neighbors: ", neighbors)
    print("\n Neighbors Inverse: ", neighbors_inverse)

    # Apply multi-controlled X gates based on neighbors
    for v in range(N):        
        [qc.x(qr_v[j]) for j, b in enumerate(reversed(bin_v[v])) if b == '0']
        
        for l in range(s):
            # Flip necessary qubits in qr_l where needed
            [qc.x(qr_l[j]) for j, b in enumerate(reversed(bin_l[l])) if b == '0']

            # focus -- bin_u is v for which r(v,l) is encoded in qr_v
            bin_u = f_int_binary(neighbors_inverse[v][l], len(qr_v))

            [qc.mcx(qr_v[:] + qr_l[:], qr_u[i]) for i, qubit_i in enumerate(reversed(bin_u)) if qubit_i == '1']

            # Restore flipped qubits in qr_l
            [qc.x(qr_l[j]) for j, b in enumerate(reversed(bin_l[l])) if b == '0']

        # Restore flipped qubits in qr_v
        [qc.x(qr_v[j]) for j, b in enumerate(reversed(bin_v[v])) if b == '0']

    return qc

qc = oracle_O_L_inverse(qc, qr_l, qr_v, qr_u, neighbors, s, N)
print("\nIntermediate 2. 'Reset' qr_u:"), access_statevector(qc)

############################ 7. Superposition of the l-register qubits ###########################

qc.h(qr_l)
print("\n7. Superposition of 'q_l' qubits:"), access_statevector(qc, flag_l=False)

### Checking the sum of neighbors-features for each node (3 and 4 have no neighbors)
from collections import defaultdict
neighbor_feature_sums = defaultdict(float)
expected_amplitudes = defaultdict(float)

for v in range(N):
    for neighbor in neighbors[v]:
        expected_amplitudes[neighbor] += features[v] / (s * math.sqrt(N))
        neighbor_feature_sums[neighbor] += features[v] / (s * math.sqrt(len(set(K)) * N)) # 2 comes from later using ancilla qubit

print("\nExpected amplitudes:", 
      *[f"node (v) = {v}, sum_features={expected_amplitudes[u]}" for u in sorted(expected_amplitudes)], sep='\n')

#################################### 8. Apply oracle O_K ####################################

def oracle_O_K(qc, qr_v, qr_k, K, N):
    """ Apply the oracle O_K to entangle qr_v (node index) and qr_k (encode k_v)."""
    bin_v = [bin(v)[2:].zfill(len(qr_v)) for v in range(N)] 

    for v in range(N): 

        if K[v] == 0:
            continue 
        # Notice: having 1 neighbor will lead to the same encoding as having 0 neighbors
        # Because: if a node has 0 neighbors, its index won't be encoded in qr_v register
        bin_k = format(K[v] - 1, f'0{len(qr_k)}b')
        
        # Flip necessary qubits in qr_v where needed
        [qc.x(qr_v[j]) for j, b in enumerate(bin_v[v]) if b == '0']

        # Apply MCX gates where k[i] is 1
        [qc.mcx(qr_v, qr_k[i]) for i, x in enumerate(bin_k) if x == '1']

        # Restore flipped qubits in qr_v
        [qc.x(qr_v[j]) for j, b in enumerate(bin_v[v]) if b == '0']

    return qc

qc = oracle_O_K(qc, qr_v, qr_k, K, N)
print("\n8. Apply oracle O_K (true degree = k_v + 1):"), access_statevector(qc, flag_a=True, flag_l=True)

######################## 9. Controlled Rotation of Ancilla a1 by F_X ########################

def F_K_inv(s):
    ''' Controlled rotation of one qubit based on a m-qubits register.
        We only need to know s (sparsity level) to create this unitary gate.
        Note that having s=2 neighbors, qubit 0 means k=1, and qubit 1 means k=2'''
    # Avoid: float division by zero
    thetas = [
        np.pi if k_int == 0 else float(2 * np.arccos(1 / k_int)) 
        for k_int in range(1, s + 1)] # Important to match our encoding in qr_k
    print("\nThetas (F_K_inv): ", thetas)
    unitary = np.eye(2 * s, dtype=complex)  # Identity matrix of size 2s

    for k, theta in enumerate(thetas):
        W = np.array([
            [np.cos(theta / 2), -np.sin(theta / 2)],
            [np.sin(theta / 2), np.cos(theta / 2)]
        ])
        id = 2 * k  # Pair of rows/cols for each k
        unitary[id:(id + 2), id:(id + 2)] = W

    return UnitaryGate(unitary, label=f'F_K_inv(s={s})')

# Append adds gates (or operations) to the quantum registers already defined
qc.append(F_K_inv(s), qr_k[:] + [qr_a2])
print("\n9. Apply F_K_inv:"), access_statevector(qc, flag_a=False, flag_l=False)

### Checking the mean of neighbors-features for each node (3 and 4 have no neighbors)
feature_means = {
    v: (neighbor_feature_sums[v] #/ K[v]
        if K[v] > 0 else 0)
    for v in range(N)
}

print("\nMean:", *[f"node (v) = {node}, neighbors' features mean={mean}" for node, mean in feature_means.items()], sep='\n')

###################################### Draw and Visualize ########################################

print("\n -------------------------------------------------------------------------------\n")
circuit_drawer(qc, output='mpl', filename='network_circuit.png')
qc.draw(output='mpl', filename='network_circuit.png')

