### Import modules
#from qiskit import Aer, execute  
from qiskit.visualization import circuit_drawer
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RYGate, UnitaryGate
from qiskit.quantum_info import Statevector
import numpy as np
import math

### Parameters (4,2,16)
n = 4 # Number of qubits for |j⟩
m = 2  # Number of qubits for |k⟩ (m << n)
# Be careful: if P =/= 2**p, issues may arise
P = 16 # Scaling factor (float * P = int)

### Network characteristics
N = 2**n  # Number of nodes
s = 2**m  # Sparsity level

### Adjacency matrix
A = np.zeros((N, N), dtype=int)
np.random.seed(42)  # For reproducibility
for i in range(N):
    neighbors = np.random.choice(np.delete(np.arange(N), i), size=min(s, N-1), replace=False)
    A[i, neighbors] = 1
A[-1, :] = 0  # Force the last node to have no neighbors 
A = np.minimum(A, A.T)  # Ensure symmetry
#print("\n -------------------------------------------------------------------------------\n")
#print("\nAdjacency Matrix (Sparse): \n", A)

### E structure (like an edge list)
E_list = [np.nonzero(row)[0].tolist() for row in A]
neighbors = [E_list[v] + [N-1] * (s - len(E_list[v])) for v in range(N)] # Fill missing neighbors with N-1 (last node)
#print("\nE_list: ", E_list)
#print("\nneighbors: ", neighbors)

### K (degree of each node)
K = [len(e) for e in E_list]

### Feature vector
p = int(np.ceil(np.log2(P))) # Number of qubits for |x⟩
# Be careful: if the features are not these, issues may arise
# In particular, if features[N] is not 0, oracle O_L shall be adjusted
features = np.round(np.linspace(1 - 1/P, 0, N), 10)
#print("\nFeature Vector: ", features)
#print("\nFeature Vector (scaled by P): \n", features*P)

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

########################################## Simulation ############################################

def access_statevector(qc, flag_l=False):
    # Generates the quantum statevector resulting from executing the quantum circuit qc
    statevector = Statevector.from_instruction(qc)

    # Get the "useful" states
    num_qubits = qc.num_qubits
    useful_states = []
    for idx, amplitude in enumerate(statevector):
        if not np.isclose(amplitude, 0):
            state_bin = bin(idx)[2:].zfill(num_qubits)

            # Check if state meets the required criteria
            if (state_bin[0] == '0' and state_bin[1] == '0'):
                if flag_l:
                    if state_bin[-1] == '0' and state_bin[-2] == '0':
                        useful_states.append((state_bin, amplitude))
                else:
                    useful_states.append((state_bin, amplitude))

    # Extracted information from the states
    info_states = []
    for state_bin, amplitude in useful_states[-4:]: # Change
        info_states.append({
            'state_bin': state_bin,
            'amplitude': amplitude,
            'v': int(state_bin[2 + p + m : 2 + p + m + n], 2), 
            'k_v': int(state_bin[2 + p : 2 + p + m], 2), 
            'x_v': f_x_binary_inverse(state_bin[2 : 2 + p], P)
        })

    # Display
    print("\nExtracted Information from Useful States:")
    for state_info in info_states:
        print(f"State: |{state_info['state_bin']}>, Amplitude: {state_info['amplitude']}, "
            f"v: {state_info['v']}, x_v: {state_info['x_v']}, k_v: {state_info['k_v']}")

    '''
    Remember we had:

    qr_l = QuantumRegister(m, 'l')  # Basis-encode non-zero entries of each row of A
    qr_v = QuantumRegister(n, 'v')  # Basis-encode node index
    qr_k = QuantumRegister(m, 'k')  # Basis-encode degree value (k)
    qr_z = QuantumRegister(p, 'z')  # Basis-encode feature value (x)
    qr_a1 = QuantumRegister(1, 'a1')  # Ancilla qubit for rotation (feature value)
    qr_a2 = QuantumRegister(1, 'a2')  # Ancilla qubit for rotation (degree inverse)

    qc = QuantumCircuit(qr_l, qr_v, qr_k, qr_z, qr_a1, qr_a2)

    state_bin: |00000000111100> = 00 0000 00 1111 00 = a1 a2 [qr_z][qr_k][q_v][qr_l]

    '''

###################################### Quantum Circuit ######################################
################################### 0. Quantum registers ####################################

qr_l = QuantumRegister(m, 'l')  # Basis-encode non-zero entries of each row of A
qr_v = QuantumRegister(n, 'v')  # Basis-encode node index
qr_k = QuantumRegister(m, 'k')  # Basis-encode degree value (k)
qr_z = QuantumRegister(p, 'z')  # Basis-encode feature value (x)
qr_a1 = QuantumRegister(1, 'a1')  # Ancilla qubit for rotation (feature value)
qr_a2 = QuantumRegister(1, 'a2')  # Ancilla qubit for rotation (degree inverse)

########### 1. Create the quantum circuit (Start with all qubits initialized to |0⟩) #########
########## 2. Apply Hadamard gates to the `qr_v` qubits to create superposition #############

qc = QuantumCircuit(qr_l, qr_v, qr_k, qr_z, qr_a1, qr_a2)
qc.h(qr_v)

#################################### 3. Apply oracle O_X ####################################

def oracle_O_X(qc, qr_v, qr_z, features, N):
    """ Apply the oracle O_X to entangle qr_v (v node index) and qr_z (encode x)."""
    bin_v = [bin(v)[2:].zfill(len(qr_v)) for v in range(N)] 
    bin_x = [f_x_binary(features[v]) for v in range(N)]
    #print(""), [print(v, bin_v[v], features[v], bin_x[v]) for v in range(N)]

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

######################## 4. Controlled Rotation of Ancilla a1 by F_X ########################
############################ 5. Superposition of the l-register qubits ###########################
#################################### 6. Apply oracle O_L #########################################

qc.h(qr_l)

def f_int_binary(int, b):
    ''' Convert int to b-bit binary representation
        Slicing [2:] removes the '0b' prefix
        zfill(b) ensures exactly b bits (pad with 0s if needed).'''
    return bin(int)[2:].zfill(b) 

def oracle_O_L(qc, qr_l, qr_v, neighbors, s, N):
    """ Apply the oracle O_L to entangle qr_l (l index) and qr_v (encode r(v,l)). """
    
    bin_l = [bin(l)[2:].zfill(len(qr_l)) for l in range(s)] 
    bin_v = [bin(v)[2:].zfill(len(qr_v)) for v in range(N)] 
    #print("\n Neighbors: ", neighbors)

    for l in range(s):
        print(f"\nProcessing l={l}...")
        
        # Flip necessary qubits in qr_l where needed
        [qc.x(qr_l[j]) for j, b in enumerate(bin_l[l]) if b == '0']
        print(f"\nStatevector after flipping qr_l for l={l}:"), access_statevector(qc)

        ### Try to do these calculations with matrices on Mathematica (before-hand)
        ### THIS IS NOT WORKING AS EXPECTED (PROBLEM)
        
        # Apply multi-controlled X gates based on neighbors (here)
        for v in range(N):
            bin_u = f_int_binary(neighbors[v][l], len(qr_v))
            #print(f"  Processing v={v}, neighbors[{v}][{l}] = {neighbors[v][l]}, bin_u = {bin_u}")

            # Apply MCX gates where qubit_i is '1'
            [qc.mcx(qr_l, qr_v[i]) for i, qubit_i in enumerate(reversed(bin_u)) if qubit_i == '1']
            #print(f"    Applied MCX gates for v={v}")

        # Restore flipped qubits in qr_l
        [qc.x(qr_l[j]) for j, b in enumerate(bin_l[l]) if b == '0']
        print(f"\nStatevector after restoring qr_l for l={l}:"), access_statevector(qc)

    return qc

qc = oracle_O_L(qc, qr_l, qr_v, neighbors, s, N)
print("\n6. Apply oracle O_L:"), access_statevector(qc)
print("\nExpected result:", *[f"k_{i+1}(v={v}) = {neighbors[v][i]} | x_v={features[v]}" for v in range(1) for i in range(s)], sep='\n')

###################################### Draw and Visualize ########################################

print("\n -------------------------------------------------------------------------------\n")