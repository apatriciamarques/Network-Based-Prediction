### Import Modules
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import Gate, Instruction
from qiskit.circuit.library import UnitaryGate, Initialize
from qiskit.quantum_info import Statevector
from qiskit.visualization import circuit_drawer
import qiskit.qpy as qpy
import numpy as np
import math
import random

#------------------------------------------------------------- PARAMETERS

### Parameters (4,2,16) (3,1,8) (2,1,4)
n = 2 # Number of qubits for |j⟩
# Having 1 neighbor as |k⟩=|0⟩, and having 2 neighbors as |k⟩=|1⟩
m = 1  # Number of qubits for |k⟩ (m << n)
# Precision level (number of qubits) for representing x_v
p = n

### Network characteristics
N = 2**n  # Number of nodes
s = 2**m  # Sparsity level
P = 2**p  # X scaling factor

### Adjacency matrix
A = np.zeros((N, N), dtype=int)

#------------------------------------------------------------- MATRICES OPTIONS

adjacency_list = []
shifts = [1, -1]  # Node is connect to the previous and next one
if s > 2: # For s=4, shifts = [1, 2, -1, -2]
    shifts = list(range(1, s//2 + 1)) + [-i for i in range(1, s//2 + 1)]
for i in range(N):
    v_list = []
    for shift in shifts:
        j = (i + shift) % N  # Wrap around (cycle structure)
        v_list.append(j)
        A[i, j] = 1
        A[j, i] = 1 
    adjacency_list.append(v_list) 

print("\nAdjacency Matrix (Sparse): \n", A)
print("\nadjacency_list: \n", adjacency_list)

### Nodes degrees
nodes_degrees = [len(e) for e in adjacency_list]
print("\nnode_degrees: ", nodes_degrees)
# Chosen for simplicity: if all nodes have the same degree, we shall divide by degree squared.
nodes_degrees_c1 = [k**2 for k in nodes_degrees]

### Feature vector
# Be careful: if the features are not these, issues may arise
# In particular, if features[N] is not 0, oracle O_L shall be adjusted
features = np.round(np.linspace(1 - 1/P, 0, N), 10)
features_int = (features * P).astype(int)
print("\nFeature Vector: \n", features)
print("\nFeature Vector (scaled by P): \n", features_int)

def f_x_binary(float):
    ''' Multiply by the scaling factor (should be integer, and int forces that)
        Convert to p-bit binary representation
        Slicing [2:] removes the '0b' prefix
        zfill(p) ensures exactly p bits (pad with 0s if needed).'''
    return bin(int(float * P))[2:].zfill(p) 

def check_f_x_binary(vec, P):
    ''' Check if the functions f_x_binary and f_x_binary_inverse are working properly.'''
    vec_bin = [f_x_binary(x) for x in vec]
    vec_bin_inv = [(int(x_bin, 2) / P) for x_bin in vec_bin]
    # Check if all elements match
    all_match = all(np.isclose(vec[i], vec_bin_inv[i]) for i in range(len(vec)))
    print("\nDo all elements (between vec and vec_bin_inv) match? ", "Yes" if all_match else "No")
    if not all_match:
        for i in range(len(vec)):
            if not np.isclose(vec[i], vec_bin_inv[i]):
                print(f"Mismatch at index {i}: Original = {vec[i]}, Decoded = {vec_bin_inv[i]}")

#------------------------------------------------------------- SIMULATION

def access_statevector(qc, flag_c=True, flag_i=True, flag_a=False, flag_l=True):
    ''' Generates the quantum statevector resulting from executing the quantum circuit qc.
        If flag=True, shows all substates, including non-zero qubits. '''
    
    statevector = Statevector.from_instruction(qc)

    print("\nExtracted Information from Useful States:")
    for idx, amplitude in enumerate(statevector):
        if amplitude.real == 0.00:  
            continue  # Hide states

        state_bin = format(idx, f'0{qc.num_qubits}b')

        if (flag_c or state_bin[n + 2]) and \
           (flag_i or (state_bin[n] == '0' and state_bin[n + 1] == '0')) and \
           (flag_l or all(state_bin[n + 3 + i] == '0' for i in range(2*m))) and \
           (flag_a or all(state_bin[-i] == '0' for i in range(1, 6))):

            v = int(state_bin[:n], 2)
            c = int(state_bin[n + 2:n + 3], 2)
            l1 = int(state_bin[n + 3:n + 3 + m], 2)
            l2 = int(state_bin[n + 3 + m:n + 3 + 2*m], 2)
            u1 = int(state_bin[n + 3 + 2*m:2*n + 3 + 2*m], 2)
            u2 = int(state_bin[2*n + 3 + 2*m:3*n + 3 + 2*m], 2)
            k_v = int(state_bin[3*n + 3 + 2*m:3*n + 3 + 4*m], 2)
            x_u = int(state_bin[3*n + 3 + 4*m:3*n + 3 + 4*m + p], 2) / P
            a_clas = int(state_bin[3*n + 3 + 4*m + p:3*n + 4 + 4*m + p], 2)

            print(f"State: |{state_bin}>, Amplitude: {round(amplitude.real, 10)}, "
                  f"v: {v}, c: {c}, l1: {l1}, l2: {l2}, u1: {u1}, u2: {u2}, x_u: {x_u}, k_v: {k_v}, a_clas: {a_clas}")

    '''

    qr_v = QuantumRegister(n, 'v')  # Basis-encode node index
    qr_i = QuantumRegister(2, 'i')  # Basis-encode power (i) of feature value (x^i)
    qr_c = QuantumRegister(1, 'c')  # Basis-encode power (c) of matrix A (A^c)
    qr_l1 = QuantumRegister(m, 'l1')  # Basis-encode non-zero entries of each row of A
    qr_l2 = QuantumRegister(m, 'l2')  # Basis-encode non-zero entries of each row of A^2
    qr_u1 = QuantumRegister(n, 'u1')  # Basis-encode node's l-neighbor index of v (auxiliary)
    qr_u2 = QuantumRegister(n, 'u2')  # Basis-encode node's l-neighbor index of u (auxiliary)
    qr_k = QuantumRegister(2*m, 'k')  # Basis-encode degree value (k) [!]
    qr_z = QuantumRegister(p, 'z')  # Basis-encode feature value (x)
    qr_a = QuantumRegister(5, 'a')  # Ancilla qubit for rotation
    
    qc = QuantumCircuit(qr_a, qr_z, qr_k, qr_u2, qr_u1, qr_l2, qr_l1, qr_c, qr_i, qr_v)

    state_bin: [qr_v][qr_i]qr_c[qr_l1][qr_l2][qr_u1][qr_u2][qr_k][qr_z][qr_a] (n,m,N)
    state_bin: 1111   01    0    10     01    0001   0011   11    0110  00000 (4,2,16)
    state_bin: 111    01    0    1      0     001    011    1     010   00000 (3,1,8)

    '''

class IntegerToRegisterGate(Gate):
    ''' 
        Gate to take an integer and encode it into a quantum register.
        Set a register to an integer value via X gates.
    '''
    def __init__(self, name, integer, num_qubits): # Initialize gate
        max_value = 2**num_qubits - 1
        if integer < 0 or integer > max_value:
            raise ValueError(f"Integer {integer} cannot be represented with {num_qubits} qubits")

        self.integer = integer
        self.num_qubits = num_qubits
        super().__init__(name, num_qubits, [integer]) # 0 classical bits, 1 param (the integer)
        self._define()

    def _define(self): # Behaviour/operations of the gate
        binary = format(self.integer, f'0{self.num_qubits}b')
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        [qc.x(i) for i, bit in enumerate(reversed(binary)) if bit == '1']
        self.definition = qc  # Quantum Circuit

    def inverse(self):
        return self  # X gates are self-inverse
    
def FeatureRotation(thetas, control_size, gate_name):
    ''' Controlled rotation of one qubit based on a register with nr_qubits.
        Here, control_size = 2**nr_qubits.
    '''
    print("\nThetas (F_X): ", thetas)
    unitary = np.eye(2 * control_size, dtype=complex)  

    for x, theta in enumerate(thetas):
        W = np.array([
            [np.cos(theta / 2), -np.sin(theta / 2)],
            [np.sin(theta / 2), np.cos(theta / 2)]
        ])
        id = 2 * x  # Pair of rows/cols for each x
        unitary[id:(id + 2), id:(id + 2)] = W

    return UnitaryGate(unitary, label=gate_name)

########################################## Oracle O_L ###############################################

def oracle_O_L(adjacency_list, n, m):
    ''' Return an oracle O_L Gate that encodes r(v,l) into qr_u, controlled on qr_v and qr_l. '''
    qc = QuantumCircuit(m + n + n, name='O_L')  # m qubits (l), n qubits (v), n qubits (u output)

    for v in range(len(adjacency_list)):
        for l, u in enumerate(adjacency_list[v]):
            subgate = IntegerToRegisterGate('O_L_val', u, n)
            ctrl_gate = subgate.control(num_ctrl_qubits=m + n, ctrl_state=f'{(l + v * (2 ** m)):0{m + n}b}')
            qc.append(ctrl_gate, list(range(m + n + n)))
            
    return qc.to_gate(label='O_L')

gate_O_L = oracle_O_L(adjacency_list, n, m)
print(gate_O_L.definition)

######################################## Oracle O_L_test #############################################

## Test Node (Note that I'm using a test node which is also in the train data)
test_v = 0

def oracle_O_L_test(adjacency_list, test_v, n, m):
    '''
    Return an oracle O_L gate that encodes r(v, l) = u,
    where v is the test node (externally/pre-defined) and the oracle is controlled on qr_l (m qubits).
    '''
    qc = QuantumCircuit(m + n, name=f'O_L_v{test_v}')  # m qubits for l, n qubits for u (output)

    for l, u in enumerate(adjacency_list[test_v]):
        subgate = IntegerToRegisterGate('O_L_val', u, n)
        ctrl_gate = subgate.control(num_ctrl_qubits=m, ctrl_state=f'{l:0{m}b}')
        qc.append(ctrl_gate, list(range(m + n)))  # qr_l (controls) + qr_u (targets)

    return qc.to_gate(label=f'O_L_v{test_v}')

gate_O_L_test = oracle_O_L_test(adjacency_list, test_v, n, m)

########################################## Oracle O_X ###############################################

def oracle_O_X(features, n, p, P, ctrl_val):
    '''Return an oracle_O_X gate that encodes feature x_u into qr_z, controlled on qr_c and qr_u: '''
    qc = QuantumCircuit(1 + n + p, name=f'O_X')

    for u, x_u in enumerate(features):
        subgate = IntegerToRegisterGate(f'x_u', int(x_u * P), p)

        # control states: qr_c=0 for u1, qr_c=1 for u2
        ctrl_gate = subgate.control(num_ctrl_qubits = 1 + n, ctrl_state = f'{(ctrl_val + u * 2):0{1 + n}b}')
        qc.append(ctrl_gate, list(range(1 + n + p)))

    return qc.to_gate(label=f'O_X')

ctrl_gate_O_X_c0 = oracle_O_X(features, n, p, P, ctrl_val=0) # If ctrl_val == 0: encodes to qr_z from qr_u1
ctrl_gate_O_X_c1 = oracle_O_X(features, n, p, P, ctrl_val=1) # If ctrl_val == 1: encodes to qr_z from qr_u2
print(ctrl_gate_O_X_c0.definition)
print(ctrl_gate_O_X_c1.definition)

########################################## Oracle O_K ###############################################

def oracle_O_K(nodes_degrees, n, k_bits):
    '''Return an oracle_O_K gate that encodes (k_v - 1) into qr_k, controlled on qr_v.'''
    qc = QuantumCircuit(n + k_bits, name='O_K')

    for v, k_v in enumerate(nodes_degrees):
        if k_v == 0:
            continue
        subgate = IntegerToRegisterGate('O_K_val', k_v - 1, k_bits)
        ctrl_gate = subgate.control(num_ctrl_qubits = n, ctrl_state = f'{v:0{n}b}')
        qc.append(ctrl_gate, list(range(n + k_bits)))

    return qc.to_gate(label='O_K')

ctrl_gate_O_K_c0 = oracle_O_K(nodes_degrees, n, 2*m).control(num_ctrl_qubits = 1, ctrl_state = f'{0:01b}')
ctrl_gate_O_K_c1 = oracle_O_K(nodes_degrees_c1, n, 2*m).control(num_ctrl_qubits = 1, ctrl_state = f'{1:01b}')
print(ctrl_gate_O_K_c0.definition)
print(ctrl_gate_O_K_c1.definition)

####################################### Oracle O_K_test ###############################################

def oracle_O_K_test(nodes_degrees, test_v, k_bits):
    '''
    Return an oracle O_K gate that encodes (k_v - 1) into qr_k,
    for a fixed v given by test_v. No control on qr_v anymore.
    '''
    qc = QuantumCircuit(k_bits, name=f'O_K_v{test_v}')

    k_v = nodes_degrees[test_v]
    if k_v != 0:
        subgate = IntegerToRegisterGate(f'O_K_val', k_v - 1, k_bits)
        qc.append(subgate, range(k_bits))

    return qc.to_gate(label=f'O_K_v{test_v}')

ctrl_gate_O_K_c0_test = oracle_O_K_test(nodes_degrees, test_v, 2*m).control(num_ctrl_qubits = 1, ctrl_state = f'{0:01b}')
ctrl_gate_O_K_c1_test = oracle_O_K_test(nodes_degrees_c1, test_v, 2*m).control(num_ctrl_qubits = 1, ctrl_state = f'{1:01b}')

################################### Quantum registers ######################################

qr_v = QuantumRegister(n, 'v')  # Basis-encode node index
qr_i = QuantumRegister(2, 'i')  # Basis-encode power (i) of feature value (x^i)
qr_c = QuantumRegister(1, 'c')  # Basis-encode power (c) of matrix A (A^c)
qr_l1 = QuantumRegister(m, 'l1')  # Basis-encode non-zero entries of each row of A
qr_l2 = QuantumRegister(m, 'l2')  # Basis-encode non-zero entries of each row of A^2
qr_u1 = QuantumRegister(n, 'u1')  # Basis-encode node's l-neighbor index of v (auxiliary)
qr_u2 = QuantumRegister(n, 'u2')  # Basis-encode node's l-neighbor index of u (auxiliary)
qr_k = QuantumRegister(2*m, 'k')  # Basis-encode degree value (k) [m for c=0, 2m for c=1]
qr_z = QuantumRegister(p, 'z')  # Basis-encode feature value (x)
qr_a = QuantumRegister(5, 'a')  # Ancilla qubit for rotation

############################ Extra Qubits for Classification State Preparation #################
qr_y = QuantumRegister(1, 'y') # Label qubit (Entangled with v)
qr_a_clas = QuantumRegister(1, 'a_clas') # Classifier ancilla

''' Need n + 8 qubits always (i, c, a). At least, more 2n for u1/u2. Total of >= 3n + 8. '''

########### 0. Create the quantum circuit (Start with all qubits initialized to |0⟩) ########

qc = QuantumCircuit(qr_y, qr_a_clas, qr_a, qr_z, qr_k, qr_u2, qr_u1, qr_l2, qr_l1, qr_c, qr_i, qr_v)

########## 1. Apply Hadamard gates to the `qr_v` qubits to create superposition #############

qc.h(qr_v)
print("\nExpected amplitudes:", 1/math.sqrt(N))

### Alternative:
# w_v = [
#     1 / np.sqrt(2), # |00>
#     1 / np.sqrt(6), # |01>
#     1 / np.sqrt(6), # |10>
#     1 / np.sqrt(6)  # |11>
# ]
# w_v /= np.linalg.norm(w_v)
# w_v_gate = Initialize(w_v)
# w_v_gate = w_v_gate.gates_to_uncompute().inverse()
# print(w_v_gate.definition)
# qc.append(w_v_gate, qr_v)

# theta_init = 2 * np.arctan(1 / np.sqrt(3))  # ≈ π/3

# qc.ry(theta_init, qr_v[0])
# qc.cx(qr_v[0], qr_v[1])
# qc.ry(theta_init, qr_v[1])
# qc.cx(qr_v[0], qr_v[1])
# qc.cz(qr_v[0], qr_v[1])

# print("\nExpected amplitudes:", np.array(w_v))
# access_statevector(qc)

############################ 2. Superposition of the l1-register qubits ######################

qc.h(qr_l1)
print("\nExpected amplitudes: ", 1/math.sqrt(N*s))

################################## 3. Apply Oracle O_L #######################################
################################ ADAPTED FOR HADAMARD CLASSIFIER #############################

qc.h(qr_a_clas)

### Train data
qc.x(qr_a_clas)  # Invert control
qc.append(gate_O_L.control(1), [*qr_a_clas, *qr_l1, *qr_v, *qr_u1])
qc.x(qr_a_clas)  # Undo inversion

### Test Data
qc.append(gate_O_L_test.control(1), [*qr_a_clas, *qr_l1, *qr_u1])

print("\n adjacency_list: ", adjacency_list)
access_statevector(qc)
print(f"\nExpected amplitudes: {1/math.sqrt(2*N*s)}")

########################## 4. Superposition of the c-register qubits #########################

qc.h(qr_c) # For first and second-neighbors calculations

#################################### CONTROLLED ON qr_c = 1 ############################################

################### 5. Superposition of the l2-register qubits controlled on qr_c ######################

qc.ch(control_qubit = qr_c, target_qubit = qr_l2) 
print(f"\nExpected amplitudes for qr_c being |0⟩ or |1⟩: {1/(2 * math.sqrt(N*s))} or {1/(2 * s * math.sqrt(N))}")

############################### 6. Oracle O_L controlled on qr_c ########################################

qc.x(qr_a_clas)  # Invert control
qc.append(gate_O_L.control(2), [*qr_a_clas, *qr_c, *qr_l2, *qr_u1, *qr_u2])
qc.x(qr_a_clas)  # Undo inversion

### Test Data
qc.append(gate_O_L_test.control(2), [*qr_a_clas, *qr_c, *qr_l2, *qr_u2])

# access_statevector(qc) # 111m for (3,1,8)
print("\n adjacency_list: ", adjacency_list)

############################ 7. Superposition of the i-register qubits ######################

qc.h(qr_i)
print(f"\nExpected amplitudes for qr_c being |0⟩ or |1⟩: {1/(4 * math.sqrt(N*s))} or {1/(4 * s * math.sqrt(N))}")

######################################### 8. Oracle O_X ###############################################

qc.append(ctrl_gate_O_X_c0, [*qr_c, *qr_u1, *qr_z])  # qr_c = 0
qc.append(ctrl_gate_O_X_c1, [*qr_c, *qr_u2, *qr_z])  # qr_c = 1

# access_statevector(qc)
print("\nNode index and feature value: "), [print("node (u): ", u, features[u]) for u in range(N)]

######################## 9. Controlled Rotation of Ancilla a1 by F_X ########################

thetas_F_X = [float(2 * np.arccos(x_int / P)) for x_int in range(P)]
F_X = FeatureRotation(thetas_F_X, P, "F_X")

def controlled_F_X(qc, qr_i, qr_a, qr_z, F_X):
    """ Apply the F_X transformation controlled by qr_i conditions.
        Does not need control on qr_c. Only matters qr_i and qr_z. """

    qc.append(F_X, [qr_a[1]] + qr_z[:])

    # Control qr_i conditions for qr_a targets
    control_map = {
        qr_a[2]: ["01", "10", "11"],
        qr_a[3]: ["10", "11"],
        qr_a[4]: ["11"]
    }

    for target, ctrl_states in control_map.items():
        for ctrl_state in ctrl_states:
            subgate_F_X = F_X.control(2, ctrl_state=ctrl_state)
            qc.append(subgate_F_X, [*qr_i, target, *qr_z])

    return qc

controlled_F_X(qc, qr_i, qr_a, qr_z, F_X)

### Access statevector (> 33 min for (3,1,8))

# access_statevector(qc)
print("\nExpected amplitudes (Node u):")
for u in range(N):
    for i in range(1, 5):
        for c in [0, 1]:
            if c == 0:
                amplitude = (features[u])**i / (4 * math.sqrt(N * s))
            if c == 1:
                amplitude = (features[u])**i / (4 * s * math.sqrt(N))
            print(f"Node u = {u}, x_u = {features[u]}, i = {i}, c = {c}, amplitude = {amplitude}")

############################## 10. Oracle O_X_† (reset qr_z) ###############################

qc.append(ctrl_gate_O_X_c0, [*qr_c, *qr_u1, *qr_z])
qc.append(ctrl_gate_O_X_c1, [*qr_c, *qr_u2, *qr_z])

###################################### Second Neighbors ####################################

# 11. Oracle O_L_† (reset qr_u2) (controlled, again)

qc.x(qr_a_clas)  # Invert control
qc.append(gate_O_L.control(2), [*qr_a_clas, *qr_c, *qr_l2, *qr_u1, *qr_u2])
qc.x(qr_a_clas)  # Undo inversion

### Test Data
qc.append(gate_O_L_test.control(2), [*qr_a_clas, *qr_c, *qr_l2, *qr_u2])

# 12. Superposition in l2 (controlled, again)
qc.ch(control_qubit = qr_c, target_qubit = qr_l2) # x sqrt(s) where qr_c=1

###################################### First Neighbors ####################################

# 13. Oracle O_L_† (reset qr_u1)

### Train data
qc.x(qr_a_clas)  # Invert control
qc.append(gate_O_L.control(1), [*qr_a_clas, *qr_l1, *qr_v, *qr_u1])
qc.x(qr_a_clas)  # Undo inversion

### Test Data
qc.append(gate_O_L_test.control(1), [*qr_a_clas, *qr_l1, *qr_u1])

# 14. Superposition in l1
qc.h(qr_l1) # x sqrt(s)

############################### Sum of neighbors' features' results #############################

from collections import defaultdict

expected_amplitudes = defaultdict(lambda: [[0, 0] for _ in range(4)])

for v in range(N):  
    for i in range(1, 5):  
        for u1 in adjacency_list[v]:  # First-level neighbors of v
            expected_amplitudes[v][i - 1][0] += (features[u1] ** i) / (4 * s * math.sqrt(N))  # For u1
            for u2 in adjacency_list[u1]:  # Second-level neighbors of u1
                expected_amplitudes[v][i - 1][1] += (features[u2] ** i) / (4 * s**2 * math.sqrt(N))  # For u2

print("\nExpected amplitudes:")
for v in sorted(expected_amplitudes):
    for i in range(4): 
        print(f"node v = {v}, i = {i}, c = 0, sum_features = {expected_amplitudes[v][i][0]}")
        print(f"node v = {v}, i = {i}, c = 1, sum_features = {expected_amplitudes[v][i][1]}")

#################################### 15. Apply oracle O_K ####################################

# Train
qc.x(qr_a_clas)  # Invert control
qc.append(ctrl_gate_O_K_c0.control(1), [*qr_a_clas, *qr_c, *qr_v, *qr_k])  # qr_c = 0
qc.append(ctrl_gate_O_K_c1.control(1), [*qr_a_clas, *qr_c, *qr_v, *qr_k])  # qr_c = 1
qc.x(qr_a_clas)  # Invert control

# Test
qc.append(ctrl_gate_O_K_c0_test.control(1), [*qr_a_clas, *qr_c, *qr_k])  # qr_c = 0
qc.append(ctrl_gate_O_K_c1_test.control(1), [*qr_a_clas, *qr_c, *qr_k])  # qr_c = 1

######################## 16. Controlled Rotation of Ancilla a1 by F_X ######################## (> 1 min for (2,1,4))

thetas_F_K_inv = [float(2 * np.arccos(1 / (k_int + 1))) for k_int in range(2*s)] # [Times two comes from using 2m qubits]
F_K_inv = FeatureRotation(thetas_F_K_inv, 2*s, "F_K_inv")

qc.append(F_K_inv, [qr_a[0]] + qr_k[:])
# access_statevector(qc, flag_a=False) #, flag_l=False)
print("\nExpected amplitudes:")
for v in sorted(expected_amplitudes):
    for i in range(4): 
        print(f"node v = {v}, i = {i}, c = 0, sum_features = {expected_amplitudes[v][i][0]/nodes_degrees[v]}")
        print(f"node v = {v}, i = {i}, c = 1, sum_features = {expected_amplitudes[v][i][1]/nodes_degrees_c1[v]}")

# Uncompute (reset qr_k)
# Train
qc.x(qr_a_clas)  # Invert control
qc.append(ctrl_gate_O_K_c0.control(1), [*qr_a_clas, *qr_c, *qr_v, *qr_k])  # qr_c = 0
qc.append(ctrl_gate_O_K_c1.control(1), [*qr_a_clas, *qr_c, *qr_v, *qr_k])  # qr_c = 1
qc.x(qr_a_clas)  # Invert control

# Test
qc.append(ctrl_gate_O_K_c0_test.control(1), [*qr_a_clas, *qr_c, *qr_k])  # qr_c = 0
qc.append(ctrl_gate_O_K_c1_test.control(1), [*qr_a_clas, *qr_c, *qr_k])  # qr_c = 1

#############################################################################################################################
##################################################### CLASSIFIER ############################################################
#############################################################################################################################

# Threshold-based labels: 0 if x < 0.75, else 1
th_class = 0.75
class_labels = [0 if x < th_class else 1 for x in features] # run circuit.py before this
print("Class labels:", class_labels) 
print("Features:", features)
    
########################################## Oracle O_Y ###############################################

def oracle_O_Y(class_labels, n):
    ''' Return an oracle O_Y Gate that encodes y_v into qr_y, controlled on qr_v. '''
    qc = QuantumCircuit(n + 1, name='O_Y')

    for v, y_v in enumerate(class_labels):
        subgate = IntegerToRegisterGate('O_Y_val', y_v, 1)
        ctrl_gate = subgate.control(num_ctrl_qubits = n, ctrl_state = f'{v:0{n}b}')
        qc.append(ctrl_gate, list(range(n + 1)))
            
    return qc.to_gate(label='O_Y')

gate_O_Y = oracle_O_Y(class_labels, n)

################################## 1. Apply Oracle O_Y #######################################

qc.append(gate_O_Y, [*qr_v, *qr_y])
print("\n class_labels: ", class_labels)

#################################### Hadamard - Swap - Hadamard ##################################

# 16. Hadamard
qc.h(qr_a_clas) 
qc.barrier()

###################################### Two single-qubit measurements ##################################

# Measure test qubits into classical bits (classical registers to store results)
cr_a = ClassicalRegister(1, 'cr_a') # for ancilla qubit
cr_y = ClassicalRegister(1, 'cr_y') # for label qubit
qc.add_register(cr_a)
qc.add_register(cr_y)
# Leftmost bit corresponds to the last classical register measured
qc.measure(qr_a_clas, cr_a)
qc.measure(qr_y, cr_y)

######################################### Simulate (Qiskit 2.0.1) #####################################

# Fully decompose all custom gates (including those inside oracles)
qc = qc.decompose() # from ctrl-O_K to O_K
qc = qc.decompose() # from O_K to IntegertoRegister
qc = qc.decompose() # from IntegerToRegister to basic gates
qc = qc.decompose()
qc = qc.decompose()
qc = qc.decompose()

# Simulate
from qiskit_aer import AerSimulator
simulator = AerSimulator() #method='automatic') # Small circuit -> statevector method (gives wavefunction, no shots). Might switch to qasm.
print(simulator.configuration())

# rewriting circuit to match the topology of a specific quantum device and/or to optimize the circuit for execution
qc_transpiled = transpile(qc, basis_gates=['u3', 'cx'], optimization_level=1) # simulator)
nr_shots = 100000
job = simulator.run(qc_transpiled, shots=nr_shots)
print("Gonna get the result...")
result = job.result() # unknown instruction: O_L unless I decompose

# Results
if result.success:
    print("Available keys in result:", result.results)
    print("Experiment result keys:", [res.header.name for res in result.results])
    # counts = result.get_counts()
    counts = result.get_counts(result.results[0].header.name)
    print("Counts:", counts)
else:
    print("Job failed:", result.status)

# Expectation Value
def f_expectation_value(counts):
        shots = sum(counts.values())
        return (
            counts.get('0 0', 0) - counts.get('0 1', 0)
            - counts.get('1 0', 0) + counts.get('1 1', 0)
        ) / float(shots)

exp_val = f_expectation_value(counts)
pred_label = 0 if exp_val > 0 else 1

print(f"Expectation Value: {exp_val}. Predicted Label: {pred_label}. Test Node: {test_v}")