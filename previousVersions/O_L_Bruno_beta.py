### Import modules
#from qiskit import Aer, execute  
from qiskit.visualization import circuit_drawer
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RYGate, UnitaryGate
from qiskit.quantum_info import Statevector
from qiskit.circuit import Gate, Parameter

import numpy as np
import math



### Parameters (4,2,16)
n = 4 # Number of qubits for |j⟩
m = 2 # Number of qubits for |k⟩ (m << n)
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

#Bruno: fiz esta gate para simplificar o codito
class IntegerToRegisterGate(Gate):
    def __init__(self, integer, num_qubits):
        # Check if integer can be represented with given qubits
        max_value = 2**num_qubits - 1
        if integer < 0 or integer > max_value:
            raise ValueError(f"Integer {integer} cannot be represented with {num_qubits} qubits")
        
        # Fix: Put integer in a list for params
        super().__init__('IntToReg', num_qubits, [integer])  # Changed this line
        self.integer = integer
        self.num_qubits = num_qubits

    def _define(self):
        qc = QuantumCircuit(self.num_qubits)
        binary = format(self.integer, f'0{self.num_qubits}b')
        
        for i, bit in enumerate(reversed(binary)):
            if bit == '1':
                qc.x(i)
                
        self._definition = qc
        
    def inverse(self):
        return IntegerToRegisterGate(self.integer, self.num_qubits)

 
###################################### Quantum Circuit ######################################
################################### 0. Quantum registers ####################################

qr_l = QuantumRegister(m, 'l')  # Basis-encode non-zero entries of each row of A
qr_v = QuantumRegister(n, 'v')  # Basis-encode node index  # Basis-encode node index
qr_k = QuantumRegister(m, 'k')  # Basis-encode degree value (k)
qr_z = QuantumRegister(p, 'z')  # Basis-encode feature value (x)
qr_a1 = QuantumRegister(1, 'a1')  # Ancilla qubit for rotation (feature value)
qr_a2 = QuantumRegister(1, 'a2')  # Ancilla qubit for rotation (degree inverse)

########### 1. Create the quantum circuit (Start with all qubits initialized to |0⟩) #########
########## 2. Apply Hadamard gates to the `qr_v` qubits to create superposition #############
Adjacency_list=[[1,12,15,15],[4,5,8],[6,8]] #31 e 


qc = QuantumCircuit(qr_z,qr_k,qr_v,qr_l,qr_a1,qr_a2)#reondernei a ordem do qubits

  # n=4 from your parameters

qc.h(qr_l)

print(qc)
print("\n6. test:"), access_statevector(qc)

# com a nova gate o oracle_O_L fica mais simples. 
# estou apenas a fazer O_L sem superposicao de estados e nos
# apenas para nodes 0
# Depois tens de usar a mesma idea para lidar com superposicao de nodes inicias nos nos
# Acho que vais precisar de dois regist de nodos, um do nodo e um do vinho, sem isso o condigo fica complicado
#, porque o controlo sera igual ao node onde re sergista 
def oracle_O_L(qc,qr_l, qr_v, Adjacency_list):
#   for n in range len(N): #vamos precisar de um segundo ciclo
    for i,integer in enumerate(Adjacency_list[0]):
        print(i,integer)
        my_gate = IntegerToRegisterGate(integer, n)
        controlled_gate = my_gate.control(num_ctrl_qubits=m,ctrl_state=i)# quando tiveres nodes como controlo, punha aqui um segundo controlo
        qc.append(controlled_gate, [*qr_l, *qr_v])

#esta construcao nao e eficiente, mas estava a pensar que vamos assumir isto como o nosso oracle, e medir quantas vezes chama-mos 0_L
oracle_O_L(qc,qr_l, qr_v, Adjacency_list)
print("\n6. test:"), access_statevector(qc)


print("\n6. test:"), access_statevector(qc)