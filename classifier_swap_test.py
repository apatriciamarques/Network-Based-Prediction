#!/usr/bin/env python
# coding: utf-8

# # Definitions

# In[ ]:


### Import Modules
# %matplotlib inline
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import Gate
from qiskit.visualization import circuit_drawer
import random
# import nbimporter
# import circuit

## Test Node (Note that I'm using a test node which is also in the train data)
test_v = 0

## Import Circuit # Option 1: perform QMME
get_ipython().run_line_magic('run', './circuit.ipynb')


# ## Labels Options

# In[284]:


### Known/Train Labels Option 1: Random

## Known Labels (suppose we know the label of the test sample)
# class_labels = [random.randint(0, 1) for _ in range(2 * n)]
# print("\nclass labels: ", class_labels)


# In[285]:


### Known/Train Labels Option 2: Threshold for Cancer (y = f(x))

# Threshold-based labels: 0 if x < 0.75, else 1
th_class = 0.75
class_labels = [0 if x < th_class else 1 for x in features] # run circuit.py before this
print("Class labels:", class_labels) 
print("Features:", features)


# # Simulation

# ## Count Probabilities

# In[286]:


######################################### Simulate (Qiskit 2.0.1) #####################################
import matplotlib.pyplot as plt

def measurement_results(qc, nr_shots = 10000000, type = 'include_l'):
    ''' Simulate and get results (distribution/probabilities). '''
    # Measure test qubits into classical bits (classical registers to store results)
    if type == 'mid-results':
        len_zeros = 5
        cr_test = ClassicalRegister(3 + len_zeros, 'cr_test')
        qc.add_register(cr_test)
        qc.measure(qr_i_test[:] + qr_c_test[:] + qr_a_test[:], cr_test[:])
    elif type == 'include_l':
        len_zeros = 5 + 2 * m
        cr_test = ClassicalRegister(3 + len_zeros, 'cr_test')
        qc.add_register(cr_test)
        qc.measure(qr_i_test[:] + qr_c_test[:] + qr_l1_test[:] + qr_l2_test[:] + qr_a_test[:], cr_test[:])
    elif type == 'include_l_u':
        len_zeros = 5 
        cr_test = ClassicalRegister(3 + len_zeros + 2 * n + 2 * m, 'cr_test')
        qc.add_register(cr_test)
        qc.measure(qr_u1[:] + qr_u2[:] + qr_i_test[:] + qr_c_test[:] + qr_l1_test[:] + qr_l2_test[:] + qr_a_test[:], cr_test[:])
        # Read as: [a][a][a][a][a] [l2] [l1] [c] [i][i] [u2][u2] [u1][u1].
    else:
        raise ValueError("Type must be 'mid-results' or 'include_l'")
    
    # Fully decompose all custom gates (including those inside oracles)
    qc = qc.decompose() # from ctrl-O_K to O_K
    qc = qc.decompose() # from O_K to IntegertoRegister
    qc = qc.decompose() # from IntegerToRegister to basic gates
    qc = qc.decompose()
    qc = qc.decompose()
    qc = qc.decompose()

    # Simulate
    from qiskit_aer import AerSimulator
    simulator = AerSimulator()
    job = simulator.run(qc, shots=nr_shots)
    result = job.result()

    # Results
    counts = result.get_counts()
    print("Counts:", counts)

    # Plot 1 – Full distribution
    from qiskit.visualization import plot_distribution
    plot_distribution(counts, figsize=(40, 6), bar_labels=True)
    plt.show()

    # Plot 2 – Only when a_test = '00000'
    from qiskit.visualization import plot_histogram
    probs_simulation = {
        bitstring: count / nr_shots
        for bitstring, count in counts.items()
        if bitstring[:len_zeros] == '0' * len_zeros
    }

    plot_histogram(probs_simulation, figsize=(40, 6), bar_labels=True)
    plt.show()


# ## Integer to Register Gate

# In[287]:


class IntegerToRegisterGate(Gate):
    ''' Gate to take an integer and encode it into a quantum register. '''
    def __init__(self, name, integer, num_qubits):
        max_value = 2**num_qubits - 1
        if integer < 0 or integer > max_value:
            raise ValueError(f"Integer {integer} cannot be represented with {num_qubits} qubits")
        
        super().__init__(name, num_qubits, [integer]) # Initialize gate
        self.integer = integer
        self.num_qubits = num_qubits

    def _define(self): # Behaviour/operations of the gate
        qc = QuantumCircuit(self.num_qubits)
        binary = format(self.integer, f'0{self.num_qubits}b')
        [qc.x(i) for i, bit in enumerate(reversed(binary)) if bit == '1']
        self._definition = qc
        
    def inverse(self):
        return IntegerToRegisterGate(self.integer, self.num_qubits)


# ## Oracle (Basis-Encoder)

# In[288]:


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


# In[289]:


######################################## Oracle O_L_test #############################################

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


# In[290]:


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


# # Preparation State

# ## Initialization

# In[291]:


############################ Extra Qubits for Classification State Preparation #################
############# Add 1 + 3 + 5 + 2m = 9 + 2m More Qubits (Doesn't Scale with N, but with s) #######

qr_y = QuantumRegister(1, 'y') # Label qubit (Entangled with v)
qr_a_clas = QuantumRegister(1, 'a_clas') # Classifier ancilla
### Test Node Neighbors (register l (QMME) not always 0s)
qr_l1_test = QuantumRegister(m, 'l1_test')
qr_l2_test = QuantumRegister(m, 'l2_test')
### Test Node Features (8 "augmented" by ancilla rotations = 256)
qr_i_test = QuantumRegister(2, 'i_test')
qr_c_test = QuantumRegister(1, 'c_test')
qr_a_test = QuantumRegister(5, 'a_test')


# In[292]:


##################################### Option 1: With QMME ############################################

qc.add_register(qr_y)
qc.add_register(qr_a_clas)
qc.add_register(qr_l1_test)
qc.add_register(qr_l2_test)
qc.add_register(qr_i_test)
qc.add_register(qr_c_test)
qc.add_register(qr_a_test)

############################## Option 2: Without QMME (for Testing) ##################################

## Initialize registers at zeros without QMME
## Need to comment where O_Y is applied

# qc = QuantumCircuit(qr_a_test, qr_z, qr_k, qr_u2, qr_u1, qr_l2_test, qr_l1_test, qr_c_test, qr_i_test) #, qr_v)
    


# ## Access Neighbors

# In[293]:


################################## 1. Apply Oracle O_Y #######################################

qc.append(gate_O_Y, [*qr_v, *qr_y])
print("\n class_labels: ", class_labels)

############################ 2. Superposition of the l1-register qubits ######################

qc.h(qr_l1_test)
print("\nExpected amplitudes: ", 1/math.sqrt(s))

################################ 3. Apply Oracle O_L_test ####################################

qc.append(gate_O_L_test, [*qr_l1_test, *qr_u1])
print("\n adjacency_list: ", adjacency_list)

######################## 4. Superposition of the c_test-register qubits ######################

qc.h(qr_c_test) # For first and second-neighbors calculations


# In[294]:


#################################### CONTROLLED ON qr_c = 1 ############################################

############## 5. Superposition of the l2-register qubits controlled on qr_c_test ######################

qc.ch(control_qubit = qr_c_test, target_qubit = qr_l2_test) 

######################### 6. Oracle O_L controlled on qr_c_test ########################################

qc.append(ctrl_gate_O_L, [qr_c_test, *qr_l2_test, *qr_u1, *qr_u2])


# ## Feature Encoding

# In[295]:


############################ 7. Superposition of the i_test-register qubits ######################

qc.h(qr_i_test) # c = 0: p = 0.167 norm = 0,125 / c = 1: p = 0.083 norm = 0,063 (as expected)

##################################### 8. Oracle O_X ###############################################

qc.append(ctrl_gate_O_X_c0, [*qr_c_test, *qr_u1, *qr_z])  # qr_c = 0
qc.append(ctrl_gate_O_X_c1, [*qr_c_test, *qr_u2, *qr_z])  # qr_c = 1

# print(1/(4 * math.sqrt(s)), 1/(2 * math.sqrt(2 * s)))


# In[296]:


######################## 9. Controlled Rotation of Ancilla a1 by F_X ########################

controlled_F_X(qc, qr_i_test, qr_a_test, qr_z, F_X)
# measurement_results(qc)
# access_statevector_prep(qc)


# In[297]:


##################################### Expected Amplitudes (right) ##########################################

print("\nExpected amplitudes (Node u):")
for u in range(N):
    for i in range(1, 5):
        for c in [0, 1]:
            if c == 0:
                amplitude = (features[u])**i / (2 * math.sqrt(2 * s))
            if c == 1:
                amplitude = (features[u])**i / (2 * s * math.sqrt(2))
            print(f"Node u = {u}, x_u = {features[u]}, i = {i}, c = {c}, amplitude = {amplitude}")

########################################### Expected Probabilies ##########################################

# probs_00000 = []
# for u in range(N):
#     for i in range(1, 5):
#         for c in [0, 1]:
#             if c == 0:
#                 prob = (features[u])**(2*i) / (8 * s)
#             if c == 1:
#                 prob = (features[u])**(2*i) / (8 * s**2)
#             probs_00000.append((u, i, c, prob))

# for (u, i, c, prob) in probs_00000:
#     norm = prob / sum(p[3] for p in probs_00000)
#     print(f"Node u = {u}, x_u = {features[u]}, i = {i}, c = {c}, prob = {prob}, norm = {norm}")


# ## Uncomputation

# In[298]:


############################## 10. Oracle O_X_† (reset qr_z) ###############################

qc.append(ctrl_gate_O_X_c0, [*qr_c_test, *qr_u1, *qr_z])
qc.append(ctrl_gate_O_X_c1, [*qr_c_test, *qr_u2, *qr_z])

###################################### Second Neighbors ###################################

# 11. Oracle O_L_† (reset qr_u2) (controlled, again)
qc.append(ctrl_gate_O_L, [qr_c_test, *qr_l2_test, *qr_u1, *qr_u2])
# 12. Superposition in l2 (controlled, again)
qc.ch(control_qubit = qr_c_test, target_qubit = qr_l2_test) # x sqrt(s) where qr_c=1

###################################### First Neighbors ####################################

# 13. Oracle O_L_† (reset qr_u1)
qc.append(gate_O_L_test, [*qr_l1_test, *qr_u1])
# 14. Superposition in l1
qc.h(qr_l1_test) # x sqrt(s)


# In[299]:


# for i in range(4): 
#     print(f"node v = {test_v}, i = {i}, c = 0, sum_features = {expected_amplitudes[test_v][i][0] * math.sqrt(N)}")
#     print(f"node v = {test_v}, i = {i}, c = 1, sum_features = {expected_amplitudes[test_v][i][1] * math.sqrt(N)}")

# access_statevector_prep(qc)


# ## Division by Degree

# In[300]:


############################# 15. Apply oracle O_K_test ####################################

qc.append(ctrl_gate_O_K_c0_test, [*qr_c_test, *qr_k])  # qr_c = 0
qc.append(ctrl_gate_O_K_c1_test, [*qr_c_test, *qr_k])  # qr_c = 1


# In[301]:


######################## 16. Controlled Rotation of Ancilla a1_test by F_X ########################

qc.append(F_K_inv, [qr_a_test[0]] + qr_k[:])

# Uncompute (reset qr_k)
qc.append(ctrl_gate_O_K_c0_test, [*qr_c_test, *qr_k])  # qr_c = 0
qc.append(ctrl_gate_O_K_c1_test, [*qr_c_test, *qr_k])  # qr_c = 1


# In[302]:


################################ If the circuit starts with zeros #################################

for i in range(4): 
    print(f"node v = {test_v}, i = {i}, c = 0, sum_features = {expected_amplitudes[test_v][i][0] * math.sqrt(N) / nodes_degrees[test_v]}")
    print(f"node v = {test_v}, i = {i}, c = 1, sum_features = {expected_amplitudes[test_v][i][1] * math.sqrt(N) / nodes_degrees_c1[test_v]}")


# In[303]:


# access_statevector_prep(qc)
# measurement_results(qc)


# # Classifier Circuit

# ## Swap-Test Circuit

# In[304]:


#################################### Hadamard - Swap - Hadamard ##################################

# 16. Hadamard
qc.h(qr_a_clas) 
# 17. Multi-controlled Swap
# Controlled-SWAP gate, also known as the Fredkin gate.
qc.cswap(qr_a_clas, qr_i[0], qr_i_test[0])
qc.cswap(qr_a_clas, qr_i[1], qr_i_test[1])
qc.cswap(qr_a_clas, qr_c, qr_c_test)
qc.cswap(qr_a_clas, qr_a[0], qr_a_test[0])
qc.cswap(qr_a_clas, qr_a[1], qr_a_test[1])
qc.cswap(qr_a_clas, qr_a[2], qr_a_test[2])
qc.cswap(qr_a_clas, qr_a[3], qr_a_test[3])
qc.cswap(qr_a_clas, qr_a[4], qr_a_test[4])
# 18. Hadamard
qc.h(qr_a_clas) 
qc.barrier()


# ## Measurements

# In[305]:


import matplotlib.pyplot as plt

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
simulator = AerSimulator() # method='qasm') #method='automatic') # Small circuit -> statevector method. Might switch to qasm.
print(simulator.configuration())

# rewriting circuit to match the topology of a specific quantum device and/or to optimize the circuit for execution
qc_transpiled = transpile(qc, basis_gates=['u3', 'cx'], optimization_level=1) # simulator)
nr_shots = 100000
job = simulator.run(qc_transpiled, shots=nr_shots)
result = job.result() # unknown instruction: O_L unless I decompose

# Results
counts = result.get_counts()
print("Counts:", counts)


# In[ ]:


def expectation_value(counts):
    """
    Calculate the two-qubit expectation value <σ_z(a),σ_z(l)>.
    From a dictionary to a string.
    """
    shots = sum(counts.values())
    return (
        counts.get('00', 0)
        - counts.get('01', 0)
        - counts.get('10', 0)
        + counts.get('11', 0)
    ) / float(shots)


# In[ ]:


# Plot 1 – Full distribution
from qiskit.visualization import plot_distribution
plot_distribution(counts, figsize=(40, 6), bar_labels=True)
plt.show()


# # Draw and Visualize

# In[ ]:


###################################### Draw and Visualize ########################################

# print("\n -------------------------------------------------------------------------------\n")
# circuit_drawer(qc, output='mpl', filename='network_circuit.png')
# qc.draw(output='mpl', filename='network_circuit.png')

