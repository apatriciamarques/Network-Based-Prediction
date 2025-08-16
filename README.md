# Quantum Multi-Order Moment Embedding (QMME) Protocol for Graph-Based Node Classification

The QMME protocol encodes higher-order neighborhood statistics into amplitude-encoded quantum states. These states are then classified using a fully quantum, kernel-based binary classification circuit.

> Paper: _Quantum Multi-Order Moment Embedding for Graph-Based Node Classification_  
> Patrícia Marques, Andreas Wichert, Bruno Coutinho (2025)

## Key Features

- End-to-end quantum pipeline: embedding + classification
- No classical training or optimization
- Logarithmic qubit scaling, polynomial-depth circuits
- Tested on synthetic sparse graphs

## Requirements

- Python 3.9+
- [Qiskit](https://qiskit.org/)
- NumPy / SciPy
- Matplotlib
