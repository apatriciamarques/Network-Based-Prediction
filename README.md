# Quantum Multi-Order Moment Embedding (QMME) Protocol for Graph-Based Node Classification

## Quantum Network-Based Prediction of Cancer Driver Genes

The QMME protocol encodes higher-order neighborhood statistics into amplitude-encoded quantum states. These states are then classified using a fully quantum, kernel-based binary classification circuit.

> Paper: _Quantum Multi-Order Moment Embedding for Graph-Based Node Classification_  
> Patrícia Marques, Andreas Wichert, Bruno Coutinho (2025)

## Key Features

- End-to-end quantum pipeline: embedding + classification
- No classical training or optimization
- Logarithmic qubit scaling, polynomial-depth circuits
- Tested on synthetic sparse graphs and on real protein-protein interaction networks

## Requirements

- Python 3.9+
- [Qiskit](https://qiskit.org/)
- NumPy / SciPy
- Matplotlib

---

## Run C++ Project on the Server

Follow these steps to build and run the **C++ pipeline implementation** of the QMME protocol:

### One-time setup
We installed dependencies (Boost 1.86.0 + cnpy) into `~/.local`.  

### Build & Run

```bash
mkdir build → only needed the first time
cd build → always before building
cmake .. → tell CMake where the project files are (previous folder)
make -j$(nproc) → compiles the Pipeline program (CMakeLists has compile settings)
./Pipeline → runs the binary program Pipeline
