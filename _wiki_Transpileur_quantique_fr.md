# Quantum Transpiler

Classical transpilation describes the translation of code written in a certain programming language into code written in another language. This is a process analogous to compilation.

In the context of quantum computing, transpilation aims to ensure that a quantum circuit uses only the native gates of the quantum machine on which it will be executed. Transpilation also ensures that multi-qubit operations are assigned to physically connected qubits on the quantum chip.


## Transpilation Steps

### Measurement Decomposition

Measurements are performed in a given basis, such as the X, Y, or Z bases, among others. Most quantum computers measure in the Z basis (computational basis). If another basis is required, rotations must be added at the end of the circuit to adjust the measurement basis.

### Intermediate Decomposition

An initial decomposition of the operations is necessary to execute the circuit on a quantum machine in order to limit the number of different operations used by the circuit. For example, operations with more than two qubits must be decomposed into two-qubit or one-qubit operations.

### Placement

The idea is to establish an association between the wires of the created quantum circuit and the physical qubits of the machine. This step can be reduced to a subgraph isomorphism problem.

### Routing

Despite the placement step, it is possible that some two-qubit operations cannot be correctly assigned to physical couplers available on the machine. In this case, swap operations are used to virtually bring the qubits concerned closer together and allow their connection. However, these swap operations are very costly, making an optimal initial placement essential to minimize their use.

Example of routing to join two distant qubits. A CNOT gate between qubit 0 and 2 is converted into two SWAP gates and a CNOT gate on neighboring qubits.

### Optimization

Qubits accumulate errors and lose their coherence over time. To limit these effects, the optimization process reduces the number of operations applied to each qubit using different classical algorithms. For example, it removes trivial operations and inverse operations; combines rotations on the same axis; and more generally, replaces sections of circuits with equivalent circuits, generating fewer errors.

### Native Gate Decomposition

Each quantum computer has a finite set of basic operations (native gates), from which all other operations can be composed. For example, MonarQ has a set of 13 native gates. Transpilation thus decomposes all non-native operations of a circuit into native operations.


## Using the Calcul Québec Transpiler with MonarQ

Calcul Québec has developed a transpiler that allows circuits to be sent to MonarQ transparently, using the transpilation steps mentioned above. This transpiler is integrated into a PennyLane device and is therefore designed to be used specifically with PennyLane. For details, see the documentation.
