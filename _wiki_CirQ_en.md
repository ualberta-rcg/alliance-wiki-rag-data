# CirQ

Developed by Google, CirQ is an open-source quantum computing library to build, optimize, simulate, and run quantum circuits.  More specifically, CirQ allows simulation of circuits on particular qubit configurations, which can optimize a circuit for a certain qubit architecture. Information on the features can be found in the [CirQ documentation](link_to_documentation) and [GitHub](link_to_github). Like Snowflurry, CirQ can be used to run quantum circuits on the MonarQ quantum computer.


## Installation

The CirQ simulator is available on all clusters. To access it, load the Python language. It is best to work in a Python virtual environment.

```bash
module load python/3.11
virtualenv --no-download --clear ~/ENV && source ~/ENV/bin/activate
pip install --no-index --upgrade pip
pip install --no-index cirq==1.4.1
python -c "import cirq"
pip freeze > cirq-1.4.1-reqs.txt
```

The last command creates the `cirq-1.4.1-reqs.txt` file, which can also be used in a job script (see example below).


## Running on a Cluster

**File: script.sh**

```bash
#!/bin/bash
#SBATCH --account=def-someuser # Modify with your account name
#SBATCH --time=00:15:00        # Modify as needed
#SBATCH --cpus-per-task=1      # Modify as needed
#SBATCH --mem-per-cpu=1G       # Modify as needed

# Load modules dependencies.
module load StdEnv/2023 gcc python/3.11

# Generate your virtual environment in $SLURM_TMPDIR.
virtualenv --no-download ${SLURM_TMPDIR}/env
source ${SLURM_TMPDIR}/env/bin/activate

# Install CirQ and its dependencies.
pip install --no-index --upgrade pip
pip install --no-index --requirement ~/cirq-1.4.1-reqs.txt

# Edit with your CirQ program.
python cirq_example.py
```

You can then submit your job to the scheduler.


## Use Case: Bell States

Bell states are the simplest states that explain both superposition and entanglement on qubits. The CirQ library allows you to construct a Bell state like this:

```python
import cirq
from cirq.contrib.svg import SVGCircuit
from cirq import H, CNOT

qubits = cirq.LineQubit.range(2)
circuit = cirq.Circuit(H.on(qubits[0]),CNOT.on(qubits[0],qubits[1]))
circuit.append(cirq.measure(qubits, key='m'))
SVGCircuit(circuit)
```

This code builds and displays a circuit that prepares a Bell state. The H gate (Hadamard gate) creates an equal superposition of |0⟩ and |1⟩ on the first qubit, while the CNOT gate (controlled X gate) creates entanglement between the two qubits. This Bell state is therefore an equal superposition of the states |00⟩ and |11⟩. Simulating this circuit using CirQ allows you to visualize the results. In this diagram, the integer 3 represents the state |11⟩ since 3 is written 11 in binary.

```python
import matplotlib.pyplot as plt
s = cirq.Simulator().run(circuit, repetitions=1000)
counts = s.histogram(key='m')
cirq.plot_state_histogram(counts, plt.subplot())
```

**(Remember to replace `link_to_documentation` and `link_to_github` with the actual links.)**
