# CirQ

CirQ is an open-source quantum computing library developed in Python by Google. It allows you to build, optimize, simulate, and execute quantum circuits.  Specifically, CirQ enables the simulation of circuits on specific qubit configurations, which can optimize a circuit for a particular qubit architecture. Information on the library's features is available in the [documentation](link_to_documentation) and on CirQ's [GitHub](link_to_github).  Similar to Snowflurry, CirQ can be used to run quantum circuits on the MonarQ quantum computer.


## Installation

The CirQ quantum computer simulator is available on all our clusters. The Python programming language must be loaded before accessing it. It is best to work within a Python virtual environment.

```bash
[name@server ~]$ module load python/3.11
[name@server ~]$ virtualenv --no-download --clear ~/ENV && source ~/ENV/bin/activate
[name@server ~]$ pip install --no-index --upgrade pip
[name@server ~]$ pip install --no-index cirq==1.4.1
[name@server ~]$ python -c "import cirq"
[name@server ~]$ pip freeze > cirq-1.4.1-reqs.txt
```

The last command creates a file named `cirq-1.4.1-reqs.txt`, which you can reuse in a job script, as described below.


## Running on a Cluster

**File: `script.sh`**

```bash
#!/bin/bash
#SBATCH --account=def-someuser # specify your account name
#SBATCH --time=00:15:00        # modify as needed
#SBATCH --cpus-per-task=1      # modify as needed
#SBATCH --mem-per-cpu=1G       # modify as needed

# Load module dependencies.
module load StdEnv/2023 gcc python/3.11

# Create the virtual environment in $SLURM_TMPDIR.
virtualenv --no-download ${SLURM_TMPDIR}/env
source ${SLURM_TMPDIR}/env/bin/activate

# Install CirQ and its dependencies.
pip install --no-index --upgrade pip
pip install --no-index --requirement ~/cirq-1.4.1-reqs.txt

# Run the CirQ program.
python cirq_example.py
```

You can then [submit your job to the scheduler](link_to_scheduler_submission).


## Example Usage: Bell States

Bell states are the simplest states that allow explaining both superposition and entanglement on qubits.

The CirQ library allows you to construct a Bell state like this:

```python
[name@server ~]$ python
python> import cirq
python> from cirq.contrib.svg import SVGCircuit
python> from cirq import H, CNOT

python> qubits = cirq.LineQubit.range(2)
python> circuit = cirq.Circuit(H.on(qubits[0]),CNOT.on(qubits[0],qubits[1]))
python> circuit.append(cirq.measure(qubits, key='m'))
python> SVGCircuit(circuit)
```

This code constructs and displays a circuit that prepares a Bell state. The H gate (Hadamard gate) creates an equal superposition of |0⟩ and |1⟩ on the first qubit while the CNOT gate (controlled X gate) creates entanglement between the two qubits. This Bell state is therefore an equal superposition of the states |00⟩ and |11⟩. Simulating this circuit using CirQ allows visualizing the results. In this diagram, the integer 3 represents the state |11⟩ since 3 is written 11 in binary.

```python
[name@server ~]$ python
python> import matplotlib.pyplot as plt
python> s = cirq.Simulator().run(circuit, repetitions=1000)
python> counts = s.histogram(key='m')
python> cirq.plot_state_histogram(counts, plt.subplot())
```

**(Retrieved from "https://docs.alliancecan.ca/mediawiki/index.php?title=CirQ&oldid=161728")**
