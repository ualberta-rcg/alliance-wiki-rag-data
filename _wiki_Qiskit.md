# Qiskit

Qiskit is an open-source quantum programming library developed in Python by IBM. Like PennyLane and Snowflurry, it allows you to build, simulate, and run quantum circuits.

## Contents

* [Installation](#installation)
* [Running Qiskit on a Cluster](#running-qiskit-on-a-cluster)
* [Using Qiskit with MonarQ (To Come)](#using-qiskit-with-monarq-to-come)
* [Example Use: Bell States](#example-use-bell-states)

## Installation

1. Load Qiskit's dependencies:

```bash
name@server ~]$ module load StdEnv/2023 gcc python/3.11 symengine/0.11.2
```

2. Create and activate a Python virtual environment:

```bash
name@server ~]$ virtualenv --no-download --clear ~/ENV && source ~/ENV/bin/activate
```

3. Install a specific version of Qiskit:

```bash
(ENV) [name@server ~] pip install --no-index --upgrade pip
(ENV) [name@server ~] pip install --no-index qiskit==X.Y.Z qiskit_aer==X.Y.Z
```

Where `X.Y.Z` represents the version number, for example, `1.4.0`. To install the latest version available on our clusters, do not specify a version. Here, we only imported `qiskit` and `qiskit_aer`. You can add other Qiskit software according to your needs following the structure `qiskit_package==X.Y.Z` where `qiskit_package` represents the desired software, for example, `qiskit-finance`. Currently available wheels are listed on the [Python Wheels](link_to_wheels_page_here) page.

4. Validate the Qiskit installation:

```bash
(ENV)[name@server ~] python -c 'import qiskit'
```

5. Freeze the environment and dependencies:

```bash
(ENV)[name@server ~] pip freeze --local > ~/qiskit_requirements.txt
```

## Running Qiskit on a Cluster

**File:** `script.sh`

```bash
#!/bin/bash
#SBATCH --account=def-someuser # Specify your account name
#SBATCH --time=00:15:00        # Modify if needed
#SBATCH --cpus-per-task=1      # Modify if needed
#SBATCH --mem-per-cpu=1G       # Modify if needed

# Load module dependencies.
module load StdEnv/2023 gcc python/3.11 symengine/0.11.2

# Generate the virtual environment in $SLURM_TMPDIR.
virtualenv --no-download ${SLURM_TMPDIR}/env
source ${SLURM_TMPDIR}/env/bin/activate

# Install Qiskit and its dependencies.
pip install --no-index --upgrade pip
pip install --no-index --requirement ~/qiskit_requirements.txt

# Modify the Qiskit program.
python qiskit_example.py
```

You can then submit your job to the scheduler.


## Using Qiskit with MonarQ (To Come)


## Example Use: Bell States

We will create the first Bell state on Narval in simulation. First, we need to import the necessary modules.

```python
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
```

Next, we define the circuit. We apply a Hadamard gate to create a superposition state on the first qubit and then apply a CNOT gate to entangle the first and second qubits.

```python
circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0,1)
circuit.measure_all()
```

We specify the simulator we want to use. `AerSimulator` being the default simulator. We obtain the count of the final states of the qubits after 1000 measurements.

```python
simulator = AerSimulator()
result = simulator.run(circuit, shots=1000).result()
counts = result.get_counts()
print(counts)
# {'00': 489, '11': 535}
```

We display a histogram of the results with the command `plot_histogram(counts)`.

Histogram of the results of 1000 measurements on the first Bell state.


**(Retrieved from "https://docs.alliancecan.ca/mediawiki/index.php?title=Qiskit&oldid=173686")**
