# Qiskit

Developed in Python by IBM, Qiskit is an open-source quantum computing library. Like PennyLane and Snowflurry, it allows you to build, simulate and run quantum circuits.

## Installation

1. Load the Qiskit dependencies.

```bash
[name@server ~]$ module load StdEnv/2023 gcc python/3.11 symengine/0.11.2
```

2. Create and activate a Python virtual environment.

```bash
[name@server ~]$ virtualenv --no-download --clear ~/ENV && source ~/ENV/bin/activate
```

3. Install a version of Qiskit.

```bash
(ENV) [name@server ~] pip install --no-index --upgrade pip
(ENV) [name@server ~] pip install --no-index qiskit==X.Y.Z qiskit_aer==X.Y.Z
```

where `X.Y.Z` is the version number, for example `1.4.0`. To install the most recent version available on our clusters, do not specify a number. Here, we only imported `qiskit` and `qiskit_aer`. You can add other Qiskit software with the syntax `qiskit_package==X.Y.Z` where `qiskit_package` is the software name, for example `qiskit-finance`. To see the wheels that are currently available, see [Available Python wheels](link_to_available_wheels).

4. Validate the installation.

```bash
(ENV)[name@server ~] python -c 'import qiskit'
```

5. Freeze the environment and its dependencies.

```bash
(ENV)[name@server ~] pip freeze --local > ~/qiskit_requirements.txt
```

## Running Qiskit on a cluster

**File:** `script.sh`

```bash
#!/bin/bash
#SBATCH --account=def-someuser #Modify with your account name
#SBATCH --time=00:15:00        #Modify as needed
#SBATCH --cpus-per-task=1      #Modify as needed
#SBATCH --mem-per-cpu=1G       #Modify as needed

# Load module dependencies.
module load StdEnv/2023 gcc python/3.11 symengine/0.11.2

# Generate your virtual environment in $SLURM_TMPDIR.
virtualenv --no-download ${SLURM_TMPDIR}/env
source ${SLURM_TMPDIR}/env/bin/activate

# Install Qiskit and its dependencies.
pip install --no-index --upgrade pip
pip install --no-index --requirement ~/qiskit_requirements.txt

# Modify your Qiskit program.
python qiskit_example.py
```

You can then [submit your job to the scheduler](link_to_scheduler_submission).


## Using Qiskit with MonarQ (in preparation)


## Use case: Bell states

Before you create a simulation of the first Bell state on Narval, the required modules need to be loaded.

```python
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
```

Define the circuit. Apply an Hadamard gate to create a superposition state on the first qubit and a CNOT gate to intricate the first and second qubits.

```python
circuit = QuantumCircuit(2,2)
circuit.h(0)
circuit.cx(0,1)
circuit.measure_all()
```

We want to use the default simulator, `AerSimulator` being the default simulator. We obtain the count of the final states of the qubits after 1000 measurements.

```python
simulator = AerSimulator()
result = simulator.run(circuit, shots=1000).result()
counts = result.get_counts()
print(counts)
# {'00': 489, '11': 535}
```

We display a histogram of the results with the command `plot_histogram(counts)`.

The results are displayed.

```python
plot_histogram(counts)
```

Results of 1000 measurements on the first Bell state.


**(Note:  Replace `link_to_available_wheels` and `link_to_scheduler_submission` with the actual links.)**
