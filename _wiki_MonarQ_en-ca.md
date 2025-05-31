# MonarQ

Availability: January 2025

Connection Node: monarq.calculquebec.ca

MonarQ is a 24-qubit superconducting quantum computer developed in Montreal by Anyon Systems and located at the École de technologie supérieure. For more information on MonarQ's specifications and performance, see Technical Specifications below.


## Contents

1. Technical specifications
2. Accessing MonarQ
3. Technical Specifications
4. Quantum Computing Software
5. Getting Started on MonarQ
6. Frequently Asked Questions
7. Other Tools
8. Applications
9. Technical Support


## Technical Specifications

## Accessing MonarQ

To begin the process of accessing MonarQ, [fill out this form](link_to_form_needed).  It must be completed by the principal investigator.

You must have an account with the Alliance to access MonarQ.

Meet with our team to discuss the specifics of your project, access, and billing details.

Receive access to the MonarQ dashboard and generate your access token.

To get started, see Getting Started on MonarQ below.

Contact our quantum team at quantique@calculquebec.ca if you have any questions or would like a more general discussion before requesting access.


## Technical Specifications

**Qubit Mapping**

Like the quantum processors available today, MonarQ operates in an environment where noise remains a significant factor. Performance metrics, updated with each calibration, are accessible via the Thunderhead portal. Access to this portal requires MonarQ access approval.

The following metrics are available, among others:

*   24-qubit quantum processor
*   Single-qubit gate with 99.8% fidelity and 32ns duration
*   Two-qubit gate with 96% fidelity and 90ns duration
*   Coherence time of 4-10μs (depending on the state)
*   Maximum circuit depth of approximately 350 for single-qubit gates and 115 for two-qubit gates


## Quantum Computing Software

Several specialized software libraries exist for quantum computing and developing quantum algorithms. These libraries allow you to build circuits that are executed on simulators which mimic the performance and results obtained on a quantum computer such as MonarQ. They can be used on all Alliance clusters.

*   PennyLane, a Python command library
*   Snowflurry, a Julia command library
*   Qiskit, a Python command library

The quantum logic gates of the MonarQ processor are called via a software library, Snowflurry, written in Julia. Although MonarQ is natively compatible with Snowflurry, there is a PennyLane-CalculQuébec plugin developed by Calcul Québec that allows you to run circuits on MonarQ while benefiting from the features and development environment offered by PennyLane.


## Getting Started on MonarQ

**Prerequisites:** Ensure you have access to MonarQ and your login credentials (username, API token). For any questions, write to quantique@calculquebec.ca.

**Step 1: Connect to Narval**

MonarQ is only accessible from Narval, a Calcul Québec cluster. Access to Narval is via the connection node narval.alliancecan.ca.

For help connecting to Narval, see the SSH page.

**Step 2: Create the Environment**

Create a Python virtual environment (3.11 or later) to use PennyLane and the PennyLane-CalculQuébec plugin. These are already installed on Narval, and you will only need to import the software libraries you want.

```bash
[name@server ~] $ module load python/3.11
[name@server ~] $ virtualenv --no-download --clear ~/ENV && source ~/ENV/bin/activate
[name@server ~] $ pip install --no-index --upgrade pip
[name@server ~] $ pip install --no-index --upgrade pennylane-calculquebec
[name@server ~] $ python -c "import pennylane; import pennylane_calculquebec"
```

**Step 3: Configure your credentials on MonarQ and define MonarQ as a device**

Open a Python .py file and import the necessary dependencies, such as PennyLane and MonarqClient in the example below.

Create a client with your credentials. Your token is available from the Thunderhead portal. The host is monarq.calculquebec.ca.

Create a PennyLane device with your client. You can also specify the number of qubits (wires) to use and the number of samples (shots).

For help, see pennylane_calculquebec.

**File: my_circuit.py**

```python
import pennylane as qml
from pennylane_calculquebec.API.client import MonarqClient

my_client = MonarqClient("monarq.calculquebec.ca", "your username", "your access token", "your project")
dev = qml.device("monarq.default", client=my_client, wires=3, shots=1000)
```

**Step 4: Create your circuit**

In the same Python file, you can now code your quantum circuit.

**File: my_circuit.py**

```python
@qml.qnode(dev)
def bell_circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    return qml.counts()

result = bell_circuit()
print(result)
```

**Step 5: Run your circuit from the scheduler**

The `sbatch` command is used to submit a task.

```bash
$ sbatch simple_job.sh
Submitted batch job 123456
```

With a Slurm script like this:

**File: simple_job.sh**

```bash
#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --account=def-someuser # Your username
#SBATCH --cpus-per-task=1      # Modify if necessary
#SBATCH --mem-per-cpu=1G       # Modify if necessary
python my_circuit.py
```

The result of the circuit is written to a file whose name begins with slurm-, followed by the task ID and the suffix .out, for example slurm-123456.out.

This file contains the result of our circuit in a dictionary `{'000': 496, '001': 0, '010': 0, '011': 0, '100': 0, '101': 0, '110': 0, '111': 504}`.

For more information on how to submit jobs on Narval, see Running Jobs.


## Frequently Asked Questions

(FAQ)


## Other Tools

Quantum Transpiler


## Applications

MonarQ is suited to computations requiring small amounts of high-fidelity qubits, making it an ideal tool for the development and testing of quantum algorithms. Other possible applications include modeling small quantum systems; testing new quantum programming and error correction methods and techniques; and more generally, fundamental research in quantum computing.


## Technical Support

If you have any questions about our quantum services, write to quantique@calculquebec.ca.

Sessions on quantum computing and programming with MonarQ are [listed here](link_to_sessions_needed).
