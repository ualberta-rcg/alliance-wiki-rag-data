# MonarQ

This page is a translated version of the page MonarQ and the translation is 100% complete.

Other languages: Canadian English, English, français

Availability: January 2025

Login node: monarq.calculquebec.ca

MonarQ is a 24-qubit superconducting quantum computer developed in Montreal by Anyon Systems and located at the École de technologie supérieure. See section Technical specifications below.

Its name is inspired by the monarch butterfly, a symbol of evolution and migration. The capital Q denotes the quantum nature of the computer and its origins in Quebec. Acquisition of MonarQ was made possible with the support of the Ministère de l'Économie, de l'Innovation et de l'Énergie du Québec (MEIE) and Canada Economic Development (CED).


## Getting access to MonarQ

To begin the process of getting access to MonarQ, [complete this form](link_to_form). It can only be completed by the principal investigator.

You must have an [account with the Alliance](link_to_alliance_account) in order to get access to MonarQ.

1. Meet with our team to discuss the specifics of your project.
2. Receive access to the MonarQ dashboard and generate your access token.
3. To get started using MonarQ, see Getting started below.

Contact our quantum team at quantique@calculquebec.ca if you have any questions or if you want to have a more general discussion before requesting access to MonarQ.


## Technical specifications

MonarQ qubit mapping

Like quantum processors available today, MonarQ operates in an environment where noise remains a significant factor. Performance metrics, updated at each calibration, are accessible via the Thunderhead portal which you will be able to use after being approved for access to MonarQ.

Among the metrics are:

*   24-qubit quantum processor
*   Single-qubit gate: 99.8% fidelity with gate duration of 15ns
*   Two-qubit gate: 95.6% fidelity with gate duration of 35ns
*   Coherence time: 4-10μs (depending on state)
*   Maximum circuit depth: approximately 350 for single-qubit gates and 115 for two-qubit gates


## Quantum computing software

There are several specialized software libraries for quantum computing and the development of quantum algorithms. These libraries allow you to build circuits that are executed on simulators that mimic the performance and results obtained on a quantum computer such as MonarQ. They can be used on all Alliance clusters.

*   PennyLane, for Python commands
*   Snowflurry, for Julia commands
*   Qiskit, for Python commands

The quantum logic gates of the MonarQ processor are called through a Snowflurry software library written in Julia. Although MonarQ is natively compatible with Snowflurry, there is a PennyLane-Snowflurry plugin developed by Calcul Québec that allows you to execute circuits on MonarQ while benefiting from the features and development environment offered by PennyLane.


## Getting started

### Prerequisites

Make sure you have access to MonarQ and that you have your login credentials (username, API token). If you have any questions, write to quantique@calculquebec.ca.

### Step 1: Connect to Narval

MonarQ is only accessible from Narval, a Calcul Québec cluster. Narval is accessed from the login node narval.alliancecan.ca.

For help connecting to Narval, see SSH (link_to_ssh_instructions).

### Step 2: Create the environment

Create a Python virtual environment (3.11 or later) to use PennyLane and the PennyLane-CalculQuébec plugin. These are already installed on Narval so that you will only have to import the software libraries you want.

```bash
[name@server ~]$ module load python/3.11
[name@server ~]$ virtualenv --no-download --clear ~/ENV && source ~/ENV/bin/activate
[name@server ~]$ pip install --no-index --upgrade pip
[name@server ~]$ pip install --no-index --upgrade pennylane-calculquebec
[name@server ~]$ python -c "import pennylane; import pennylane_calculquebec"
```

### Step 3: Configure your identifiers on MonarQ and define MonarQ as your device

Open a Python .py file and import the required dependencies (in the following example, PennyLane and MonarqClient). Create a client with your identifiers. Your token is available through the Thunderhead portal. The host is monarq.calculquebec.ca. Create a PennyLane device with your client. You can also enter the number of qubits (`wires`) and the number of shots. For more information, see pennylane_calculquebec (link_to_pennylane_calculquebec).

**File: my_circuit.py**

```python
import pennylane as qml
from pennylane_calculquebec.API.client import MonarqClient

my_client = MonarqClient("monarq.calculquebec.ca", "your username", "your access token", "your project")
dev = qml.device("monarq.default", client=my_client, wires=3, shots=1000)
```

### Step 4: Create your circuit

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

### Step 5: Execute your circuit from the scheduler

The `sbatch` command is used to submit a task.

```bash
$ sbatch simple_job.sh
Submitted batch job 123456
```

The Slurm script is similar to:

**File: simple_job.sh**

```bash
#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --account=def-someuser # Votre username
#SBATCH --cpus-per-task=1      # Modifiez s'il y a lieu
#SBATCH --mem-per-cpu=1G       # Modifiez s'il y a lieu
python my_circuit.py
```

The result is written to a file with a name starting with `slurm-`, followed by the task ID and the `.out` suffix, for example `slurm-123456.out`. The file contains the result in dictionary `{'000': 496, '001': 0, '010': 0, '011': 0, '100': 0, '101': 0, '110': 0, '111': 504}`.

For more information on submitting tasks on Narval, see Running jobs (link_to_running_jobs).


## FAQ

Foire aux questions (FAQ) (link_to_faq)


## Other tools

Quantum transpilation (link_to_quantum_transpilation)


## Applications

MonarQ is suited for computations requiring small quantities of high-fidelity qubits, making it an ideal tool to develop and test quantum algorithms. Other possible applications include modelling small quantum systems; testing new methods and techniques for quantum programming and error correction; and more generally, fundamental research in quantum computing.


## Technical support

For questions about our quantum services, write to quantique@calculquebec.ca.

Sessions on quantum computing and programming with MonarQ are [listed here](link_to_sessions).


**(Remember to replace the bracketed placeholders like `link_to_form` with actual links.)**
