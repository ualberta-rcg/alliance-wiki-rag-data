# PennyLane

PennyLane is an open-source software platform for differentiable quantum computing, first released on GitHub in 2018. Developed in Toronto by Xanadu, PennyLane allows you to design quantum circuits and run them on various quantum simulators and hardware. The platform is designed to facilitate the simulation, optimization, and learning of hybrid quantum algorithms that combine classical and quantum processing.


## Contents

* [Fonctionnalités](#fonctionnalités)
    * [Interface quantique unifiée](#interface-quantique-unifiée)
    * [Intégration avec des bibliothèques d'apprentissage automatique](#intégration-avec-des-bibliothèques-dapprentissage-automatique)
    * [Optimisation de circuits quantiques](#optimisation-de-circuits-quantiques)
    * [Outils de visualisation](#outils-de-visualisation)
    * [Communauté et développement](#communauté-et-développement)
* [Utiliser PennyLane avec MonarQ](#utiliser-pennylane-avec-monarq)
* [Création de l'environnement virtuel](#création-de-lenvironnement-virtuel)
* [Exécuter PennyLane sur une grappe](#exécuter-pennylane-sur-une-grappe)
* [Exemple d’utilisation : États de Bell](#exemple-dutilisation-états-de-bell)
* [Références](#références)


## Fonctionnalités

PennyLane offers several features to facilitate research and development in the field of differentiable quantum computing.

### Interface quantique unifiée

PennyLane provides a unified interface that allows you to design quantum circuits and run them on different simulators and quantum hardware. The platform supports several popular quantum simulators, such as Qiskit, CirQ, Strawberry Fields, and QuTip. PennyLane also supports several quantum hardware, including quantum devices from Xanadu, IBM, Rigetti, and IonQ.

Calcul Québec has developed the PennyLane-CalculQuebec plugin, which uses the PennyLane interface to design and run quantum circuits on MonarQ.

### Intégration avec des bibliothèques d'apprentissage automatique

PennyLane seamlessly integrates with popular machine learning libraries such as TensorFlow and PyTorch, allowing you to use machine learning tools to build hybrid quantum machine learning models and optimize quantum circuits.

### Optimisation de circuits quantiques

Using differentiable optimization techniques and combining classical and quantum differentiation methods, PennyLane optimizes the parameters of quantum circuits to solve various problems.

### Outils de visualisation

PennyLane provides visualization tools to facilitate understanding of how quantum circuits work.

### Communauté et développement

PennyLane is an open-source project with an active community of developers and users. The project is constantly updated with new features and improvements, and everyone can contribute to the development of the platform.


## Utiliser PennyLane avec MonarQ

MonarQ is designed to be programmed with Snowflurry, a software library programmed in Julia and developed by Anyon Systems. However, thanks to the PennyLane-CalculQuebec plugin, PennyLane circuits can be created using Snowflurry in the background. This allows you to run circuits on MonarQ while benefiting from the features and development environment offered by PennyLane. See the PennyLane-CalculQuebec documentation for installation and usage guides.

A quantum transpiler is also available from PennyLane to optimize its circuits for MonarQ.


## Création de l'environnement virtuel

Let's create a Python virtual environment to use PennyLane.

```bash
module load python/3.11
virtualenv --no-download --clear ~/ENV && source ~/ENV/bin/activate
pip install --no-index --upgrade pip
python -c "import pennylane"
```

You can also write the last three commands above in a file `pennylane-reqs.txt` and call the file within a session with the commands:

```bash
module load python/3.11
pip install --no-index -r pennylane-reqs.txt
```


## Exécuter PennyLane sur une grappe

File: `script.sh`

```bash
#!/bin/bash
#SBATCH --account=def-someuser # Indicate your account name
#SBATCH --time=00:15:00        # Modify if necessary
#SBATCH --cpus-per-task=1      # Modify if necessary
#SBATCH --mem-per-cpu=1G       # Modify if necessary

# Load module dependencies.
module load StdEnv/2023 gcc python/3.11

# Generate the virtual environment in $SLURM_TMPDIR.
virtualenv --no-download ${SLURM_TMPDIR}/env
source ${SLURM_TMPDIR}/env/bin/activate

# Install Pennylane and its dependencies.
pip install --no-index --upgrade pip
pip install --no-index --requirement ~/pennylane_requirements.txt

# Run your PennyLane program.
python pennylane_example.py
```

You can then submit your job to the scheduler.


## Exemple d’utilisation : États de Bell

Let's start by creating the virtual environment, as described above. We will then generate the first Bell state using PennyLane.

```python
import pennylane as qml

# Define the quantum circuit to generate the first Bell state
def bell_circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])

# Define the quantum circuit simulator
dev = qml.device('default.qubit', wires=2)

# Define the quantum circuit as a QNode function
@qml.qnode(dev)
def generate_bell_state():
    bell_circuit()
    return qml.state()

# Generate and display the first Bell state
bell_state_0 = generate_bell_state()
print("Premier état de Bell :", bell_state_0)
# Premier état de Bell : [0.70710678+0.j 0.        +0.j 0.        +0.j 0.70710678+0.j]
```


## Références

* [Site officiel de PennyLane](link_to_official_website)
* [Documentation de PennyLane sur GitHub](link_to_github_docs)
* [PennyLane-CalculQuebec](link_to_pennylane_calculquebec)


**(Note:  Please replace the bracketed links above with the actual URLs.)**
