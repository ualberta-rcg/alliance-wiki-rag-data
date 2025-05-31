# Dalton

This page is a translated version of the page [Dalton](https://docs.alliancecan.ca/mediawiki/index.php?title=Dalton&oldid=57927) and the translation is 100% complete.

Other languages: [English](https://docs.alliancecan.ca/mediawiki/index.php?title=Dalton&oldid=57927), [français](https://docs.alliancecan.ca/mediawiki/index.php?title=Dalton/fr&oldid=57928)


## Introduction

The core of the Dalton2016 software suite consists of two powerful applications for the study of the electronic structures of molecules: Dalton and LSDalton. Together, these applications offer extensive functionalities for calculating molecular properties at the HF, DFT, MCSCF, and CC theoretical levels. Several of its properties are unique to the Dalton2016 suite.

Project website: [http://daltonprogram.org/](http://daltonprogram.org/)

Documentation: [http://daltonprogram.org/documentation/](http://daltonprogram.org/documentation/)

Forum: [http://forum.daltonprogram.org/](http://forum.daltonprogram.org/)


## Modules

```bash
module load nixpkgs/16.09 intel/2016.4 openmpi/2.0.2 dalton/2017-alpha
```

Note that `dalton/2017-alpha` depends on an OpenMPI version other than the default version. For information on the `module` command, see [Using Modules](link_to_using_modules_page_if_available).


## Utilisation

Here is an example:

*   Input file: `dft_rspexci_nosym.dal` (see examples below)
*   Molecule specification: `H2O_cc-pVDZ_nosym.mol` (see examples below)

To use the atomic bases, add the option `-b ${BASLIB}` on the command line (see examples below).

To define the number of processes with a command-line option or an environment variable:

Add the option `-N ${SLURM_NTASKS}` on the command line for the launcher (see Script 1 in the examples below) or `export DALTON_NUM_MPI_PROCS=${SLURM_NTASKS}` (see Script 2 in the examples below).

To run Dalton, load the module and use the launcher `dalton`.

```bash
dalton -b ${BASLIB} -N ${SLURM_NTASKS} -dal dft_rspexci_nosym.dal -mol H2O_cc-pVDZ_nosym.mol
```

or

```bash
export DALTON_NUM_MPI_PROCS=${SLURM_NTASKS}
dalton -b ${BASLIB} -dal dft_rspexci_nosym.dal -mol H2O_cc-pVDZ_nosym.mol
```


## Exemples : scripts et fichiers d’entrée

### Exemple 1 : dft_rspexci_nosym

**INPUT**

**MOLECULE**

**Script 1**

**Script 2**

**File: `dft_rspexci_nosym.dal`**

```
**DALTON INPUT
.RUN RESPONSE
**INTEGRALS
.PROPRINT
**WAVE FUNCTIONS
.DFT
 B3LYP
**RESPONSE
*LINEAR
.SINGLE RESIDUE
.ROOTS
 3
**END OF DALTON INPUT
```

**File: `H2O_cc-pVDZ_nosym.mol`**

```
BASIS
cc-pVDZ
H2O

    2    0
        8.    1
O     0.0  0.0000000000 0.0
        1.    2
H1    1.430    0.0  1.1
H2   -1.430    0.0  1.1
```

**File: `run_dalton_job.sh`**

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=3500M
#SBATCH --time=00-30:00
# Load the module:
module load nixpkgs/16.09 intel/2016.4 openmpi/2.0.2 dalton/2017-alpha
# Setting the variables:
dltonlaun=dalton
dltonexec=dalton.x
daltoninput=dft_rspexci_nosym.dal
daltonmol=H2O_cc-pVDZ_nosym.mol
echo "Starting run at: `date`"
echo "Running the example: INPUT=${daltoninput} - Molecule=${daltonmol}"
${dltonlaun} -b ${BASLIB} -N ${SLURM_NTASKS} -dal ${daltoninput} -mol ${daltonmol}
echo "Program finished with exit code $? at: `date`"
```

**File: `run_dalton_job.sh`**

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem-per-cpu=3500M
#SBATCH --time=00-30:00
# Load the module:
module load nixpkgs/16.09 intel/2016.4 openmpi/2.0.2 dalton/2017-alpha
# Setting the variables:
dltonlaun=dalton
dltonexec=dalton.x
daltoninput=dft_rspexci_nosym.dal
daltonmol=H2O_cc-pVDZ_nosym.mol
# Set the number of cores DALTON_NUM_MPI_PROCS to ${SLURM_NTASKS}
export DALTON_NUM_MPI_PROCS=${SLURM_NTASKS}
echo "Starting run at: `date`"
echo "Running the example: INPUT=${daltoninput} - Molecule=${daltonmol}"
${dltonlaun} -b ${BASLIB} -dal ${daltoninput} -mol ${daltonmol}
echo "Program finished with exit code $? at: `date`"
```


### Exemple 2 : dft_rspexci_sym.dal

**INPUT**

**MOLECULE**

**Script 1**

**Script 2**

**File: `dft_rspexci_sym.dal`**

```
**DALTON INPUT
.RUN RESPONSE
**INTEGRALS
.PROPRINT
**WAVE FUNCTIONS
.DFT
 B3LYP
**RESPONSE
*LINEAR
.SINGLE RESIDUE
**END OF DALTON INPUT
```

**File: `H2O_cc-pVDZ_sym.mol`**

```
BASIS
cc-pVDZ
H2O

    2
        8.    1
O     0.0  0.0000000000 0.0
        1.    2
H1    1.430    0.0  1.1
H2   -1.430    0.0  1.1
```

**File: `run_dalton_job.sh`**

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=3500M
#SBATCH --time=00-30:00
# Load the module:
module load nixpkgs/16.09 intel/2016.4 openmpi/2.0.2 dalton/2017-alpha
# Setting the variables:
dltonlaun=dalton
dltonexec=dalton.x
daltoninput=dft_rspexci_sym.dal
daltonmol=H2O_cc-pVDZ_sym.mol
echo "Starting run at: `date`"
echo "Running the example: INPUT=${daltoninput} - Molecule=${daltonmol}"
${dltonlaun} -b ${BASLIB} -N ${SLURM_NTASKS} -dal ${daltoninput} -mol ${daltonmol}
echo "Program finished with exit code $? at: `date`"
```

**File: `run_dalton_job.sh`**

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=3500M
#SBATCH --time=00-30:00
# Load the module:
module load nixpkgs/16.09 intel/2016.4 openmpi/2.0.2 dalton/2017-alpha
# Setting the variables:
dltonlaun=dalton
dltonexec=dalton.x
daltoninput=dft_rspexci_sym.dal
daltonmol=H2O_cc-pVDZ_sym.mol
# Set the number of cores DALTON_NUM_MPI_PROCS to ${SLURM_NTASKS}
export DALTON_NUM_MPI_PROCS=${SLURM_NTASKS}
echo "Starting run at: `date`"
echo "Running the example: INPUT=${daltoninput} - Molecule=${daltonmol}"
${dltonlaun} -b ${BASLIB} -dal ${daltoninput} -mol ${daltonmol}
echo "Program finished with exit code $? at: `date`"
```

Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Dalton/fr&oldid=57928](https://docs.alliancecan.ca/mediawiki/index.php?title=Dalton/fr&oldid=57928)"
