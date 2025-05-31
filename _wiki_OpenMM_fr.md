# OpenMM

OpenMM<sup>[1]</sup> is a toolkit designed for molecular simulation. It can be used as a standalone application to perform simulations or as a library that you call from your code. OpenMM is a unique package due to its great flexibility in custom force fields and resolution (or integration) algorithms, its openness, and its excellent performance, especially with recent GPUs.


## Simulation with AMBER Topologies and Restart Files

### Preparing the Python Virtual Environment

This example uses the openmm/7.7.0 module.

1. Create and activate the Python virtual environment.

```bash
[name@server ~] module load python
[name@server ~] virtualenv $HOME/env-parmed
[name@server ~] source $HOME/env-parmed/bin/activate
```

2. Install the ParmEd and netCDF4 Python modules.

```bash
(env-parmed)[name@server ~] pip install --no-index parmed==3.4.3 netCDF4
```

### Submitting a Task

The following script is for a simulation task that uses a GPU.

**File:** `submit_openmm.cuda.sh`

```bash
#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=0-01:00:00
# Usage: sbatch $0
module purge
module load StdEnv/2020 gcc/9.3.0 cuda/11.4 openmpi/4.0.3
module load python/3.8.10 openmm/7.7.0 netcdf/4.7.4 hdf5/1.10.6 mpi4py/3.0.3
source $HOME/env-parmed/bin/activate

python openmm_input.py
```

Here, `openmm_input.py` is a Python script that loads Amber files, creates the OpenMM simulation system, configures the integration, and runs the dynamics (see this example).


### Performance and Benchmarking

The *Molecular Dynamics Performance Guide* was created by an ACENET team.  The guide describes the optimal conditions for running tasks on our clusters with AMBER, GROMACS, and NAMD.

On the CUDA platform, OpenMM only needs one CPU per GPU because the CPUs are not used for calculations. OpenMM can use multiple GPUs in a node, but it is more efficient to perform simulations with a single GPU. As demonstrated by the tests on Narval and those on Cedar, the simulation speed with multiple GPUs is slightly increased on nodes with NvLink where the GPUs are directly connected. Without NvLink, the simulation speed increases very little with P100 GPUs (tests on Cedar).


[1]: https://openmm.org/

Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=OpenMM/fr&oldid=162828](https://docs.alliancecan.ca/mediawiki/index.php?title=OpenMM/fr&oldid=162828)"
