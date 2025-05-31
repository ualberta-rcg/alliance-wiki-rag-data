# AMBER

This page is a translated version of the page AMBER and the translation is 100% complete.

Other languages: [English](link-to-english-page), [français](current-page)


## Introduction

Amber designates a suite of applications for performing molecular dynamics simulations, particularly with biomolecules. Each application has a different name, but the suite works rather well and constitutes a powerful tool for performing several usual calculations.


## AmberTools et Amber

The modules for AmberTools and Amber are available on our clusters.

The `ambertools` module for AmberTools offers tools to prepare and analyze simulations. The `sander` application is used for molecular dynamics simulations. All these tools are free and open source.

The `amber` module for Amber contains everything offered by `ambertools`, but adds `pmemd`, a more advanced application for molecular dynamics simulations.

For the list of installed versions and their dependent modules, run the `module spider` command or consult the [Software Available](link-to-software-page) page.


## Charger des modules

The following tables show available versions with CPU and GPU (CUDA) support, along with notes on dependencies:

**Table 1: Amber & AmberTools Versions (StdEnv/2023)**

| Version             | avec CPU                                      | avec GPU (CUDA)                                   | Notes                                           |
|----------------------|-------------------------------------------------|----------------------------------------------------|-------------------------------------------------|
| amber/22.5-23.5      | StdEnv/2023 gcc/12.3 openmpi/4.1.5 amber/22.5-23.5 | StdEnv/2023 gcc/12.3 openmpi/4.1.5 cuda/12.2 amber/22.5-23.5 | GCC, FlexiBLAS & FFTW                         |
| ambertools/23.5     | StdEnv/2023 gcc/12.3 openmpi/4.1.5 ambertools/23.5 | StdEnv/2023 gcc/12.3 openmpi/4.1.5 cuda/12.2 ambertools/23.5 | GCC, FlexiBLAS & FFTW                         |


**Table 2: Amber & AmberTools Versions (StdEnv/2020)**

| Version             | avec CPU                                      | avec GPU (CUDA)                                   | Notes                                           |
|----------------------|-------------------------------------------------|----------------------------------------------------|-------------------------------------------------|
| ambertools/21       | StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 scipy-stack ambertools/21 | StdEnv/2020 gcc/9.3.0 cuda/11.4 openmpi/4.0.3 scipy-stack ambertools/21 | GCC, FlexiBLAS & FFTW                         |
| amber/20.12-20.15   | StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 amber/20.12-20.15   | StdEnv/2020 gcc/9.3.0 cuda/11.4 openmpi/4.0.3 amber/20.12-20.15   | GCC, FlexiBLAS & FFTW                         |
| amber/20.9-20.15    | StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 amber/20.9-20.15    | StdEnv/2020 gcc/9.3.0 cuda/11.0 openmpi/4.0.3 amber/20.9-20.15    | GCC, MKL & FFTW                              |
| amber/18.14-18.17   | StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 amber/18.14-18.17   | StdEnv/2020 gcc/8.4.0 cuda/10.2 openmpi/4.0.3     | GCC, MKL                                      |


**Table 3: Amber & AmberTools Versions (StdEnv/2016)**

| Version             | avec CPU                                      | avec GPU (CUDA)                                   | Notes                                           |
|----------------------|-------------------------------------------------|----------------------------------------------------|-------------------------------------------------|
| amber/18            | StdEnv/2016 gcc/5.4.0 openmpi/2.1.1 scipy-stack/2019a amber/18 | StdEnv/2016 gcc/5.4.0 openmpi/2.1.1 cuda/9.0.176 scipy-stack/2019a amber/18 | GCC, MKL                                      |
| amber/18.10-18.11   | StdEnv/2016 gcc/5.4.0 openmpi/2.1.1 scipy-stack/2019a amber/18.10-18.11   | StdEnv/2016 gcc/5.4.0 openmpi/2.1.1 cuda/9.0.176 scipy-stack/2019a amber/18.10-18.11   | GCC, MKL                                      |
| amber/18.10-18.11   | StdEnv/2016 gcc/7.3.0 openmpi/3.1.2 scipy-stack/2019a amber/18.10-18.11   | StdEnv/2016 gcc/7.3.0 cuda/9.2.148 openmpi/3.1.2 scipy-stack/2019a amber/18.10-18.11   | GCC, MKL                                      |
| amber/16            | StdEnv/2016.4 amber/16                         |                                                    | Available only on Graham. Some Python features are not supported. |


## Utilisation

### AmberTools 21

The AmberTools 21 module is currently available on all clusters and offers `sander`, `sander.LES`, `sander.LES.MPI`, `sander.MPI`, `sander.OMP`, `sander.quick.cuda`, and `sander.quick.cuda.MPI`. After loading the module, configure the environment variables with `source $EBROOTAMBERTOOLS/amber.sh`

### Amber 20

Amber20 is currently available on all clusters. There are two modules: 20.9-20.15 and 20.12-20.15.

20.9-20.15 uses MKL and cuda/11.0; note that MKL libraries do not work well with AMD and CPUs.

20.12-20.15 uses FlexiBLAS and cuda/11.4; FlexiBLAS detects the CPU type and uses libraries optimized for the hardware.  Additionally, CUDA/11.4 is required to perform simulations on A100 GPUs (installed on Narval).

The modules for use with CPU offer the applications available with AmberTools/20 plus `pmemd` (sequential) and `pmemd.MPI` (parallel). The modules for use with GPU add `pmemd.cuda` (single GPU) and `pmemd.cuda.MPI` (multiple GPUs).


### Known issues

1. The `amber/20.12-20.15` module does not offer the `MMPBSA.py.MPI` executable.
2. `MMPBSA.py` from modules `amber/18-10-18.11` and `amber/18.14-18.17` cannot perform PB calculations; use the newer `amber/20` modules instead.


## Exemples de soumission de tâches

### Avec un seul GPU

For simulations with a GPU on Narval, use `amber/20.12-20.15`. Modules compiled with a CUDA version < 11.4 do not work on an A100 GPU.

**File: `pmemd_cuda_job.sh`**

```bash
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=2000
#SBATCH --time=10:00:00
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 cuda/12.2 amber/22

pmemd.cuda -O -i input.in -p topol.parm7 -c coord.rst7 -o output.mdout -r restart.rst7
```

### Tâche MPI parallèle avec CPU

**File: `pmemd_MPI_job_graham.sh`**

```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32
#SBATCH --mem-per-cpu=2000
#SBATCH --time=1:00:00
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 amber/22

srun pmemd.MPI -O -i input.in -p topol.parm7 -c coord.rst7 -o output.mdout -r restart.rst7
```

**File: `pmemd_MPI_job_cedar.sh`**

```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=48
#SBATCH --mem-per-cpu=2000
#SBATCH --time=1:00:00
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 amber/22

srun pmemd.MPI -O -i input.in -p topol.parm7 -c coord.rst7 -o output.mdout -r restart.rst7
```

**File: `pmemd_MPI_job_beluga.sh`**

```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=40
#SBATCH --mem-per-cpu=2000
#SBATCH --time=1:00:00
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 amber/22

srun pmemd.MPI -O -i input.in -p topol.parm7 -c coord.rst7 -o output.mdout -r restart.rst7
```

**File: `pmemd_MPI_job_narval.sh` (first instance)**

```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=64
#SBATCH --mem-per-cpu=2000
#SBATCH --time=1:00:00
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 amber/22

srun pmemd.MPI -O -i input.in -p topol.parm7 -c coord.rst7 -o output.mdout -r restart.rst7
```

**File: `pmemd_MPI_job_narval.sh` (second instance)**

```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=40
#SBATCH --mem-per-cpu=2000
#SBATCH --time=1:00:00
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 amber/22

srun pmemd.MPI -O -i input.in -p topol.parm7 -c coord.rst7 -o output.mdout -r restart.rst7
```

(Note:  There's a duplicate `pmemd_MPI_job_narval.sh` in the source.  Both versions are included here.)


### Tâche QM/MM distribuée avec plusieurs GPU

In the following example, eight GPUs are requested.

**File: `quick_MPI_job.sh`**

```bash
#!/bin/bash
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4000
#SBATCH --time=2:00:00
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 cuda/12.2 ambertools/23.5

srun sander.quick.cuda.MPI -O -i input.in -p topol.parm7 -c coord.rst7 -o output.mdout -r restart.rst7
```


### Tâche MMPBSA parallèle

In the following example, 32 MPI processes are used. The scalability of MMPBSA is linear because each trajectory frame is processed independently.

**File: `mmpbsa_job.sh`**

```bash
#!/bin/bash
#SBATCH --ntasks=32
#SBATCH --mem-per-cpu=4000
#SBATCH --time=1:00:00
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 amber/22

srun MMPBSA.py.MPI -O -i mmpbsa.in -o mmpbsa.dat -sp solvated_complex.parm7 -cp complex.parm7 -rp receptor.parm7 -lp ligand.parm7 -y trajectory.nc
```

The scripts can be modified according to the computational resource needs of your tasks (see [Running Jobs](link-to-running-jobs-page)).


## Performance et étalonnage benchmarking

The [Molecular Dynamics Performance Guide](link-to-performance-guide) was created by an ACENET team. The guide describes the optimal conditions for running tasks on our clusters with GROMACS, NAMD, and OpenMM.

Calibration of simulations with PMEMD [1]
Calibration of QM/MM simulations with SANDER.QUICK [2].


**(Note:  Bracketed numbers [1] and [2] likely refer to citations or footnotes that were not included in the original HTML.)**
