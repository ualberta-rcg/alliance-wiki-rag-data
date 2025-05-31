# AMBER

## Introduction

Amber is the collective name for a suite of programs that allow users to perform molecular dynamics simulations, particularly on biomolecules. None of the individual programs carry this name, but the various parts work reasonably well together, and provide a powerful framework for many common calculations.

## Amber vs. AmberTools

We have modules for both Amber and AmberTools available in our [software stack](link-to-software-stack).

The **AmberTools** (module `ambertools`) contains a number of tools for preparing and analyzing simulations, as well as `sander` to perform molecular dynamics simulations, all of which are free and open source.

**Amber** (module `amber`) contains everything that is included in `ambertools`, but adds the advanced `pmemd` program for molecular dynamics simulations.

To see a list of installed versions and which other modules they depend on, you can use the `module spider` command or check the [Available software](link-to-available-software) page.

## Loading modules

| StdEnv     | AMBER version           | modules for running on CPUs                     | modules for running on GPUs (CUDA)                  | Notes                                         |
|------------|--------------------------|-------------------------------------------------|----------------------------------------------------|-------------------------------------------------|
| 2023       | amber/22.5-23.5          | StdEnv/2023 gcc/12.3 openmpi/4.1.5 amber/22.5-23.5 | StdEnv/2023 gcc/12.3 openmpi/4.1.5 cuda/12.2 amber/22.5-23.5 | GCC, FlexiBLAS & FFTW                         |
|            | ambertools/23.5         | StdEnv/2023 gcc/12.3 openmpi/4.1.5 ambertools/23.5 | StdEnv/2023 gcc/12.3 openmpi/4.1.5 cuda/12.2 ambertools/23.5 | GCC, FlexiBLAS & FFTW                         |
| 2020       | ambertools/21           | StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 scipy-stack ambertools/21 | StdEnv/2020  gcc/9.3.0 cuda/11.4 openmpi/4.0.3 scipy-stack ambertools/21 | GCC, FlexiBLAS & FFTW                         |
|            | amber/20.12-20.15       | StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 amber/20.12-20.15 | StdEnv/2020  gcc/9.3.0 cuda/11.4 openmpi/4.0.3 amber/20.12-20.15 | GCC, FlexiBLAS & FFTW                         |
|            | amber/20.9-20.15        | StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 amber/20.9-20.15  | StdEnv/2020  gcc/9.3.0 cuda/11.0 openmpi/4.0.3 amber/20.9-20.15  | GCC, MKL & FFTW                              |
|            | amber/18.14-18.17       | StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 amber/18.14-18.17 | StdEnv/2020  gcc/8.4.0  cuda/10.2  openmpi/4.0.3           | GCC, MKL                                     |
| 2016       | amber/18                 | StdEnv/2016 gcc/5.4.0 openmpi/2.1.1 scipy-stack/2019a amber/18 | StdEnv/2016 gcc/5.4.0 openmpi/2.1.1 cuda/9.0.176 scipy-stack/2019a amber/18 | GCC, MKL                                     |
|            | amber/18.10-18.11       | StdEnv/2016 gcc/5.4.0 openmpi/2.1.1 scipy-stack/2019a amber/18.10-18.11 | StdEnv/2016 gcc/5.4.0 openmpi/2.1.1 cuda/9.0.176 scipy-stack/2019a amber/18.10-18.11 | GCC, MKL                                     |
|            | amber/18.10-18.11       | StdEnv/2016 gcc/7.3.0 openmpi/3.1.2 scipy-stack/2019a amber/18.10-18.11 | StdEnv/2016 gcc/7.3.0  cuda/9.2.148 openmpi/3.1.2 scipy-stack/2019a amber/18.10-18.11 | GCC, MKL                                     |
|            | amber/16                 | StdEnv/2016.4 amber/16                         |                                                    | Available only on Graham. Some Python functionality is not supported |


## Using modules

### AmberTools 21

Currently, AmberTools 21 module is available on all clusters. AmberTools provide the following MD engines: sander, sander.LES, sander.LES.MPI, sander.MPI, sander.OMP, sander.quick.cuda, and sander.quick.cuda.MPI. After loading the module set AMBER environment variables:

```bash
source $EBROOTAMBERTOOLS/amber.sh
```

### Amber 20

There are two versions of amber/20 modules: 20.9-20.15 and 20.12-20.15. The first one uses MKL and cuda/11.0, while the second uses FlexiBLAS and cuda/11.4. MKL libraries do not perform well on AMD CPU, and FlexiBLAS solves this problem. It detects CPU type and uses libraries optimized for the hardware. cuda/11.4 is required for running simulations on A100 GPUs installed on Narval.

CPU-only modules provide all MD programs available in AmberTools/20 plus pmemd (serial) and pmemd.MPI (parallel). GPU modules add pmemd.cuda (single GPU), and pmemd.cuda.MPI (multi-GPU).

### Known issues

1. Module amber/20.12-20.15 does not have MMPBSA.py.MPI executable.
2. MMPBSA.py from amber/18-10-18.11 and amber/18.14-18.17 modules cannot perform PB calculations. Use more recent amber/20 modules for this type of calculations.

## Job submission examples

### Single GPU job

For GPU-accelerated simulations on Narval, use amber/20.12-20.15. Modules compiled with CUDA version < 11.4 do not work on A100 GPUs. Below is an example submission script for a single-GPU job with amber/20.12-20.15.

**File:** `pmemd_cuda_job.sh`

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

### CPU-only parallel MPI job

**Graham**

**File:** `pmemd_MPI_job_graham.sh`

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

**Cedar**

**File:** `pmemd_MPI_job_cedar.sh`

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

**Béluga**

**File:** `pmemd_MPI_job_beluga.sh`

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

**Narval**

**File:** `pmemd_MPI_job_narval.sh`

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

**Niagara**

**File:** `pmemd_MPI_job_narval.sh`

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

### QM/MM distributed multi-GPU job

The example below requests eight GPUs.

**File:** `quick_MPI_job.sh`

```bash
#!/bin/bash
#SBATCH --ntasks=8 --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4000
#SBATCH --time=02:00:00
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 cuda/12.2 ambertools/23.5

srun sander.quick.cuda.MPI -O -i input.in -p topol.parm7 -c coord.rst7 -o output.mdout -r restart.rst7
```

### Parallel MMPBSA job

The example below uses 32 MPI processes. MMPBSA scales linearly because each trajectory frame is processed independently.

**File:** `mmpbsa_job.sh`

```bash
#!/bin/bash
#SBATCH --ntasks=32
#SBATCH --mem-per-cpu=4000
#SBATCH --time=1:00:00
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 amber/22

srun MMPBSA.py.MPI -O -i mmpbsa.in -o mmpbsa.dat -sp solvated_complex.parm7 -cp complex.parm7 -rp receptor.parm7 -lp ligand.parm7 -y trajectory.nc
```

You can modify scripts to fit your simulation requirements for computing resources. See [Running jobs](link-to-running-jobs) for more details.

## Performance and benchmarking

A team at [ACENET](link-to-acenet) has created a [Molecular Dynamics Performance Guide](link-to-performance-guide) for Alliance clusters. It can help you determine optimal conditions for AMBER, GROMACS, NAMD, and OpenMM jobs. The present section focuses on AMBER performance.

View benchmarks of simulations with PMEMD [1]  View benchmarks of QM/MM simulations with SANDER.QUICK [2].


**(Remember to replace bracketed placeholders like `[link-to-software-stack]` with actual links.)**
