# GROMACS

Other languages: English français

## General

GROMACS is a versatile package to perform molecular dynamics for systems with hundreds to millions of particles. It is primarily designed for biochemical molecules like proteins, lipids, and nucleic acids that have a lot of complicated bonded interactions, but since GROMACS is extremely fast at calculating the nonbonded interactions (that usually dominate simulations), many groups are also using it for research on non-biological systems, e.g., polymers.

### Strengths

*   GROMACS provides extremely high performance compared to all other programs.
*   Since GROMACS 4.6, we have excellent CUDA-based GPU acceleration on GPUs that have Nvidia compute capability >= 2.0 (e.g., Fermi or later).
*   GROMACS comes with a large selection of flexible tools for trajectory analysis.
*   GROMACS can be run in parallel, using either the standard MPI communication protocol, or via our own "Thread MPI" library for single-node workstations.
*   GROMACS is free software, available under the GNU Lesser General Public License (LGPL), version 2.1.

### Weak points

*   To get very high simulation speed, GROMACS does not do much additional analysis and/or data collection on the fly. It may be a challenge to obtain somewhat non-standard information about the simulated system from a GROMACS simulation.
*   Different versions may have significant differences in simulation methods and default parameters. Reproducing results of older versions with a newer version may not be straightforward.
*   Additional tools and utilities that come with GROMACS are not always of the highest quality, may contain bugs, and may implement poorly documented methods. Reconfirming the results of such tools with independent methods is always a good idea.

### GPU support

The top part of any log file will describe the configuration, and in particular whether your version has GPU support compiled in. GROMACS will automatically use any GPUs it finds. GROMACS uses both CPUs and GPUs; it relies on a reasonable balance between CPU and GPU performance. The new neighbour structure required the introduction of a new variable called "cutoff-scheme" in the mdp file. The behaviour of older GROMACS versions (before 4.6) corresponds to `cutoff-scheme = group`, while in order to use GPU acceleration you must change it to `cutoff-scheme = verlet`, which has become the new default in version 5.0.


## Quickstart guide

This section summarizes configuration details.

### Environment modules

The following versions have been installed:

StdEnv/2023, StdEnv/2020, StdEnv/2018.3, StdEnv/2016.4

| GROMACS version | modules for running on CPUs                      | modules for running on GPUs (CUDA)                   | Notes                                                                     |
|-----------------|----------------------------------------------------|------------------------------------------------------|-----------------------------------------------------------------------------|
| gromacs/2024.4   | StdEnv/2023 gcc/12.3 openmpi/4.1.5 gromacs/2024.4   | StdEnv/2023 gcc/12.3 openmpi/4.1.5 cuda/12.2 gromacs/2024.4 | GCC, FlexiBLAS & FFTW                                                    |
| gromacs/2024.1   | StdEnv/2023 gcc/12.3 openmpi/4.1.5 gromacs/2024.1   | StdEnv/2023 gcc/12.3 openmpi/4.1.5 cuda/12.2 gromacs/2024.1 | GCC, FlexiBLAS & FFTW                                                    |
| gromacs/2023.5   | StdEnv/2023 gcc/12.3 openmpi/4.1.5 gromacs/2023.5   | StdEnv/2023 gcc/12.3 openmpi/4.1.5 cuda/12.2 gromacs/2023.5 | GCC, FlexiBLAS & FFTW                                                    |
| gromacs/2023.3   | StdEnv/2023 gcc/12.3 openmpi/4.1.5 gromacs/2023.3   | StdEnv/2023 gcc/12.3 openmpi/4.1.5 cuda/12.2 gromacs/2023.3 | GCC, FlexiBLAS & FFTW                                                    |
| gromacs/2023.2   | StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 gromacs/2023.2   | StdEnv/2020 gcc/9.3.0 cuda/11.4 openmpi/4.0.3 gromacs/2023.2 | GCC, FlexiBLAS & FFTW                                                    |
| gromacs/2023     | StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 gromacs/2023     | StdEnv/2020 gcc/9.3.0 cuda/11.4 openmpi/4.0.3 gromacs/2023     | GCC, FlexiBLAS & FFTW                                                    |
| gromacs/2022.3   | StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 gromacs/2022.3   | StdEnv/2020 gcc/9.3.0 cuda/11.4 openmpi/4.0.3 gromacs/2022.3 | GCC, FlexiBLAS & FFTW                                                    |
| gromacs/2022.2   | StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 gromacs/2022.2   |                                                      | GCC, FlexiBLAS & FFTW                                                    |
| gromacs/2021.6   | StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 gromacs/2021.6   | StdEnv/2020 gcc/9.3.0 cuda/11.4 openmpi/4.0.3 gromacs/2021.6 | GCC, FlexiBLAS & FFTW                                                    |
| gromacs/2021.4   | StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 gromacs/2021.4   | StdEnv/2020 gcc/9.3.0 cuda/11.4 openmpi/4.0.3 gromacs/2021.4 | GCC, FlexiBLAS & FFTW                                                    |
| gromacs/2021.2   | StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 gromacs/2021.2   | StdEnv/2020 gcc/9.3.0 cuda/11.0 openmpi/4.0.3 gromacs/2021.2 | GCC & MKL                                                              |
| gromacs/2020.6   | StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 gromacs/2020.6   | StdEnv/2020 gcc/9.3.0 cuda/11.4 openmpi/4.0.3 gromacs/2020.6 | GCC, FlexiBLAS & FFTW                                                    |
| gromacs/2020.4   | StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 gromacs/2020.4   | StdEnv/2020 gcc/9.3.0 cuda/11.0 openmpi/4.0.3 gromacs/2020.4 | GCC & MKL                                                              |
| gromacs/2020.2   | StdEnv/2018.3 gcc/7.3.0 openmpi/3.1.2 gromacs/2020.2   | StdEnv/2018.3 gcc/7.3.0 cuda/10.0.130 openmpi/3.1.2 gromacs/2020.2 | GCC & MKL  (Deprecated)                                                  |
| gromacs/2019.6   | StdEnv/2018.3 gcc/7.3.0 openmpi/3.1.2 gromacs/2019.6   | StdEnv/2018.3 gcc/7.3.0 cuda/10.0.130 openmpi/3.1.2 gromacs/2019.6 | GCC & MKL  (Deprecated)                                                  |
| gromacs/2019.3   | StdEnv/2018.3 gcc/7.3.0 openmpi/3.1.2 gromacs/2019.3   | StdEnv/2018.3 gcc/7.3.0 cuda/10.0.130 openmpi/3.1.2 gromacs/2019.3 | GCC & MKL (Deprecated)                                                   |
| gromacs/2018.7   | StdEnv/2018.3 gcc/7.3.0 openmpi/3.1.2 gromacs/2018.7   | StdEnv/2018.3 gcc/7.3.0 cuda/10.0.130 openmpi/3.1.2 gromacs/2018.7 | GCC & MKL (Deprecated)                                                  |
| gromacs/2018.3   | StdEnv/2016.4 gcc/6.4.0 openmpi/2.1.1 gromacs/2018.3   | StdEnv/2016.4 gcc/6.4.0 cuda/9.0.176 openmpi/2.1.1 gromacs/2018.3 | GCC & FFTW (Deprecated)                                                  |
| gromacs/2018.2   | StdEnv/2016.4 gcc/6.4.0 openmpi/2.1.1 gromacs/2018.2   | StdEnv/2016.4 gcc/6.4.0 cuda/9.0.176 openmpi/2.1.1 gromacs/2018.2 | GCC & FFTW (Deprecated)                                                  |
| gromacs/2018.1   | StdEnv/2016.4 gcc/6.4.0 openmpi/2.1.1 gromacs/2018.1   | StdEnv/2016.4 gcc/6.4.0 cuda/9.0.176 openmpi/2.1.1 gromacs/2018.1 | GCC & FFTW (Deprecated)                                                  |
| gromacs/2018     | StdEnv/2016.4 gromacs/2018                           | StdEnv/2016.4 cuda/9.0.176 gromacs/2018                           | Intel & MKL (Deprecated)                                                  |
| gromacs/2016.5   | StdEnv/2016.4 gcc/6.4.0 openmpi/2.1.1 gromacs/2016.5   | StdEnv/2016.4 gcc/6.4.0 cuda/9.0.176 openmpi/2.1.1 gromacs/2016.5 | GCC & FFTW (Deprecated)                                                  |
| gromacs/2016.3   | StdEnv/2016.4 gromacs/2016.3                           | StdEnv/2016.4 cuda/8.0.44 gromacs/2016.3                           | Intel & MKL (Deprecated)                                                  |
| gromacs/5.1.5    | StdEnv/2016.4 gromacs/5.1.5                            | StdEnv/2016.4 cuda/8.0.44 gromacs/5.1.5                            | Intel & MKL (Deprecated)                                                  |
| gromacs/5.1.4    | StdEnv/2016.4 gromacs/5.1.4                            | StdEnv/2016.4 cuda/8.0.44 gromacs/5.1.4                            | Intel & MKL (Deprecated)                                                  |
| gromacs/5.0.7    | StdEnv/2016.4 gromacs/5.0.7                            | StdEnv/2016.4 cuda/8.0.44 gromacs/5.0.7                            | Intel & MKL (Deprecated)                                                  |
| gromacs/4.6.7    | StdEnv/2016.4 gromacs/4.6.7                            | StdEnv/2016.4 cuda/8.0.44 gromacs/4.6.7                            | Intel & MKL (Deprecated)                                                  |
| gromacs/4.6.7    | StdEnv/2016.4 gcc/5.4.0 openmpi/2.1.1 gromacs/4.6.7    | StdEnv/2016.4 gcc/5.4.0 cuda/8.0 openmpi/2.1.1 gromacs/4.6.7    | GCC & MKL & ThreadMPI (Deprecated)                                         |


**Notes:**

*   GROMACS versions 2020.0 up to and including 2021.5 contain a bug when used on GPUs of Volta or newer generations (i.e., V100, T4, and A100) with `mdrun` option `-update gpu` that could have perturbed the virial calculation and, in turn, led to incorrect pressure coupling.  The GROMACS developers state in the 2021.6 Release Notes: [1] The GPU update is not enabled by default, so the error can only appear in simulations where it was manually selected, and even in this case the error might be rare since we have not observed it in practice in the testing we have performed. Further discussion of this bug can be found in the GitLab issue #4393 of the GROMACS project. [2]
*   Version 2020.4 and newer have been compiled for the new Standard software environment StdEnv/2020.
*   Version 2018.7 and newer have been compiled with GCC compilers and the MKL-library, as they run a bit faster.
*   Older versions have been compiled with either GCC compilers and FFTW or Intel compilers, using Intel MKL and Open MPI 2.1.1 libraries from the default environment as indicated in the table above.
*   CPU (non-GPU) versions are available in both single- and double precision, with the exception of 2019.3 (‡), where double precision is not available for AVX512.

These modules can be loaded using a `module load` command with the modules as stated in the second column in the above table. For example:

```bash
$ module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 gromacs/2024.4
```

or

```bash
$ module load StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 gromacs/2023.2
```

These versions are also available with GPU support, albeit only with single precision. In order to load the GPU enabled version, the `cuda` module needs to be loaded first. The modules needed are listed in the third column of the above table, e.g.:

```bash
$ module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 cuda/12.2 gromacs/2024.4
```

or

```bash
$ module load StdEnv/2020 gcc/9.3.0 cuda/11.4 openmpi/4.0.3 gromacs/2023.2
```

For more information on environment modules, please refer to the [Using modules](link-to-using-modules-page) page.


### Suffixes

#### GROMACS 5.x, 2016.x and newer

GROMACS 5 and newer releases consist of only four binaries that contain the full functionality. All GROMACS tools from previous versions have been implemented as sub-commands of the `gmx` binaries. Please refer to [GROMACS 5.0 Tool Changes](link-to-5.0-tool-changes) and the [GROMACS documentation manuals](link-to-gromacs-manuals) for your version.

*   `gmx` - mixed ("single") precision GROMACS with OpenMP (threading) but without MPI.
*   `gmx_mpi` - mixed ("single") precision GROMACS with OpenMP and MPI.
*   `gmx_d` - double precision GROMACS with OpenMP but without MPI.
*   `gmx_mpi_d` - double precision GROMACS with OpenMP and MPI.

#### GROMACS 4.6.7

The double precision binaries have the suffix `_d`. The parallel single and double precision `mdrun` binaries are:

*   `mdrun_mpi`
*   `mdrun_mpi_d`

### Submission scripts

Please refer to the [Running jobs](link-to-running-jobs-page) page for help on using the SLURM workload manager.

#### Serial jobs

Here's a simple job script for serial mdrun:

**File:** `serial_gromacs_job.sh`

```bash
#!/bin/bash
#SBATCH --time=0-0:30         # time limit (D-HH:MM)
#SBATCH --mem-per-cpu=1000M   # memory per CPU (in MB)
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 gromacs/2024.4
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
gmx mdrun -nt 1 -deffnm em
```

This will run the simulation of the molecular system in the file `em.tpr`.

#### Whole nodes

Commonly the systems which are being simulated with GROMACS are so large that you want to use a number of whole nodes for the simulation. Generally, the product of `--ntasks-per-node=` and `--cpus-per-task` has to match the number of CPU cores in the compute nodes of the cluster. Please see section [Performance and benchmarking](#performance-and-benchmarking) below.

**File:** `gromacs_whole_node_graham.sh`

```bash
#!/bin/bash
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=16     # request 16 MPI tasks per node
#SBATCH --cpus-per-task=2        # 2 OpenMP threads per MPI task => total: 16 x 2 = 32 CPUs/node
#SBATCH --mem-per-cpu=2000M      # memory per CPU (in MB)
#SBATCH --time=0-01:00           # time limit (D-HH:MM)
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 gromacs/2024.4
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
srun --cpus-per-task=$OMP_NUM_THREADS gmx_mpi mdrun -deffnm md
```

**File:** `gromacs_whole_node_cedar.sh`

```bash
#!/bin/bash
#SBATCH --nodes=1                        # number of nodes
#SBATCH --ntasks-per-node=24             # request 24 MPI tasks per node
#SBATCH --cpus-per-task=2                # 2 OpenMP threads per MPI task => total: 24 x 2 = 48 CPUs/node
#SBATCH --constraint="[skylake|cascade]" # restrict to AVX512 capable nodes.
#SBATCH --mem-per-cpu=2000M              # memory per CPU (in MB)
#SBATCH --time=0-01:00                   # time limit (D-HH:MM)
module purge
module load arch/avx512 # switch architecture for up to 30% speedup
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 gromacs/2024.4
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
srun --cpus-per-task=$OMP_NUM_THREADS gmx_mpi mdrun -deffnm md
```

**File:** `gromacs_whole_node_beluga.sh` (two versions provided in original)

```bash
#!/bin/bash
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=20     # request 20 MPI tasks per node
#SBATCH --cpus-per-task=2        # 2 OpenMP threads per MPI task => total: 20 x 2 = 40 CPUs/node
#SBATCH --mem-per-cpu=2000M      # memory per CPU (in MB)
#SBATCH --time=0-01:00           # time limit (D-HH:MM)
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 gromacs/2024.4
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
srun --cpus-per-task=$OMP_NUM_THREADS gmx_mpi mdrun -deffnm md
```

```bash
#!/bin/bash
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=32     # request 32 MPI tasks per node
#SBATCH --cpus-per-task=2        # 2 OpenMP threads per MPI task => total: 32 x 2 = 64 CPUs/node
#SBATCH --mem-per-cpu=2000M      # memory per CPU (in MB)
#SBATCH --time=0-01:00           # time limit (D-HH:MM)
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 gromacs/2024.4
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
srun --cpus-per-task=$OMP_NUM_THREADS gmx_mpi mdrun -deffnm md
```

**File:** `gromacs_whole_node_niagara.sh`

```bash
#!/bin/bash
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=10     # request 10 MPI tasks per node
#SBATCH --cpus-per-task=4        # 4 OpenMP threads per MPI task => total: 10 x 4 = 40 CPUs/node
#SBATCH --mem-per-cpu=2000M      # memory per CPU (in MB)
#SBATCH --time=0-01:00           # time limit (D-HH:MM)
module purge --force
module load CCEnv
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 gromacs/2024.4
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
srun --cpus-per-task=$OMP_NUM_THREADS gmx_mpi mdrun -deffnm md
```

#### GPU job

Please read [Using GPUs with Slurm](link-to-using-gpus-with-slurm) for general information on using GPUs on our systems. This is a job script for `mdrun` using 4 OpenMP threads and one GPU:

**File:** `gpu_gromacs_job.sh`

```bash
#!/bin/bash
#SBATCH --gpus-per-node=1        # request 1 GPU per node
#SBATCH --cpus-per-task=4        # number of OpenMP threads per MPI process
#SBATCH --mem-per-cpu=2000M      # memory limit per CPU core (megabytes)
#SBATCH --time=0:30:00           # time limit (D-HH:MM:ss)
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 cuda/12.2 gromacs/2024.4
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
gmx mdrun -ntomp ${SLURM_CPUS_PER_TASK:-1} -deffnm md
```

##### Notes on running GROMACS on GPUs

Note that using more than a single GPU usually leads to poor efficiency. Carefully test and compare multi-GPU and single-GPU performance before deciding to use more than one GPU for your simulations.

GROMACS versions 2020.0 up to and including 2021.5 contain a bug when used on GPUs of Volta or newer generations (i.e., V100, T4, and A100) with `mdrun` option `-update gpu` that could have perturbed the virial calculation and, in turn, led to incorrect pressure coupling. The GROMACS developers state in the 2021.6 Release Notes: [3] The GPU update is not enabled by default, so the error can only appear in simulations where it was manually selected, and even in this case the error might be rare since we have not observed it in practice in the testing we have performed. Further discussion of this bug can be found in the GitLab issue #4393 of the GROMACS project. [4]

Our clusters (Beluga, Cedar, Graham, and Narval) have differently configured GPU nodes. On the page [Using GPUs with Slurm#Available GPUs](link-to-available-gpus) you can find more information about the different node configurations (GPU models and number of GPUs and CPUs per node). GROMACS imposes a number of constraints for choosing the number of GPUs, tasks (MPI ranks), and OpenMP threads. For GROMACS 2018.2 the constraints are:

*   The number of `--tasks-per-node` always needs to be the same as, or a multiple of the number of GPUs (`--gpus-per-node`).
*   GROMACS will not run GPU runs with only 1 OpenMP thread unless forced by setting the `-ntomp` option.
*   According to GROMACS developers, the optimum number of `--cpus-per-task` is between 2 and 6.
*   Avoid using a larger fraction of CPUs and memory than the fraction of GPUs you have requested in a node.

You can explore some benchmark results on our [MDBench portal](link-to-mdbench-portal).

##### Running multiple simulations on a GPU

GROMACS and other MD simulation programs are unable to fully use recent GPU models such as the Nvidia A100 and H100 unless the molecular system is very large (millions of atoms). Running a typical simulation on such a GPU wastes a significant fraction of the allocated computational resources.

There are two recommended solutions to this problem. The first one is to run multiple simulations on a single GPU using `mdrun -multidir` as described below. This is the preferred solution if you run multiple similar simulations, for instance:

*   Repeating the same simulation to acquire more conformational space sampling
*   Simulating multiple protein variants, multiple small ligands in complex with the same protein, multiple temperatures or ionic concentrations, etc.
*   Ensemble-based simulations such as replica exchange

Similar simulations are needed to ensure proper load balancing. If the simulations are dissimilar, some will progress faster and finish earlier than others, leading to idle resources.

The following job script runs three similar simulations in separate directories (`sim1`, `sim2`, `sim3`) using a single GPU. If you change the number of simulations, make sure to adjust `--ntasks-per-node` and `--cpus-per-task`: there should be one task per simulation, while the total number of CPU cores should remain constant.

**File:** `gpu_gromacs_job_multidir.sh`

```bash
#!/bin/bash
#SBATCH --gpus-per-node=1        # request 1 GPU per node
#SBATCH --ntasks-per-node=3      # number of MPI processes and simulations
#SBATCH --cpus-per-task=4        # number of OpenMP threads per MPI process
#SBATCH --mem-per-cpu=2000M      # memory limit per CPU core (megabytes)
#SBATCH --time=0:30:00           # time limit (D-HH:MM:ss)
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 cuda/12.2 gromacs/2024.4
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
srun gmx_mpi mdrun -ntomp ${SLURM_CPUS_PER_TASK:-1} -deffnm md -multidir sim1 sim2 sim3
```

The second solution is to use a MIG instance (a fraction of a GPU) rather than a full GPU. This is the preferred solution if you have a single simulation or if your simulations are dissimilar, for instance:

*   Systems with different sizes (more than a 10% difference in the numbers of atoms)
*   Systems with different shapes or compositions, such as a membrane-bound versus a soluble protein

Note that Hyper-Q / MPS should never be used with GROMACS. The built-in `-multidir` option achieves the same functionality more efficiently.


## Usage

More content for this section will be added at a later time.

### System preparation

In order to run a simulation, one needs to create a `tpr` file (portable binary run input file). This file contains the starting structure of the simulation, the molecular topology, and all the simulation parameters. `Tpr` files are created with the `gmx grompp` command (or simply `grompp` for versions older than 5.0). Therefore one needs the following files:

*   The coordinate file with the starting structure. GROMACS can read the starting structure from various file formats, such as `.gro`, `.pdb`, or `.cpt` (checkpoint).
*   The (system) topology (`.top`) file. It defines which force field is used and how the force field parameters are applied to the simulated system. Often the topologies for individual parts of the simulated system (e.g., molecules) are placed in separate `.itp` files and included in the `.top` file using a `#include` directive.
*   The run parameter (`.mdp`) file. See the GROMACS user guide for a detailed description of the options.

`Tpr` files are portable, that is they can be `grompp`ed on one machine, copied over to a different machine and used as an input file for `mdrun`. One should always use the same version for both `grompp` and `mdrun`. Although `mdrun` is able to use `tpr` files that have been created with an older version of `grompp`, this can lead to unexpected simulation results.


### Running simulations

MD Simulations often take much longer than the maximum walltime for a job to complete and therefore need to be restarted. To minimize the time a job needs to wait before it starts, you should maximize the number of nodes you have access to by choosing a shorter running time for your job. Requesting a walltime of 24 hours or 72 hours (three days) is often a good trade-off between waiting time and running time.

You should use the `mdrun` parameter `-maxh` to tell the program the requested walltime so that it gracefully finishes the current timestep when reaching 99% of this walltime. This causes `mdrun` to create a new checkpoint file at this final timestep and gives it the chance to properly close all output files (trajectories, energy- and log-files, etc.). For example use `#SBATCH --time=24:00` along with `gmx mdrun -maxh 24 ...` or `#SBATCH --time=3-00:00` along with `gmx mdrun -maxh 72 ...`.

**File:** `gromacs_job.sh`

```bash
#!/bin/bash
#SBATCH --nodes=1                # number of Nodes
#SBATCH --tasks-per-node=32      # number of MPI processes per node
#SBATCH --mem-per-cpu=4000       # memory limit per CPU (megabytes)
#SBATCH --time=24:00:00          # time limit (D-HH:MM:ss)
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 gromacs/2024.4
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
srun gmx_mpi mdrun -deffnm md -maxh 24
```

#### Restarting simulations

You can restart a simulation by using the same `mdrun` command as the original simulation and adding the `-cpi state.cpt` parameter where `state.cpt` is the filename of the most recent checkpoint file. `Mdrun` will by default (since version 4.5) try to append to the existing files (trajectories, energy- and log-files, etc.). GROMACS will check the consistency of the output files and - if needed - discard timesteps that are newer than that of the checkpoint file. Using the `-maxh` parameter ensures that the checkpoint and output files are written in a consistent state when the simulation reaches the time limit. The GROMACS manual contains more detailed information [5] [6].

**File:** `gromacs_job_restart.sh`

```bash
#!/bin/bash
#SBATCH --nodes=1                # number of Nodes
#SBATCH --tasks-per-node=32      # number of MPI processes per node
#SBATCH --mem-per-cpu=4000       # memory limit per CPU (megabytes)
#SBATCH --time=24:00:00          # time limit (D-HH:MM:ss)
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 gromacs/2024.4
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
srun gmx_mpi mdrun -deffnm md -maxh 24 -cpi md.cpt
```

#### Checkpointing simulations

You can use GROMACS’ ability to restart a simulation to split a long simulation over multiple short jobs. Shorter jobs wait less in the queue. In particular, those that request 3 hours or less are eligible for backfill scheduling. (See our [job scheduling policies](link-to-job-scheduling-policies).) This is especially useful if your research group has only a default resource allocation (e.g., `def-sponsor`) on the cluster, but will benefit even those with competitive resource allocations (e.g., `rrg-sponsor`).

By using a job array, you can automate checkpointing. With an array job script such as the following, a single `sbatch` call submits multiple short jobs, but only the first one is eligible to start. As soon as this first job has