# GROMACS

This page is a translated version of the page [GROMACS](https://docs.alliancecan.ca/mediawiki/index.php?title=GROMACS) and the translation is 100% complete.

Other languages: [English](https://docs.alliancecan.ca/mediawiki/index.php?title=GROMACS), [français](https://docs.alliancecan.ca/mediawiki/index.php?title=GROMACS/fr)


## General Information

GROMACS is a molecular dynamics simulation software package capable of handling systems ranging from a few hundred to several million particles.  Initially designed for biochemical molecules with many complex bonded and non-bonded interactions such as proteins, lipids, and nucleic acids, GROMACS is also used by many research groups on non-biological systems (e.g., polymers) due to its speed in calculating the non-bonded interactions, which are often dominant in the computational time of a simulation.


### Strengths

*   Very good performance compared to other applications.
*   Since GROMACS 4.6, excellent CUDA acceleration on GPUs with Nvidia compute capability >= 2.0 (e.g., Fermi or later).
*   Large selection of trajectory analysis tools.
*   Parallel execution with standard MPI protocol or the Thread MPI library for single-node workstations.
*   Free software under the LGPL version 2.1 (GNU Lesser General Public License).


### Weaknesses

To increase simulation speed, interactive analysis and/or data collection are reduced.  It can therefore be difficult to obtain non-standard information about the system being simulated.

Simulation methods and default parameters vary greatly from one version to another. It can be difficult to reproduce the same results with different versions.

The analysis tools and utilities added to the program are not always of the best quality: they may contain bugs and the methods are often poorly documented. We recommend using independent methods to verify the results.


### GPU Usage

The first part of the log file describes the configuration and indicates whether the version used allows the use of GPUs. GROMACS automatically uses any detected GPU.

GROMACS uses both CPUs and GPUs; performance depends on a reasonable balance between the two.

The new neighbor structure required the addition of the new cutoff-scheme variable to the MDP file.

The behavior of versions before 4.6 corresponds to `cutoff-scheme = group`, while to use GPU acceleration it is necessary to use `cutoff-scheme = verlet`, which is the default in version 5.


## Getting Started

This section covers configuration details.


### Environment Modules

The following versions are available:

*   StdEnv/2023
*   StdEnv/2020
*   StdEnv/2018.3
*   StdEnv/2016.4

| GROMACS Version | Modules for running on CPUs | Modules for running on GPUs (CUDA) | Remarks |
|---|---|---|---|
| gromacs/2024.4 | StdEnv/2023  gcc/12.3  openmpi/4.1.5  gromacs/2024.4 | StdEnv/2023 gcc/12.3  openmpi/4.1.5  cuda/12.2  gromacs/2024.4 | GCC, FlexiBLAS & FFTW |
| gromacs/2024.1 | StdEnv/2023  gcc/12.3  openmpi/4.1.5  gromacs/2024.1 | StdEnv/2023 gcc/12.3  openmpi/4.1.5  cuda/12.2  gromacs/2024.1 | GCC, FlexiBLAS & FFTW |
| gromacs/2023.3 | StdEnv/2023  gcc/12.3  openmpi/4.1.5  gromacs/2023.3 | StdEnv/2023 gcc/12.3  openmpi/4.1.5  cuda/12.2  gromacs/2023.3 | GCC, FlexiBLAS & FFTW |
| gromacs/2023.2 | StdEnv/2020  gcc/9.3.0  openmpi/4.0.3  gromacs/2023.2 | StdEnv/2020  gcc/9.3.0  cuda/11.4  openmpi/4.0.3  gromacs/2023.2 | GCC, FlexiBLAS & FFTW |
| gromacs/2023 | StdEnv/2020  gcc/9.3.0  openmpi/4.0.3  gromacs/2023 | StdEnv/2020  gcc/9.3.0  cuda/11.4  openmpi/4.0.3  gromacs/2023 | GCC, FlexiBLAS & FFTW |
| gromacs/2022.3 | StdEnv/2020  gcc/9.3.0  openmpi/4.0.3  gromacs/2022.3 | StdEnv/2020  gcc/9.3.0  cuda/11.4  openmpi/4.0.3  gromacs/2022.3 | GCC, FlexiBLAS & FFTW |
| gromacs/2022.2 | StdEnv/2020  gcc/9.3.0  openmpi/4.0.3  gromacs/2022.2 |  | GCC, FlexiBLAS & FFTW |
| gromacs/2021.6 | StdEnv/2020  gcc/9.3.0  openmpi/4.0.3  gromacs/2021.6 | StdEnv/2020  gcc/9.3.0  cuda/11.4  openmpi/4.0.3  gromacs/2021.6 | GCC, FlexiBLAS & FFTW |
| gromacs/2021.4 | StdEnv/2020  gcc/9.3.0  openmpi/4.0.3  gromacs/2021.4 | StdEnv/2020  gcc/9.3.0  cuda/11.4  openmpi/4.0.3  gromacs/2021.4 | GCC, FlexiBLAS & FFTW |
| gromacs/2021.2 | StdEnv/2020  gcc/9.3.0  cuda/11.4  openmpi/4.0.3  gromacs/2021.2 | StdEnv/2020  gcc/9.3.0  openmpi/4.0.3  gromacs/2021.2 | GCC & MKL |
| gromacs/2020.6 | StdEnv/2020  gcc/9.3.0  openmpi/4.0.3  gromacs/2020.6 | StdEnv/2020  gcc/9.3.0  cuda/11.4  openmpi/4.0.3  gromacs/2020.6 | GCC, FlexiBLAS & FFTW |
| gromacs/2020.4 | StdEnv/2020  gcc/9.3.0  cuda/11.4  openmpi/4.0.3  gromacs/2020.4 | StdEnv/2020  gcc/9.3.0  openmpi/4.0.3  gromacs/2020.4 | GCC & MKL |
| gromacs/2020.2 | StdEnv/2018.3  gcc/7.3.0 openmpi/3.1.2 gromacs/2020.2 | StdEnv/2018.3  gcc/7.3.0 cuda/10.0.130 openmpi/3.1.2  gromacs/2020.2 | GCC & MKL |
| gromacs/2019.6 | StdEnv/2018.3  gcc/7.3.0 openmpi/3.1.2 gromacs/2019.6 | StdEnv/2018.3  gcc/7.3.0 cuda/10.0.130 openmpi/3.1.2  gromacs/2019.6 | GCC & MKL |
| gromacs/2019.3 | StdEnv/2018.3  gcc/7.3.0 openmpi/3.1.2 gromacs/2019.3 | StdEnv/2018.3  gcc/7.3.0 cuda/10.0.130 openmpi/3.1.2  gromacs/2019.3 | GCC & MKL ‡ |
| gromacs/2018.7 | StdEnv/2018.3  gcc/7.3.0 openmpi/3.1.2 gromacs/2018.7 | StdEnv/2018.3  gcc/7.3.0 cuda/10.0.130 openmpi/3.1.2  gromacs/2018.7 | GCC & MKL |
| gromacs/2018.3 | StdEnv/2016.4  gcc/6.4.0 openmpi/2.1.1 gromacs/2018.3 | StdEnv/2016.4  gcc/6.4.0 cuda/9.0.176 openmpi/2.1.1  gromacs/2018.3 | GCC & FFTW |
| gromacs/2018.2 | StdEnv/2016.4  gcc/6.4.0 openmpi/2.1.1 gromacs/2018.2 | StdEnv/2016.4  gcc/6.4.0 cuda/9.0.176 openmpi/2.1.1  gromacs/2018.2 | GCC & FFTW |
| gromacs/2018.1 | StdEnv/2016.4  gcc/6.4.0 openmpi/2.1.1 gromacs/2018.1 | StdEnv/2016.4  gcc/6.4.0 cuda/9.0.176 openmpi/2.1.1  gromacs/2018.1 | GCC & FFTW |
| gromacs/2018 | StdEnv/2016.4  gromacs/2018 | StdEnv/2016.4  cuda/9.0.176 gromacs/2018 | Intel & MKL |
| gromacs/2016.5 | StdEnv/2016.4  gcc/6.4.0  openmpi/2.1.1 gromacs/2016.5 | StdEnv/2016.4  gcc/6.4.0  cuda/9.0.176  openmpi/2.1.1 gromacs/2016.5 | GCC & FFTW |
| gromacs/2016.3 | StdEnv/2016.4  gromacs/2016.3 | StdEnv/2016.4  cuda/8.0.44 gromacs/2016.3 | Intel & MKL |
| gromacs/5.1.5 | StdEnv/2016.4  gromacs/5.1.5 | StdEnv/2016.4  cuda/8.0.44 gromacs/5.1.5 | Intel & MKL |
| gromacs/5.1.4 | StdEnv/2016.4  gromacs/5.1.4 | StdEnv/2016.4  cuda/8.0.44 gromacs/5.1.4 | Intel & MKL |
| gromacs/5.0.7 | StdEnv/2016.4  gromacs/5.0.7 | StdEnv/2016.4  cuda/8.0.44 gromacs/5.0.7 | Intel & MKL |
| gromacs/4.6.7 | StdEnv/2016.4  gromacs/4.6.7 | StdEnv/2016.4  cuda/8.0.44 gromacs/4.6.7 | Intel & MKL |
| gromacs/4.6.7 | StdEnv/2016.4  gcc/5.4.0  openmpi/2.1.1 gromacs/4.6.7 | StdEnv/2016.4  gcc/5.4.0  cuda/8.0  openmpi/2.1.1  gromacs/4.6.7 | GCC & MKL & ThreadMPI |

**Remarks:**

*   Versions 2020.0 to 2021.5 inclusive contain a bug when used with Volta generation GPUs or later (V100, T4, and A100) with the `-update gpu` option of `mdrun` that could have disrupted the virial calculation and thus distorted the pressure coupling. In the release notes for version 2021.6, we read: [1]  The update is not enabled by default on the GPU and therefore the error can only occur in simulations where the `-update gpu` option has been explicitly selected; even in this case, the error may be rare as we have not observed it in practice in the tests we have performed.  You can find more information in GitLab, concerning issue #4393 of the GROMACS project. [2]
*   Versions since 2020.4 have been compiled for the standard software environment StdEnv/2020.
*   Versions 2018.7 and later have been compiled with GCC compilers and the MKL library as they slightly improve performance.
*   Previous versions were compiled with either GCC compilers and FFTW, or with Intel MKL compilers with Open MPI 2.1.1 libraries from the default environment, as indicated in the table above.
*   CPU (non-GPU) versions are available in single and double precision except for 2019.3 (‡), where double precision is not available for AVX512.

To load these modules, use the `module load` command with the names indicated in the table above, for example:

```bash
$ module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 gromacs/2024.4
```

or

```bash
$ module load StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 gromacs/2023.2
```

These versions also use GPUs, but only in single precision. To load the version using GPUs, first load the `cuda` module. For names, see the table above.

```bash
$ module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 cuda/12.2 gromacs/2024.4
```

or

```bash
$ module load StdEnv/2020 gcc/9.3.0 cuda/11.4 openmpi/4.0.3 gromacs/2023.2
```

See [Using Modules](link_to_module_usage_doc) for more information on environment modules.


### Suffixes

#### Versions 5.x, 2016.x and later

Versions 5 and later include four binaries, all possessing the full functionality of GROMACS.  Tools from previous versions have been implemented as subcommands of the `gmx` binaries. See [GROMACS 5.0 Tool Changes](link_to_5_0_tool_changes) and the [GROMACS documentation](link_to_gromacs_docs).

*   `gmx` - GROMACS in mixed (single) precision with OpenMP, but without MPI
*   `gmx_mpi` - GROMACS in mixed (single) precision with OpenMP and MPI
*   `gmx_d` - GROMACS in double precision with OpenMP, but without MPI
*   `gmx_mpi_d` - GROMACS in double precision with OpenMP and MPI


#### Version 4.6.7

Double-precision binaries have the suffix `_d`.

The single and double precision parallel `mdrun` binaries are:

*   `mdrun_mpi`
*   `mdrun_mpi_d`


### Job Submission Scripts

See [Running Jobs](link_to_running_jobs_doc) on how to use the Slurm scheduler.


#### Sequential Jobs

Here is a simple script for the sequential `mdrun` task.

**File: `serial_gromacs_job.sh`**

```bash
#!/bin/bash
#SBATCH --time=0-0:30         # time limit (D-HH:MM)
#SBATCH --mem-per-cpu=1000M   # memory per CPU (in MB)
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 gromacs/2024.4
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
gmx mdrun -nt 1 -deffnm em
```

The molecular system simulation will be executed in the `em.tpr` file.


#### Whole Nodes

Systems simulated by GROMACS are usually so large that you will want to use multiple entire nodes.  The product of `--ntasks-per-node=` by `--cpus-per-task` generally corresponds to the number of CPU cores in the compute nodes of the cluster. See the [Performance](#performance) section below.

**Graham, Cedar, Beluga, Narval, Niagara:**

**File: `gromacs_whole_node_graham.sh`**

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

**File: `gromacs_whole_node_cedar.sh`**

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

**File: `gromacs_whole_node_beluga.sh`**

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

**File: `gromacs_whole_node_narval.sh`**

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

**File: `gromacs_whole_node_niagara.sh`**

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


#### GPU Tasks

For more information, see [Slurm Scheduling of GPU Jobs](link_to_slurm_gpu_doc).

This task for `mdrun` uses 4 OpenMP threads and one (1) GPU.

**File: `gpu_gromacs_job.sh`**

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


##### Working with GPUs

It should be noted that using more than one GPU usually causes poor efficiency. Before using multiple GPUs for your simulations, perform benchmark tests with a single GPU and with multiple GPUs to evaluate performance.

Versions 2020.0 to 2021.5 inclusive contain a bug when used with Volta generation GPUs or later (V100, T4, and A100) with the `-update gpu` option of `mdrun` that could have disrupted the virial calculation and thus distorted the pressure coupling. In the release notes for version 2021.6, we read: [3] The update is not enabled by default on the GPU and therefore the error can only occur in simulations where the `-update gpu` option has been explicitly selected; even in this case, the error may be rare as we have not observed it in practice in the tests we have performed. You can find more information in GitLab, concerning issue #4393 of the GROMACS project. [4]

GPU nodes are configured differently on our clusters.

*   Cedar offers 4 GPUs and 24 CPU cores per node
*   Graham offers 2 GPUs and 32 CPU cores per node.

The parameters are different if you want to use all the GPUs and CPU cores of a node.

*   **Cedar:** `--gres=gpu:p100:4  --ntasks-per-node=4 --cpus-per-task=6`
*   **Graham:** `--gres=gpu:p100:2  --ntasks-per-node=4 --cpus-per-task=8`

Of course, the simulated system must be large enough to use all resources.

GROMACS imposes certain constraints on the choice of the number of GPUs, tasks (MPI rank), and OpenMP threads. For version 2018.2, the constraints are: `--tasks-per-node` must be a multiple of the number of GPUs (`--gres=gpu:`). GROMACS works with only one OpenMP thread only if the `-ntomp` option is used. The optimal number of `--cpus-per-task` is between 2 and 6, according to the developers. Avoid using a fraction of CPU and memory larger than the fraction of GPU you request in a node. Consult the results we obtained on our portal [MOLECULAR DYNAMICS PERFORMANCE GUIDE](link_to_performance_guide).


##### Running Multiple Simulations on a Single GPU

GROMACS and other MD simulation programs cannot fully exploit recent GPU models such as the Nvidia A100 and H100 unless the molecular system is very large (millions of atoms). Running a classical simulation on such a GPU wastes a significant portion of the allocated compute resources.

Two solutions are recommended to solve this problem. The first is to run multiple simulations on a single GPU using `mdrun -multidir`, as described below. This is the ideal solution if you are running several similar simulations, for example:

*   Repeating the same simulation to obtain a more precise sampling of the conformational space
*   Simulating several protein variants, several small ligands in complex with the same protein, several temperatures or ionic concentrations, etc.
*   Simulating based on ensembles, such as replica exchange

Similar simulations are needed to ensure adequate load balancing. If the simulations are dissimilar, some will progress faster and finish sooner than others, resulting in wasted resources.

The following job script runs three similar simulations in separate directories (`sim1`, `sim2`, `sim3`) with a single GPU. If you change the number of simulations, be sure to adjust the `--ntasks-per-node` and `--cpus-per-task` parameters. One task per simulation must be run, while the total number of CPU cores must remain constant.

**Narval**

**File: `gpu_gromacs_job_multidir.sh`**

```bash
#!/bin/bash
#SBATCH --gpus-per-node=1        # request 1 GPU per node
#SBATCH --ntasks-per-node=3      # number of MPI processes and simulations
#SBATCH --cpus-per-task=4        # number of OpenMP threads per MPI process
#SBATCH --mem-per-cpu=2000M      # memory limit per CPU core (megabytes)
#SBATCH --time=0:30:00           # time limit (D-HH:MM:ss)
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 cuda/12.2 gromacs/2024.4
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
srun gmx_mpi mdrun -ntomp ${SLURM_CPUS_PER_TASK:-1} -deffnm md \
-multidir sim1 sim2 sim3
```

The second solution is to use a MIG instance (a fraction of a GPU) rather than a full GPU. This is ideal if you only have one simulation or if your simulations are dissimilar, for example:

*   systems of different sizes (difference of more than 10% in the number of atoms);
*   systems of different shapes or compositions, such as a membrane protein and a soluble protein.

Note that Hyper-Q/MPS should never be used with GROMACS. The built-in `-multidir` option provides the same functionality more efficiently.


## Usage

[This section is under construction]


### System Preparation

Running a simulation requires a binary input file `tpr` (portable binary run input file) that contains the starting simulation structure, molecular topology, and simulation parameters. The command `gmx grompp` creates the `tpr` files; for versions prior to 5.0, the command is `grompp`. You need the following files:

*   The coordinate file with the starting structure; valid file formats are `.gro`, `.pdb`, or `.cpt` (GROMACS checkpoint).
*   The topology file of the system in `top` format; this determines the force field and how the force field parameters apply to the simulated system. The topologies of the individual parts of the simulated system (i.e., the molecules) are placed in separate `itp` files and are included in the `top` file with the `#include` command.
*   The `mdp` file of execution parameters. Consult the GROMACS guide for a detailed description of the options.

`tpr` files are portable and can therefore be grompp-ed on one cluster, copied to another cluster, and used as input files for `mdrun`. Always use the same GROMACS version for `grompp` and `mdrun`. Even if `mdrun` can use `tpr` files created with an older version of `grompp`, the simulation may yield unexpected results.


### Running a Simulation

MD simulations often take longer to complete than the maximum real-time allowed for a job; they must therefore be restarted. To minimize waiting time before a job starts, you can maximize the number of nodes you have access to by choosing a shorter execution time. A good compromise between waiting time and execution time is often to request a real-time duration of 24 or 72 hours.

You should use the `-maxh` parameter of `mdrun` to indicate to the program the real-time duration for the current step to complete successfully when the duration reaches 99%. In this way, `mdrun` creates a checkpoint file at this step and allows it to properly close all output files (trajectories, energy, logging, etc.). For example, use `#SBATCH --time=24:00` with `gmx mdrun -maxh 24 ...` or `#SBATCH --time=3-00:00` with `gmx mdrun -maxh 72 ...`.

**File: `gromacs_job.sh`**

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


#### Restarting a Simulation

You can restart a simulation with the same `mdrun` command as for the original simulation by adding the parameter `-cpi state.cpt`, where `state.cpt` is the name of the last checkpoint file. Since version 4.5, `mdrun` attempts by default to use existing files (trajectories, energy, logging, etc.). GROMACS checks the consistency between the output files and rejects more recent steps than the checkpoint file if necessary. The `-maxh` parameter ensures that the control and output files are consistent when the simulation reaches the execution time limit. For more information, see the GROMACS documentation [5][6].

**File: `gromacs_job_restart.sh`**

```bash
#!/bin/bash
#SBATCH --nodes=1                # number of Nodes
#SBATCH --tasks-per-node=32      # number of MPI processes per node
#SBATCH --mem-per-cpu=4000       # memory limit per CPU (megabytes)
#SBATCH --time=24:00:00          # time limit (D-HH:MM:ss)
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 gromacs/2024.4
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
srun gmx_mpi mdrun -deffnm md -maxh 24.0 -cpi md.cpt
```


#### Chunking Simulations

The simulation restart functionality can be used to divide it into several short tasks. Short tasks are shorter to execute. In particular, those that take three hours or less are eligible for replacement scheduling. (See our [job scheduling policies](link_to_scheduling_policies)). This is particularly useful if your research group only has a default resource allocation (e.g., `def-sponsor`) on the cluster, but will also be beneficial for those with competitive resource allocations (e.g., `rrg-sponsor`).

Using a job array, you can automate checkpoints. With the following script, a single call to `sbatch` submits multiple short tasks, but only the first one can start. As soon as this first task is finished, the next one can start and resume the simulation. This process is repeated until all tasks are finished or the simulation itself is finished, after which pending tasks are automatically cancelled.

**Whole nodes (Narval)**

**GPU job**

**File: `gromacs_job_checkpoint.sh`**

```bash
#!/bin/bash
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=32     # request 32 MPI tasks per node
#SBATCH --cpus-per-task=2        # 2 OpenMP threads per MPI task
#SBATCH --mem-per-cpu=2000M      # memory per CPU (in MB)
#SBATCH --time=03:00:00          # time limit (D-HH:MM:ss)
#SBATCH --array=1-20%1           # job range, running only 1 at a time
module load StdEnv/202