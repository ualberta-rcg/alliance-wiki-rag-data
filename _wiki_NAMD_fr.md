# NAMD

NAMD is an object-oriented molecular dynamics program designed for simulating large biomolecular systems. Simulations are prepared and analyzed using the VMD visualization software.

## Contents

* [Installation](#installation)
* [Environment Modules](#environment-modules)
* [Scripts](#scripts)
    * [Sequential and Multithreaded Jobs](#sequential-and-multithreaded-jobs)
    * [Parallel Jobs on CPUs](#parallel-jobs-on-cpus)
        * [MPI Jobs](#mpi-jobs)
        * [Verbs Jobs](#verbs-jobs)
        * [UCX Jobs](#ucx-jobs)
        * [OFI Jobs](#ofi-jobs)
    * [Jobs Using 1 GPU](#jobs-using-1-gpu)
    * [Jobs Using Multiple GPUs](#jobs-using-multiple-gpus)
        * [UCX Jobs with GPUs](#ucx-jobs-with-gpus)
        * [OFI Jobs with GPUs](#ofi-jobs-with-gpus)
        * [Verbs Jobs with GPUs](#verbs-jobs-with-gpus)
* [Performance and Benchmarking](#performance-and-benchmarking)
* [NAMD 3](#namd-3)
* [References](#references)


## Installation

The software is installed by our technical team and is available via modules. If you require a newer version, need to perform the installation yourself, or have any questions, please contact [technical support](link-to-support).


## Environment Modules

The latest version 3.0.1 is installed on all our clusters. The previous version 2.14 is also available. Versions 2.13 and 2.12 are also available. For jobs running on multiple nodes, use UCX.


## Scripts

Information on the Slurm scheduler can be found on the [Running Jobs](link-to-running-jobs) page.


### Sequential and Multithreaded Jobs

The following script is for a sequential simulation on a single core. To use more cores, increase the value of `--cpus-per-task`, without exceeding the number of cores available on the node.

**File:** `serial_namd_job.sh`

```bash
#!/bin/bash
#
#SBATCH --cpus-per-task=1
#SBATCH --mem 2048            # memory in Mb, increase as needed
#SBATCH -o slurm.%N.%j.out    # STDOUT file
#SBATCH -t 0:05:00            # time (D-HH:MM), increase as needed
#SBATCH --account=def-specifyaccount
# these are simple examples, please experiment with additional flags to improve your runtimes
# in particular, adding  +setcpuaffinity  flag may improve performance
# commands for NAMD version 3.0.1
module load StdEnv/2023 gcc/12.3 namd-multicore/3.0.1
namd3 +p $SLURM_CPUS_PER_TASK +idlepoll apoa1.namd
# commands for NAMD version 2.14
module load StdEnv/2020 namd-multicore/2.14
namd2 +p $SLURM_CPUS_PER_TASK +idlepoll apoa1.namd
```


### Parallel Jobs on CPUs

#### MPI Jobs

**NOTE:** Do not use MPI, but rather UCX.


#### Verbs Jobs

**NOTE:** With NAMD 2.14, use UCX on other clusters. The instructions below only apply to NAMD versions 2.13 and 2.12. These instructions will be updated once this configuration is tested on the new clusters.

In this example, we use 64 processes on 2 nodes, 32 processes per node, thus fully utilizing the 32 cores. We assume that entire nodes are used and `ntasks-per-node` should be 32 on Graham. For optimal performance, use entire nodes.

**NOTES:**

* Verbs versions do not work on Cedar due to different networking; use the MPI version instead.
* Verbs versions do not work on Beluga due to incompatibility with InfiniBand kernel drivers; use the UCX version instead.

**File:** `verbs_namd_job.sh`

```bash
#!/bin/bash
#
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --mem=0            # memory per node, 0 means all memory
#SBATCH -o slurm.%N.%j.out    # STDOUT
#SBATCH -t 0:05:00            # time (D-HH:MM)
#SBATCH --account=def-specifyaccount
NODEFILE=nodefile.dat
slurm_hl2hl.py --format CHARM > $NODEFILE
P=$SLURM_NTASKS
module load namd-verbs/2.12
CHARMRUN=`which charmrun`
NAMD2=`which namd2`
$CHARMRUN ++p $P ++nodelist $NODEFILE $NAMD2 +idlepoll apoa1.namd
```


#### UCX Jobs

The following example uses 80 processes, 40 processes per node on 2 nodes, thus fully utilizing the 80 cores. The script assumes that entire nodes are used; thus, on Beluga, `ntasks-per-node` should be 40. NAMD jobs that use entire nodes offer the best performance.

**NOTE:** UCX versions should work on all clusters.

**File:** `ucx_namd_job.sh`

```bash
#!/bin/bash
#
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=40
#SBATCH --mem=0            # memory per node, 0 means all memory
#SBATCH -o slurm.%N.%j.out    # STDOUT
#SBATCH -t 0:05:00            # time (D-HH:MM)
#SBATCH --account=def-specifyaccount
# these are simple examples, please experiment with additional flags to improve your runtimes
# in particular, adding  +setcpuaffinity  flag may improve performance
# commands for NAMD version 3.0.1
module load StdEnv/2023 gcc/13.3 namd-ucx/3.0.1
srun --mpi=pmi2 namd3 apoa1.namd
# commands for NAMD version 2.14
module load StdEnv/2020 namd-ucx/2.14
srun --mpi=pmi2 namd2 apoa1.namd
```


#### OFI Jobs

**NOTE:** OFI versions work **ONLY** on Cedar due to its different networking. Issues have been encountered with OFI and it is preferable to use UCX.

**File:** `ucx_namd_job.sh`

```bash
#!/bin/bash
#SBATCH --account=def-specifyaccount
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH -t 0:05:00            # time (D-HH:MM)
#SBATCH --mem=0            # memory per node, 0 means all memory
#SBATCH -o slurm.%N.%j.out    # STDOUT
module load StdEnv/2020 namd-ofi/2.14
srun --mpi=pmi2 namd2 stmv.namd
```


### Jobs Using 1 GPU

The next example uses eight CPU cores and one P100 GPU on a single node.

**Important:** NAMD 3 offers a new input parameter that directs more computations to the GPU. This can significantly improve performance. To use it, add the following line to your input file:

`GPUresident on;`

**File:** `multicore_gpu_namd_job.sh`

```bash
#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=2048
#SBATCH --time=0:15:00
#SBATCH --gpus-per-node=p100:1
#SBATCH --account=def-specifyaccount
# these are simple examples, please experiment with additional flags to improve your runtimes
# in particular, adding  +setcpuaffinity  flag may improve performance
# commands for NAMD version 3.0.1
module load StdEnv/2023 gcc/12.3 cuda/12.2 namd-multicore/3.0.1
namd3 +p $SLURM_CPUS_PER_TASK +idlepoll stmv.namd
# commands for NAMD version 2.14
module load StdEnv/2020 cuda/11.0 namd-multicore/2.14
namd2 +p $SLURM_CPUS_PER_TASK +idlepoll apoa1.namd
```


### Jobs Using Multiple GPUs

#### UCX Jobs with GPUs

This example is for Beluga and assumes that entire nodes are used, which offers better performance for NAMD jobs. The example uses 8 processes on 2 nodes, each process using 10 threads and 1 GPU. This fully utilizes Beluga's GPU nodes which have 40 cores and 4 GPUs per node. Note that 1 core per task is reserved for a communication thread; therefore, it is normal for NAMD to report only 72 cores used.

To use this script on another cluster, see the characteristics of the nodes available on the cluster and adjust the `--cpus-per-task` and `--gpus-per-node` options accordingly.

**NOTE:** UCX versions can be used on all clusters.

**File:** `ucx_namd_job.sh`

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10 # number of threads per task (process)
#SBATCH --gpus-per-node=v100:4
#SBATCH --mem=0            # memory per node, 0 means all memory
#SBATCH --time=0:15:00
#SBATCH --account=def-specifyaccount
module load StdEnv/2020 intel/2020.1.217 cuda/11.0 namd-ucx-smp/2.14
NUM_PES=$(expr $SLURM_CPUS_PER_TASK - 1)
srun --cpus-per-task=$SLURM_CPUS_PER_TASK --mpi=pmi2 namd2 ++ppn $NUM_PES apoa1.namd
```


#### OFI Jobs with GPUs

**NOTE:** OFI versions work **ONLY** on Cedar due to its different networking. Issues have been encountered with OFI and it is preferable to use UCX.

**File:** `ucx_namd_job.sh`

```bash
#!/bin/bash
#SBATCH --account=def-specifyaccount
#SBATCH --ntasks 8            # number of tasks
#SBATCH --nodes=2
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=p100:4
#SBATCH -t 0:05:00            # time (D-HH:MM)
#SBATCH --mem=0            # memory per node, 0 means all memory
module load StdEnv/2020 cuda/11.0 namd-ofi-smp/2.14
NUM_PES=$(expr $SLURM_CPUS_PER_TASK - 1)
srun --cpus-per-task=$SLURM_CPUS_PER_TASK --mpi=pmi2 namd2 ++ppn $NUM_PES stmv.namd
```


#### Verbs Jobs with GPUs

**NOTE:** With NAMD 2.14, use UCX GPUs on all clusters. The instructions below apply to NAMD 2.13 and 2.12.

This example uses 64 processes on 2 nodes, each node running 32 processes, thus fully utilizing the 32 cores. Each node uses 2 GPUs, so the job uses a total of 4 GPUs. The script assumes that entire nodes are used; on Graham, `ntasks-per-node` would be 32. NAMD jobs that use entire nodes offer the best performance.

**NOTE:** Verbs versions do not work on Cedar due to different networking.

**File:** `verbsgpu_namd_job.sh`

```bash
#!/bin/bash
#
#SBATCH --ntasks 64            # number of tasks
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --mem 0            # memory per node, 0 means all memory
#SBATCH --gpus-per-node=p100:2
#SBATCH -o slurm.%N.%j.out    # STDOUT
#SBATCH -t 0:05:00            # time (D-HH:MM)
#SBATCH --account=def-specifyaccount
slurm_hl2hl.py --format CHARM > nodefile.dat
NODEFILE=nodefile.dat
OMP_NUM_THREADS=32
P=$SLURM_NTASKS
module load cuda/8.0.44
module load namd-verbs-smp/2.12
CHARMRUN=`which charmrun`
NAMD2=`which namd2`
$CHARMRUN ++p $P ++ppn $OMP_NUM_THREADS ++nodelist $NODEFILE $NAMD2 +idlepoll apoa1.namd
```


## Performance and Benchmarking

The [Molecular Dynamics Performance Guide](link-to-performance-guide) was created by an ACENET team.  The guide describes optimal conditions for running jobs on our clusters with AMBER, GROMACS, and OpenMM.

Here is an example of benchmarking. The performance of NAMD varies depending on the simulated systems, particularly with the number of atoms. It is therefore very useful to perform the type of benchmarking shown here in cases where the simulation of a particular system would be long. The data collected can also be used to document your requests for resource allocation competitions.

To obtain relevant results, we suggest varying the number of steps so that the system simulation takes a few minutes and that the duration data collection is done at intervals of at least a few seconds. You may notice variations in your duration results if the execution time is too short.

The data below comes from a standard apoa1 benchmark performed on the Graham cluster which has 32-core CPU nodes and 32-core, 2-GPU GPU nodes. To do the same exercise with another cluster, you will need to consider the structure of its nodes.

In the first table, we use NAMD 2.12 loaded with the verbs module. Efficiency is calculated with (duration with 1 core) / (N * (duration with N cores)).

| # Cores | Real Time per Step | Efficiency |
|---|---|---|
| 1 | 0.8313 | 100% |
| 2 | 0.4151 | 100% |
| 4 | 0.1945 | 107% |
| 8 | 0.0987 | 105% |
| 16 | 0.0501 | 104% |
| 32 | 0.0257 | 101% |
| 64 | 0.0133 | 98% |
| 128 | 0.0074 | 88% |
| 256 | 0.0036 | 90% |
| 512 | 0.0021 | 77% |

In this case, we find that it is acceptable to use 256 cores to simulate the system. If you request more cores than necessary, your jobs will be waiting longer and your overall result will be affected.

In the next case, the benchmarking is done with the use of GPUs. The NAMD multicore module is used for simulations that can be performed with one (1) node while the verbs-smp module is used in cases of jobs requiring multiple nodes.

| # Cores | #GPU | Real Time per Step | Notes |
|---|---|---|---|
| 4 | 1 | 0.0165 | 1 node, multicore |
| 8 | 1 | 0.0088 | 1 node, multicore |
| 16 | 1 | 0.0071 | 1 node, multicore |
| 32 | 2 | 0.0045 | 1 node, multicore |
| 64 | 4 | 0.0058 | 2 nodes, verbs-smp |
| 128 | 8 | 0.0051 | 2 nodes, verbs-smp |

The data in the table clearly indicates that it is absolutely useless to use more than one node since performance decreases with 2 nodes or more. With 1 node, it is preferable to use 1 GPU/16 cores since the efficiency is maximum; the use of 2 GPUs/32 cores is acceptable if your results must be produced quickly. Since the Graham GPU nodes have the same priority order for all 1 GPU/16 core tasks, there is no advantage to using 4 or 8 cores.

One may wonder whether or not the simulation can use a GPU. The benchmarking results indicate that using a GPU node (2 GPUs/32 cores) on Graham processes the task faster than on 4 non-GPU nodes. Since the cost of a GPU node on Graham is almost twice that of a non-GPU node, it is more economical to use GPUs. This is what you should do as much as possible, but since CPU nodes are more numerous, you should also consider not using GPUs if the waiting time is too long.


## NAMD 3

NAMD 3 is available in a module. In some system configurations, performance may be better compared to NAMD 2.14.

To try it immediately, you can download the binary from the NAMD website and modify it for our systems with, for example (specify the version if needed):

```bash
tar xvfz NAMD_3.0alpha11_Linux-x86_64-multicore-CUDA-SingleNode.tar.gz
cd NAMD_3.0alpha11_Linux-x86_64-multicore-CUDA
setrpaths.sh  --path .
```

Subsequently, the `namd3` executable in this directory will be linked to the appropriate libraries. You can then submit a job that uses this executable.

For better performance with NAMD 3 on a GPU, we strongly recommend adding the following keyword to the configuration file, provided that the input configuration you are using allows it:

`GPUresident on;`

For more information on this parameter and its changes, see [this page](link-to-namd3-page).


## References

**To download:**

* [http://www.ks.uiuc.edu/Development/Download/download.cgi?PackageName=NAMD](http://www.ks.uiuc.edu/Development/Download/download.cgi?PackageName=NAMD) (you will be asked to register)

* NAMD User's guide for version 2.14
* NAMD User's guide for version 3.0.1
* NAMD version 3.0.1 release notes
* NAMD version 2.14 release notes

**Tutorials:**

* [http://www.ks.uiuc.edu/Training/Tutorials/](http://www.ks.uiuc.edu/Training/Tutorials/)

