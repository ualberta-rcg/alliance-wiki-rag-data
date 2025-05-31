# Graham

**Attention:** Graham will soon be replaced by a new system named Nibi. During the transition period, information on the capacity of each system, as well as service outages or reductions, will be available on the [Infrastructure Renewal](link-to-renewal-page) page.

**Availability:** Since 2017

**Login Node:** graham.alliancecan.ca

**Globus Endpoint:** computecanada#graham-globus

**Copy Node (rsync, scp, sftp, etc.):** gra-dtn1.alliancecan.ca

Graham is a heterogeneous system suitable for a wide variety of task types; it is located at the University of Waterloo.  Its name commemorates Wes Graham, the first director of the University of Waterloo Computing Centre.  The parallel file systems and persistent storage (often called NDC-Waterloo) are similar to those of Cedar. The interconnect is not the same, and there are different proportions of each type of compute node. A liquid cooling system uses heat exchangers on the back doors.


## Introduction to Graham

* [Running Jobs](#running-jobs)
* [Transferring Data](#transferring-data)


## Contents

1. [Particularities](#particularities)
2. [Storage](#storage)
3. [High-Performance Interconnect](#high-performance-interconnect)
4. [Visualization](#visualization)
5. [Node Characteristics](#node-characteristics)
6. [GPU](#gpu)
    * [Pascal GPU Nodes](#pascal-gpu-nodes)
    * [Volta GPU Nodes](#volta-gpu-nodes)
    * [Turing GPU Nodes](#turing-gpu-nodes)
    * [Ampere GPU Nodes](#ampere-gpu-nodes)
7. [Capacity Reduction](#capacity-reduction)


## Particularities

According to our policy, Graham compute nodes do not have internet access. To make an exception, contact [technical support](link-to-support) with the following information:

* IP Address:
* Port(s):
* Protocol: TCP or UDP
* Contact:
* End Date:

Before terminating the internet link, we will contact the contact person to verify if the rule is still necessary.

Crontab is not available on Graham.

A job should have a minimum duration of one hour and a maximum of 168 hours (seven days). For a test job, the minimum is five minutes.

The total number of running and pending jobs at the same time cannot exceed 1000. In the case of an array job, each job is counted individually.


## Storage

`/home` space

* Total volume of 133TB
* Location of `/home` directories
* Each `/home` directory has a small fixed quota
* Not allocated via the [Quick Access Service](link-to-service) or the [Resource Allocation Competition](link-to-competition); large-scale storage is done on the `/project` space
* Backed up daily

`/scratch` space

* Total volume of 3.2PB
* High-performance parallel file system
* Active or temporary `/scratch` storage
* Not allocated
* Large fixed quota per user
* Inactive data is purged

`/project` space

* Total volume of 16PB
* External persistent storage
* Allocations via the [Quick Access Service](link-to-service) or the [Resource Allocation Competition](link-to-competition)
* Not suitable for parallel read and write jobs; use the `/scratch` space
* Large adjustable quota per project
* Backed up daily


## High-Performance Interconnect

Mellanox FDR (56Gb/s) and EDR (100Gb/s) InfiniBand interconnect. FDR serves GPU nodes and cloud nodes; all other node types use EDR. A central 324-port director switch aggregates the connections of the 1024-core CPU and GPU islands. The 56 cloud nodes are located on the CPU nodes; they are grouped on a larger island and share 8 FDR links to the switch.

A high-bandwidth, low-latency non-blocking InfiniBand fabric connects all nodes and the `/scratch` storage.

Nodes configured for cloud service also have a 10Gb/s Ethernet network and 40Gb/s links to `/scratch` storage.

Graham's architecture has been planned to support multiple parallel tasks up to 1024 cores thanks to a non-blocking network.

For larger tasks, the blocking factor is 8:1; even for tasks running on multiple islands, the interconnect is high-performance.

[Diagram of interconnections for Graham](link-to-diagram)


## Visualization

Graham offers dedicated nodes for visualization that only allow VNC connections (gra-vdi.alliancecan.ca). For information on how to use them, see the [VNC](link-to-vnc) page.


## Node Characteristics

At the beginning of 2025, Graham's capacity was reduced to allow us to install the new Nibi system. The following table shows the nodes that are available as of February 2025.  All Graham nodes are equipped with Turbo Boost functionality.

| Nodes | Cores | Available Memory | CPU | Storage | GPU |
|---|---|---|---|---|---|
| 2 | 40 | 377G or 386048M | 2 x Intel Xeon Gold 6248 Cascade Lake @ 2.5GHz | 5.0TB NVMe SSD | 8 x NVIDIA V100 Volta (32GB HBM2 memory), NVLINK |
| 6 | 16 | 187G or 191840M | 2 x Intel Xeon Silver 4110 Skylake @ 2.10GHz | 11.0TB SATA SSD | 4 x NVIDIA T4 Turing (16GB GDDR6 memory) |
| 30 | 44 | 187G or 191840M | 2 x Intel Xeon Gold 6238 Cascade Lake @ 2.10GHz | 5.8TB NVMe SSD | 4 x NVIDIA T4 Turing (16GB GDDR6 memory) |
| 136 | 44 | 187G or 191840M | 2 x Intel Xeon Gold 6238 Cascade Lake @ 2.10GHz | 879GB SATA SSD | - |
| 1 | 128 | 2000G or 2048000M | 2 x AMD EPYC 7742 | 3.5TB SATA SSD | 8 x NVIDIA A100 Ampere |
| 2 | 32 | 256G or 262144M | 2 x Intel Xeon Gold 6326 Cascade Lake @ 2.90GHz | 3.5TB SATA SSD | 4 x NVIDIA A100 Ampere |
| 11 | 64 | 128G or 131072M | 1 x AMD EPYC 7713 | 1.8TB SATA SSD | 4 x NVIDIA RTX A5000 Ampere |
| 6 | 32 | 1024G or 1048576M | 1 x AMD EPYC 7543 | 8x2TB NVMe | - |

Most applications will work with Broadwell, Skylake, or Cascade Lake nodes, and the performance differences should be minimal compared to waiting times. We therefore recommend not selecting a particular node type for your jobs. If necessary, for jobs that must be run with a Cascade Lake CPU, use `--constraint=cascade` (see [how to specify CPU architecture](link-to-cpu-arch)).

For local storage on the node, it is recommended to use the temporary directory `$SLURM_TMPDIR` generated by Slurm. This directory and its contents are deleted at the end of job execution.

Note that the amount of available memory is less than the rounded value suggested by the hardware configuration. For example, 128G base type nodes actually have 128GB of RAM, but a certain amount is permanently used by the kernel and the operating system. To avoid the time loss incurred by swapping or paging, the scheduler will never allocate a job whose requirements exceed the amount of available memory indicated in the table above.

Also note that the memory allocated for the job must be sufficient for the buffer reads and writes performed by the kernel and the file system; when these operations are numerous, it is preferable to request more memory than the total amount required by the processes.


## GPU

The three generations of Tesla GPUs, from oldest to newest, are:

* V100 Volta (two nodes with NVLINK interconnect)
* Turing T4 GPUs
* A100 Ampere

P100 GPUs are no longer in service. V100s are their successors and offer twice the performance for standard calculations and eight times the performance for deep learning calculations that can use its Tensor core units. The newer T4 card is suitable for deep learning tasks; it is not efficient for double-precision calculations, but its single-precision performance is good; it also has Tensor cores and can handle reduced-precision calculations with integers.


### Pascal GPU Nodes

No longer in service.


### Volta GPU Nodes

Graham has a total of two Volta GPU nodes that have a high-bandwidth NVLINK interconnect.

The nodes are available to all users for maximum execution times of 7 days.

In the following example, the script submits a job for one of the 8-GPU nodes. The `module load` command ensures that the modules compiled for the Skylake architecture are used. Replace `nvidia-smi` with the command you want to run.

**IMPORTANT:** Determine the number of CPUs to request by applying a CPU/GPU ratio of 3.5 or less on 28-core nodes. For example, if your task needs to use 4 GPUs, you should request at most 14 CPU cores, and to use 1 GPU, request at most 3 CPU cores. Users can run a few test jobs of less than an hour to find out the performance level of the code.

The two newest Volta nodes have 40 cores and the number of cores per GPU requested must be adjusted upwards as needed; a task can, for example, use 5 CPU cores per GPU. These nodes are also interconnected. If you want to use one of these nodes, you must add the `constraint=cascade,v100` parameter to the job submission script.

**Example with a single GPU**

**File:** gpu_single_GPU_job.sh

```bash
#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=12G
#SBATCH --time=1-00:00
module load arch/avx512 StdEnv/2018.3
nvidia-smi
```

**Example with an entire node**

**File:** gpu_single_node_job.sh

```bash
#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:8
#SBATCH --exclusive
#SBATCH --cpus-per-task=28
#SBATCH --mem=150G
#SBATCH --time=1-00:00
module load StdEnv/2023
nvidia-smi
```

Graham's Volta nodes have a fast local disk that should be used if the task requires a lot of I/O operations. In the task, the environment variable `$SLURM_TMPDIR` gives the location of the temporary directory on the disk. You can copy your data files there at the beginning of the script before running the program, and copy your output files there at the end of the script. Since all files contained in `$SLURM_TMPDIR` are deleted when the task is finished, you don't have to do this. You can even [create Python virtual environments](link-to-virtual-env) in this temporary space to improve efficiency.


### Turing GPU Nodes

These nodes are used like Volta nodes, except that you should request them by specifying `--gres=gpu:t4:2`. In this example, two T4 cards per node are requested.


### Ampere GPU Nodes

The use of these nodes is similar to that of Volta nodes, except that to obtain them, you must specify `--gres=gpu:a100:2` or `--gres=gpu:a5000:2`. In this example, two Ampere cards per node are requested.


## Capacity Reduction

As of January 13, 2025, Graham's capacity will be limited to approximately 25% until Nibi is available.


**(Retrieved from "https://docs.alliancecan.ca/mediawiki/index.php?title=Graham/fr&oldid=175726")**
