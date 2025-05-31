# Advanced MPI Scheduling

Most of the time, you should submit your distributed-memory parallel MPI jobs as shown in the [MPI Job](https://docs.alliancecan.ca/mediawiki/index.php?title=Running_jobs/fr) section of the [Running Jobs](https://docs.alliancecan.ca/mediawiki/index.php?title=Running_jobs/fr) page.  Simply use `-ntasks` or `-n` to specify the number of processes and let the scheduler do its best allocation, given the cluster efficiency.

However, if you need more control over the allocation, consult SchedMD's [Support for Multi-core/Multi-thread Architectures](https://slurm.schedmd.com/multi_core.html) page, which describes how various `sbatch` command options affect process scheduling.  The Slurm FAQ answer to "[What exactly is considered a CPU?](https://slurm.schedmd.com/faq.html#what_is_a_cpu)" may also be helpful.


## Example Scenarios

### Few Cores, Unconstrained Nodes

In addition to specifying the duration of any Slurm job, you must indicate the number of MPI processes that Slurm should start. The simplest way to do this is to use `--ntasks`. Since the default memory allocation of 256MB is often insufficient, you should also specify the amount of memory required. With `--ntasks`, it is impossible to know how many cores will be on each node; you will then want to use `--mem-per-cpu` as well.

```bash
# File: basic_mpi_job.sh
#!/bin/bash
#SBATCH --ntasks=15
#SBATCH --mem-per-cpu=3G
srun application.exe
```

We have 15 MPI processes here. Core allocation could be done on 1 node, 15 nodes, or any number of nodes between 1 and 15.


### Whole Nodes

For a computationally intensive parallel job that can efficiently use 32 cores or more, you should probably request whole nodes; it is therefore useful to know what types of nodes are available on the cluster you are using.

Most nodes on Beluga, Cedar, Graham, Narval, and Niagara are configured as follows:

| Cluster | Cores | Usable Memory | Notes |
|---|---|---|---|
| Beluga | 40 | 186 GiB (~4.6 GiB/core) | Some are reserved for whole-node jobs. |
| Graham | 32 | 125 GiB (~3.9 GiB/core) | Some are reserved for whole-node jobs. |
| Cedar (Skylake) | 48 | 187 GiB (~3.9 GiB/core) | Some are reserved for whole-node jobs. |
| Narval | 64 | 249 GiB (~3.9 GiB/core) | AMD EPYC Rome processors |
| Niagara | 40 | 188 GiB | This cluster only accepts whole-node jobs. |

Whole-node jobs can be run on all nodes. In the table above, the note "Some are reserved for whole-node jobs" means that per-core jobs are prohibited on some nodes.

Here are examples of scripts requesting whole nodes:

```bash
# File: whole_nodes_beluga.sh
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=40
#SBATCH --mem=0
srun application.exe
```

```bash
# File: whole_nodes_cedar.sh
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=48
#SBATCH --mem=0
srun application.exe
```

```bash
# File: whole_nodes_graham.sh
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --mem=0
srun application.exe
```

```bash
# File: whole_nodes_narval.sh
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=64
#SBATCH --mem=0
srun application.exe
```

```bash
# File: whole_nodes_niagara.sh
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=40  # or 80 (Hyper-Threading enabled)
#SBATCH --mem=0
srun application.exe
```

Requesting `--mem=0` tells Slurm to reserve all available memory of each node assigned to the job.

However, if you need more memory per node than the smallest node can offer (e.g., more than 125GB on Graham), you should not use `--mem=0`, but request an explicit amount of memory.  Also, some memory on each node is reserved for the operating system; in the Node Characteristics section, the Available Memory column indicates the largest amount of memory a job can request:

| Cluster | Available Memory |
|---|---|
| Beluga |  |
| Cedar |  |
| Graham |  |
| Narval |  |


### Few Cores, Single Node

If you need less than a whole node, but all cores must be from the same node, you can request, for example:

```bash
# File: less_than_whole_node.sh
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=15
#SBATCH --mem=45G
srun application.exe
```

You could also use `--mem-per-cpu=3G`, but the job would be cancelled if one of the processes exceeds 3GB. The advantage with `--mem=45G` is that the memory consumed by each process does not matter, provided that overall they do not exceed 45GB.


### Computationally Intensive Parallel Job, Not Multiples of Whole Nodes

Not all jobs perform optimally on cores in multiples of 32, 40, or 48. Specifying a precise number of cores or not can affect the *execution time* (or good resource utilization) or the *waiting time* (or good use of your allotted time). For help on how to assess these factors, contact [technical support](mailto:support@computecanada.ca).


## Hybrid Jobs: MPI with OpenMP or MPI with Threads

It is important to understand that the number of Slurm *tasks* requested represents the number of *processes* that will be started with `srun`. In the case of a hybrid job that uses both MPI processes and OpenMP or Posix threads, you will want to indicate the number of MPI processes with `--ntasks` or `-ntasks-per-node` and the number of threads with `--cpus-per-task`.

```bash
#SBATCH --ntasks=16 --cpus-per-task=4 --mem-per-cpu=3G
srun --cpus-per-task=$SLURM_CPUS_PER_TASK application.exe
```

Here, 64 cores are allocated, but only 16 MPI processes (tasks) are and will be initialized. If it is also an OpenMP application, each process will start 4 threads, one per core. Each process will be able to use 12GB. With 4 cores, tasks could be allocated on 2 to 16 nodes, regardless of which ones. You must also specify `--cpus-per-task=$SLURM_CPUS_PER_TASK` for `srun`, as this has been required since Slurm 22.05 and does not harm older versions.

```bash
#SBATCH --nodes=2 --ntasks-per-node=8 --cpus-per-task=4 --mem=96G
srun --cpus-per-task=$SLURM_CPUS_PER_TASK application.exe
```

The size of this job is identical to the previous one, i.e., 16 tasks (or 16 MPI processes) each with 4 threads. The difference here is that we will get exactly 2 whole nodes. Remember that `--mem` specifies the amount of memory *per node* and we prefer it to `--mem-per-cpu` for the reason given above.


## Why `srun` Rather Than `mpiexec` or `mpirun`?

`mpirun` allows communication between processes running on different computers; recent schedulers have this same functionality. With Torque/Moab, it is not necessary to provide `mpirun` with the list of nodes or the number of processes since the scheduler takes care of it. With Slurm, it is the scheduler that decides on the affinity of the tasks, which avoids having to specify parameters such as `mpirun --map-by node:pe=4 -n 16 application.exe`.

In the previous examples, it is understood that `srun application.exe` automatically distributes the processes to the precise resources allocated to the task.

In terms of levels of abstraction, `srun` is above `mpirun`; `srun` can do everything `mpirun` does and more. With Slurm, `srun` is the preferred tool for launching all types of calculations; it is also more versatile than Torque's `pbsdsh`.  `srun` could be described as the universal Slurm tool for distributing parallel tasks; once the resources are allocated, the nature of the application does not matter, whether it is MPI, OpenMP, hybrid, serial distribution, pipeline, multiprogram, or other.

Of course, `srun` is perfectly suited to Slurm: it initiates the first step of the task, correctly initializes the environment variables `SLURM_STEP_ID` and `SLURM_PROCID`, and provides the appropriate monitoring information.

For examples of some differences between `srun` and `mpiexec`, see the [OpenMPI forum](https://www.open-mpi.org/). In some cases, `mpiexec` will offer better performance than `srun`, but `srun` reduces the risk of conflict between resources allocated by Slurm and those used by OpenMPI.


## References

*   [sbatch documentation](https://slurm.schedmd.com/sbatch.html)
*   [srun documentation](https://slurm.schedmd.com/srun.html)
*   [Open MPI and Slurm](https://www.open-mpi.org/faq/?category=running)

Category:SLURM
