# Niagara Quickstart

This page is a translated version of the page [Niagara Quickstart](https://docs.alliancecan.ca/mediawiki/index.php?title=Niagara_Quickstart&oldid=176254) and the translation is 48% complete. Outdated translations are marked like this.

Other languages: [English](https://docs.alliancecan.ca/mediawiki/index.php?title=Niagara_Quickstart&oldid=176254), français

## Cluster Characteristics

The Niagara cluster consists of 1548 Lenovo SD350 servers, each with 40 Skylake cores at 2.4GHz.  Its peak performance is 3.02 petaflops (4.75 theoretically). In June 2018, the cluster ranked 53rd in the Top 500 supercomputers.

Each cluster node has 188GB/202GB of memory, with a minimum of 4GB per core. The cluster is designed for intensive parallel tasks with an InfiniBand EDR (Enhanced Data Rate) network with Dragonfly+ topology and dynamic routing. Access to compute nodes is via a queuing system that executes tasks lasting at least 15 minutes and at most 12 or 24 hours, prioritizing intensive tasks.

Watch the [introductory video to Niagara](link-to-video).

For more information on hardware specifications, see the [Niagara page](link-to-niagara-page).


## Getting Started on Niagara

If you are a new SciNet user and belong to a group whose principal investigator does not have resources allocated by the resource allocation competition, you must [obtain a SciNet account](link-to-account-obtaining).

Please read this document carefully. The [FAQ](link-to-faq) is also a useful resource. If at any time you require assistance, or if something is unclear, please do not hesitate to [contact us](link-to-contact).


## Connecting

**Via your browser with Open OnDemand.** This is recommended for users who are not familiar with Linux or the command line. Please see our [quickstart guide](link-to-ood-guide) for more instructions on how to use Open OnDemand.

**Terminal access with ssh.** Please read the following instructions.

Niagara runs CentOS 7, which is a type of Linux. You will need to be familiar with Linux systems to work on Niagara. If you are not, it will be worth your time to review the [Linux introduction](link-to-linux-intro) or to attend a local "Linux Shell" workshop.

As with all Compute Canada and SciNet clusters, you can only connect via SSH (secure shell).

First open a terminal window (e.g., `PuTTY` on Windows or `MobaXTerm`), then connect via SSH to the login nodes with your Compute Canada credentials:

```bash
$ ssh -i /path/to/ssh_private_key -Y MYCCUSERNAME@niagara.scinet.utoronto.ca
```

or

```bash
$ ssh -i /path/to/ssh_private_key -Y MYCCUSERNAME@niagara.computecanada.ca
```

Tasks are created, edited, compiled, prepared, and submitted on the login nodes.  These login nodes are not part of the Niagara cluster, but they have the same architecture and software stack as the compute nodes.

In the above commands, `-Y` is optional, but necessary to open command-line windows on your local server.

To use the compute nodes, you must submit batch jobs to the scheduler.

If you cannot connect, first check the [cluster status](link-to-cluster-status).


## Locating Your Directories

### `/home` and `/scratch` Directories

To locate your `/home` and `/scratch` spaces, use:

```bash
$HOME=/home/g/groupname/myccusername
$SCRATCH=/scratch/g/groupname/myccusername
```

For example:

```bash
nia-login07:~$ pwd
/home/s/scinet/rzon
nia-login07:~$ cd $SCRATCH
nia-login07:rzon$ pwd
/scratch/s/scinet/rzon
```

**NOTE:** `home` is read-only on compute nodes.


### `/project` and `/archive` Spaces

Users with resources allocated by the 2018 competition can locate their project directory with:

```bash
$PROJECT=/project/g/groupname/myccusername
$ARCHIVE=/archive/g/groupname/myccusername
```

**NOTE:** Archive space is currently only available on HPSS.

**IMPORTANT: Preventive Measure**

Since paths may change, use environment variables (HOME, SCRATCH, PROJECT, ARCHIVE) instead.


### Storage and Quotas

You should familiarize yourself with the [various file systems](link-to-filesystems), what purpose they serve, and how to properly use them. This table summarizes the various file systems. See the [Data management at Niagara](link-to-datamanagement) page for more details.

| quota             | block size | duration | backup | on login nodes | on compute nodes |
|-----------------|-------------|-----------|--------|-----------------|-------------------|
| `$HOME`          | 100 GB      | 1 MB      | yes    | yes             | read only         |
| `$SCRATCH`       | 25 TB       | 16 MB     | 2 months| yes             | yes                |
| `$PROJECT`       | per group   | 16 MB     | yes    | yes             | yes                |
| `$ARCHIVE`       | per group   | 2 copies  | no     | no              | no                 |
| `$BBUFFER`       | ?           | 1 MB      | very short | no              | ?                  |


### Moving Data to Niagara

If you need to move data to Niagara for analysis, or when you need to move data off of Niagara, use the following guidelines:

* If your data is less than 10GB, move the data using the login nodes.
* If your data is greater than 10GB, move the data using the datamover nodes `nia-datamover1.scinet.utoronto.ca` and `nia-datamover2.scinet.utoronto.ca`.

Details of how to use the datamover nodes can be found on the [Data management at Niagara](link-to-datamanagement) page.


## Loading Modules

You have two options for running code on Niagara: use existing software, or [compile your own](link-to-compilation). This section focuses on the former.

Apart from essential software, applications are installed via modules. Modules configure environment variables (`PATH`, etc.). This makes available several incompatible versions of the same package. To find out which software is available, use `module spider`.

Common module subcommands are:

* `module load <module-name>`: use particular software
* `module purge`: remove currently loaded modules
* `module spider` (or `module spider <module-name>`): list available software packages
* `module avail`: list loadable software packages
* `module list`: list loaded modules

Along with modifying common environment variables, such as `PATH`, and `LD_LIBRARY_PATH`, these modules also create a `SCINET_MODULENAME_ROOT` environment variable, which can be used to access commonly needed software directories, such as `/include` and `/lib`.

There are handy abbreviations for the module commands. `ml` is the same as `module list`, and `ml <module-name>` is the same as `module load <module-name>`.


### Software Stacks: NiaEnv and CCEnv

There are actually two software environments on Niagara:

The Niagara software stack is specifically adapted to this cluster. It is available by default, but if needed can be loaded again with `module load NiaEnv`.

The usual software stack of general-purpose clusters (`Graham` and `Cedar`), compiled for a previous generation of CPUs. `module load CCEnv`

To load default modules like those of Cedar or Graham, also run `module load StdEnv`.


### Tips for Loading Modules

It is not advisable to change modules in your Niagara `.bashrc`. In some cases, the behavior can be very strange. If necessary, load the modules manually or with a separate script and load modules required for execution via your task submission script.

See information on the [default .bashrc and .bash_profile files](link-to-bashrc).

Instead, load modules by hand when needed, or by sourcing a separate script. Load run-specific modules inside your job submission script.

Short names are for default versions; for example, `intel` → `intel/2018.2`. It is usually preferable to specify the version to be able to reproduce a case.

Some modules require the prior loading of other modules. To resolve dependencies, use `module spider`.


## Compilers and Interpreters

For most compiled software, one should use the Intel compilers (`icc` for C, `icpc` for C++, and `ifort` for Fortran). Loading an `intel` module makes these available.

The GNU compiler suite (`gcc`, `g++`, `gfortran`) is also available, if you load one of the `gcc` modules.

Open source interpreted, interactive software is also available:

* Python
* R
* Julia
* Octave

Please visit the [Python](link-to-python) or [R](link-to-r) page for details on using these tools. For information on running MATLAB applications on Niagara, visit [this page](link-to-matlab).


## Commercial Applications

May I use commercial software on Niagara?

You may need to provide your own license.

SciNet and Compute Canada serve thousands of users from various disciplines; it is not possible to accommodate everyone's preferred applications.

Thus, the only commercial applications installed on Niagara are general-purpose, namely compilers, mathematical libraries, and debugging tools. This excludes Matlab, Gaussian, IDL.

Open source options are available, such as Octave, Python, and R.

We will help you install any commercial application for which you have a license. In some cases, if you have a license, you can use applications from the Compute Canada software stack.


## Compilation Example

We want to compile an application from the two source files `main.c` and `module.c` that use GSL (Gnu Scientific Library). We could proceed as follows:

```bash
nia-login07:~$ module list
Currently Loaded Modules:
1) NiaEnv/2018a (S)
Where: S: Module is Sticky, requires --force to unload or purge

nia-login07:~$ module load intel/2018.2 gsl/2.4

nia-login07:~$ ls
appl.c module.c

nia-login07:~$ icc -c -O3 -xHost -o appl.o appl.c
nia-login07:~$ icc -c -O3 -xHost -o module.o module.c
nia-login07:~$ icc -o appl module.o appl.o -lgsl -mkl

nia-login07:~$ ./appl
```

**Note:** The optimization flags `-O3 -xHost` allow the Intel compiler to use instructions specific to the existing CPU architecture (rather than for more generic x86_64 CPUs).

It is easy to link with this library when using the Intel compiler; only the `-mkl` flags are needed.

To compile with gcc, the optimization flags would be `-O3 -march=native`. To link with MKL, we suggest the [MKL link line advisor](link-to-mkl-advisor).


## Testing

You should always test your code before submitting a job to know if it is valid and to know the resources you need.

Short test jobs can be run on the login nodes. In principle: a few minutes, using at most 1-2GB of memory, a few cores.

After `module load ddt`, you can launch the `ddt` debugger on the login nodes.

Short tests that cannot be run on a login node or that require a dedicated node require interactive debugging with the `salloc` command.

```bash
nia-login07:~$ salloc -pdebug --nodes N --time=1:00:00
```

where N is the number of nodes. The interactive debugging session should not exceed one hour, should not use more than 4 cores, and each user should only have one debugging session at a time.

Another option is to use the command:

```bash
nia-login07:~$ debugjob N
```

where N is the number of nodes. If N=1, the interactive session is one hour and if N=4 (maximum value) the session is 30 minutes.


## Submitting Jobs

Niagara uses the Slurm scheduler.

From a login node, jobs are submitted by passing a script to the `sbatch` command:

```bash
nia-login07:~$ sbatch jobscript.sh
```

This places the job in the queue; it will be executed on the compute nodes in its turn.

Jobs will be counted against the Resource Allocation for research groups; if the group has not received any of these resources, the job will be counted against the Fast Access Service (formerly default allocation).

Remember:

* Scheduling is done by node, so in multiples of 40 cores.
* The real-time limit should not exceed 24 hours; for users without allocation, the limit is 12 hours.
* Writing must be done in your `scratch` or `project` directory (on compute nodes, `home` is read-only).
* Compute nodes cannot access the internet.
* Before starting, download the data to a login node.


### Node Scheduling

All resource requests for jobs are scheduled in multiples of nodes.

The nodes used by your jobs are for your exclusive use. No other user has access to them. You can access the tasks with SSH to monitor them.

Regardless of your request, the scheduler translates it into multiples of nodes allocated to the task. It is useless to request an amount of memory. Your task always gets Nx202GB of RAM, where N represents the number of nodes.

You should try to use all the cores of the nodes allocated to your task. Since there are 40 cores per node, your task should use Nx40 cores. If this is not the case, we will contact you to help you optimize your work.


### Limits

There are limits to the size and duration of your jobs, the number of jobs you can run and the number of jobs you can have queued. It matters whether a user is part of a group with a [Resources for Research Group allocation](link-to-resources) or not. It also matters in which 'partition' the jobs runs. 'Partitions' are SLURM-speak for use cases. You specify the partition with the `-p` parameter to `sbatch` or `salloc`, but if you do not specify one, your job will run in the `compute` partition, which is the most common case.

| Usage                               | Partition | Running jobs | Submitted jobs (incl. running) | Min. size of jobs | Max. size of jobs | Min. walltime | Max. walltime |
|------------------------------------|-----------|---------------|---------------------------------|--------------------|--------------------|----------------|----------------|
| Compute jobs with an allocation     | compute    | 50            | 1000                             | 1 node (40 cores) | 1000 nodes (40000 cores) | 15 minutes     | 24 hours       |
| Compute jobs without allocation     | compute    | 50            | 200                              | 1 node (40 cores) | 20 nodes (800 cores)    | 15 minutes     | 12 hours       |
| Testing or troubleshooting         | debug      | 1             | 1                                | 1 node (40 cores) | 4 nodes (160 cores)     | N/A            | 1 hour         |
| Archiving or retrieving data in HPSS | archivelong | 2 per user    | 10 per user                       | N/A                | N/A                | 15 minutes     | 72 hours       |
| Inspecting archived data           | archiveshort| 2 per user    | 10 per user                       | N/A                | N/A                | 15 minutes     | 1 hour         |

Within these limits, jobs will still have to wait in the queue. The waiting time depends on many factors such as the allocation amount, how much allocation was used in the recent past, the number of nodes and the walltime, and how many other jobs are waiting in the queue.


### File Input/Output Tips

It is important to understand the file systems, so as to perform your file I/O (Input/Output) responsibly. Refer to the [Data management at Niagara](link-to-datamanagement) page for details about the file systems.

Your files can be seen on all Niagara login and compute nodes. `$HOME`, `$SCRATCH`, and `$PROJECT` all use the parallel file system called GPFS.

GPFS is a high-performance file system which provides rapid reads and writes to large data sets in parallel from many nodes. Accessing data sets which consist of many, small files leads to poor performance on GPFS.

Avoid reading and writing lots of small amounts of data to disk. Many small files on the system waste space and are slower to access, read and write. If you must write many small files, use `ramdisk`.

Write data out in a binary format. This is faster and takes less space.

The Burst Buffer is better for i/o heavy jobs and to speed up checkpoints.


### Example of an MPI Submission Script

To run the MPI application named `appl_mpi_ex` with 320 processes, the script would be:

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=80
#SBATCH --time=1:00:00
#SBATCH --job-name mpi_job
#SBATCH --output=mpi_output_%j.txt
#SBATCH --mail-type=FAIL
cd $SLURM_SUBMIT_DIR
module load intel/2018.2
module load openmpi/3.1.0
mpirun ./mpi_example
# or "srun ./mpi_example"
```

Submit the script (named here `mpi_ex.sh`) with the command:

```bash
nia-login07:~$ sbatch mpi_ex.sh
```

The first line indicates that this is a bash script. Lines starting with `#SBATCH` go to SLURM. `sbatch` reads these lines as a job request (which it gives the name `mpi_job`).

In this case, SLURM looks for 2 nodes (each of which will have 40 cores) on which to run a total of 80 tasks, for 1 hour. (Instead of specifying `--ntasks=80`, you can also ask for `--ntasks-per-node=40`, which amounts to the same.) Note that the mpifun flag `--ppn` (processors per node) is ignored.

Once it found such a node, it runs the script:

* Change to the submission directory;
* Loads modules;
* Runs the `mpi_example` application (SLURM will inform mpirun or srun on how many processes to run).

To use hyperthreading, just change `--ntasks=80` to `--ntasks=160`, and add `--bind-to none` to the `mpirun` command (the latter is necessary for OpenMPI only, not when using IntelMPI).


### Example of an OpenMP Submission Script

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=1:00:00
#SBATCH --job-name openmp_job
#SBATCH --output=openmp_output_%j.txt
#SBATCH --mail-type=FAIL
cd $SLURM_SUBMIT_DIR
module load intel/2018.2
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
./openmp_example
# or "srun ./openmp_example".
```

Submit the script (named `openmp_ex.sh`) with the command:

```bash
nia-login07:~$ sbatch openmp_ex.sh
```

The first line indicates that this is a bash script. Lines starting with `#SBATCH` go to the scheduler. `sbatch` interprets these lines as a request and names it `openmp_ex`.

The scheduler then looks for a 40-core node to run in a task, for a duration of one hour.

Once the node is found, the script is executed:

* Change to the submission directory;
* Load modules (must also be done in the submission script on Niagara);
* Configure an environment variable to specify 40 threads (there is no hyperthreading in this example);
* Run the `appl_openmp_ex` application.

To use hyperthreading, replace `--cpus-per-task=40` with `--cpus-per-task=80`.


### Monitoring Pending Tasks

Once the task is placed in the queue, follow its progress with the following commands:

* `squeue` or `qsum` to see the tasks in the queue (`squeue -u $USER` for your own tasks);
* `squeue -j JOBID` for information on a particular task (or the longer version `scontrol show job JOBID`);
* `squeue --start -j JOBID` for an estimate of when the task will be executed (the result is not always reliable);

Since this is not very precise, you may want to know where your task is in the queue with the following bash function:

For more information, see [Running Tasks](link-to-running-tasks).


## Visualization

To learn how to use Niagara's visualization tools, see [this SciNet wiki page](link-to-visualization).


## For More Information

**Sites:**

* SciNet: https://www.scinet.utoronto.ca
* Niagara: [Niagara wiki page](link-to-niagara-page)
* System Status: https://docs.scinet.utoronto.ca/index.php/Main_Page
* Training: https://support.scinet.utoronto.ca/education

**Support:** Contact our [technical support](link-to-support)


**(Remember to replace the bracketed placeholders like `link-to-video` with actual links.)**
