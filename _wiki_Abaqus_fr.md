# Abaqus FEA

Abaqus FEA is a commercial finite element analysis and computer-aided engineering software package.

## Your License

Abaqus modules are available on our clusters, but you must possess your own license. To set up your account on the clusters you wish to use, log in and create a file in each one:

```bash
$HOME/.licenses/abaqus.lic
```

This file should contain the following two lines, for versions 202X and 6.14.1 respectively. Replace `port@server` with the flexlm port number and IP address (or fully qualified domain name) of your Abaqus license server.

**File: abaqus.lic**

```
prepend_path (
"ABAQUSLM_LICENSE_FILE",
"port@server"
)
prepend_path (
"LM_LICENSE_FILE",
"port@server"
)
```

If your license is not configured for a particular cluster, system administrators on both sides will need to make some modifications. This is necessary so that the flexlm and TCP ports of your Abaqus server can be reached by all compute nodes when your queued jobs are executed. To assist with this, contact technical support, providing:

*   The flexlm port number
*   The static port number
*   The IP address of your Abaqus license server

In return, you will receive a list of IP addresses, and your system administrator can open the firewalls on your local server to allow the cluster to connect via both ports. A special agreement usually needs to be negotiated and signed with SIMULIA for such a license to be used remotely with our hardware.


## Submitting a Job

Below are prototype Slurm scripts for submitting parallel simulations on one or multiple compute nodes using threads and MPI. In most cases, using one of the scripts from the `/project` directory in either the single-node sections will suffice. In the last line of the scripts, the `memory=` argument is optional and is for memory-intensive or problematic jobs; the 3072MB offset value might require adjustment. To obtain the list of command-line arguments, load an Abaqus module and run `abaqus -help | less`.

For a single-node job lasting less than 24 hours, the script from the `/project` directory under the first tab should suffice. For longer jobs, use a restart script.

It is preferable that jobs creating large restart files write to local disk via the use of the `SLURM_TMPDIR` environment variable used in the temporary directory scripts under the two rightmost tabs of the single-node standard and explicit analysis sections. The restart scripts presented here will continue jobs that were prematurely interrupted for any reason. Such interruptions may occur if a job reaches its requested maximum runtime before completion and is stopped by the queue, or if the compute node on which the job was running crashed due to unexpected hardware failure. Other types of restarts are possible by further modifying the input file (not shown) to continue a job with additional steps or modify the analysis (consult the documentation for version-specific details).

Jobs requiring significant memory or compute resources (beyond the capacity of a single node) should use the MPI scripts in the multiple-node sections to distribute the computation across an arbitrary set of nodes automatically determined by the scheduler. Before launching long-running jobs, it is recommended to run short scalability tests to determine the actual runtime (and memory requirements) as a function of the optimal number of cores (2, 4, 8, etc.).


### Standard Analysis

Solvers support both thread and MPI parallelization. Scripts for each mode are presented under tabs for single-node and multiple-node usage. Restart scripts for a multiple-node job are not presented at this time.


#### Single-Node Scripts

*   `/project` directory script
*   `/project` directory restart script
*   Temporary directory script
*   Temporary directory restart script


**File: "scriptsp1.txt"**

```bash
#!/bin/bash
#SBATCH --account=def-group    # Specify account
#SBATCH --time=00-06:00        # Specify days-hrs:mins
#SBATCH --cpus-per-task=4      # Specify number of cores
#SBATCH --mem=8G               # Specify total memory > 5G
#SBATCH --nodes=1              # Do not change !
##SBATCH --constraint=cascade  # Uncomment to specify node (cpu/gpu jobs)
##SBATCH --gres=gpu:t4:1       # Uncomment to specify gpu
# or
##SBATCH --constraint=rome     # Uncomment to specify node (gpu only jobs)
##SBATCH --gres=gpu:a100:1     # Uncomment to specify gpu
module load StdEnv/2020
# Latest installed version
module load abaqus/2021
# Latest installed version
#module load StdEnv/2016       # Uncomment to use
#module load abaqus/2020       # Uncomment to use
unset SLURM_GTIDS
export MPI_IC_ORDER='tcp'
echo "LM_LICENSE_FILE= $LM_LICENSE_FILE"
echo "ABAQUSLM_LICENSE_FILE= $ABAQUSLM_LICENSE_FILE"
rm -f testsp1* testsp2*
abaqus job=testsp1 input=mystd-sim.inp \
scratch=$SLURM_TMPDIR cpus=$SLURM_CPUS_ON_NODE interactive \
mp_mode=threads memory="$(( ${SLURM_MEM_PER_NODE} - 3072 ))MB" \
#  gpus=$SLURM_GPUS_ON_NODE  # uncomment this line to use gpu
```

To write restart data in increments of N=12, the input file should contain:

```
*RESTART, WRITE, OVERLAY, FREQUENCY=12
```

To write restart data for a total of 12 increments, instead enter:

```
*RESTART, WRITE, OVERLAY, NUMBER INTERVAL=12, TIME MARKS=NO
```

To check complete restart information:

```bash
egrep -i "step|start" testsp*.com testsp*.msg testsp*.sta
```

Some simulations may be improved by adding the Abaqus command `order_parallel=OFF` to the bottom of the script.


**File: "scriptsp2.txt"**

```bash
#!/bin/bash
#SBATCH --account=def-group    # Specify account
#SBATCH --time=00-06:00        # Specify days-hrs:mins
#SBATCH --cpus-per-task=4      # Specify number of cores
#SBATCH --mem=8G               # Specify total memory > 5G
#SBATCH --nodes=1              # Do not change !
##SBATCH --constraint=cascade  # Uncomment to specify node (44cores)
##SBATCH --gres=gpu:t4:1       # Uncomment to specify gpu
# or
##SBATCH --constraint=rome     # Uncomment to specify node (128cores)
##SBATCH --gres=gpu:a100:1     # Uncomment to specify gpu
module load StdEnv/2020
# Latest installed version
module load abaqus/2021
# Latest installed version
unset SLURM_GTIDS
export MPI_IC_ORDER='tcp'
echo "LM_LICENSE_FILE= $LM_LICENSE_FILE"
echo "ABAQUSLM_LICENSE_FILE= $ABAQUSLM_LICENSE_FILE"
rm -f testsp2* testsp1.lck
abaqus job=testsp2 oldjob=testsp1 input=mystd-sim-restart.inp \
scratch=$SLURM_TMPDIR cpus=$SLURM_CPUS_ON_NODE interactive \
mp_mode=threads memory="$(( ${SLURM_MEM_PER_NODE} - 3072 ))MB" \
#  gpus=$SLURM_GPUS_ON_NODE  # uncomment this line to use gpu
```

The input file for restart should contain:

```
*HEADING
*RESTART, READ
```


**File: "scriptst1.txt"**

```bash
#!/bin/bash
#SBATCH --account=def-group    # Specify account
#SBATCH --time=00-06:00        # Specify days-hrs:mins
#SBATCH --cpus-per-task=4      # Specify number of cores
#SBATCH --mem=8G               # Specify total memory > 5G
#SBATCH --nodes=1              # Do not change !
##SBATCH --constraint=cascade  # Uncomment to specify node (cpu/gpu jobs)
##SBATCH --gres=gpu:t4:1       # Uncomment to specify gpu
# or
##SBATCH --constraint=rome     # Uncomment to specify node (gpu only jobs)
##SBATCH --gres=gpu:a100:1     # Uncomment to specify gpu
module load StdEnv/2020
# Latest installed version
module load abaqus/2021
# Latest installed version
unset SLURM_GTIDS
export MPI_IC_ORDER='tcp'
echo "LM_LICENSE_FILE= $LM_LICENSE_FILE"
echo "ABAQUSLM_LICENSE_FILE= $ABAQUSLM_LICENSE_FILE"
echo "SLURM_SUBMIT_DIR = "$SLURM_SUBMIT_DIR
echo "SLURM_TMPDIR = "$SLURM_TMPDIR
rm -f testst1* testst2*
mkdir $SLURM_TMPDIR/scratch
cd $SLURM_TMPDIR
while sleep 6h ; do
echo "Saving data due to time limit ..."
cp -fv * $SLURM_SUBMIT_DIR 2>/dev/null
done &
WPID=$!
abaqus job=testst1 input=$SLURM_SUBMIT_DIR/mystd-sim.inp \
scratch=$SLURM_TMPDIR/scratch cpus=$SLURM_CPUS_ON_NODE interactive \
mp_mode=threads memory="$(( ${SLURM_MEM_PER_NODE} - 3072 ))MB" \
#  gpus=$SLURM_GPUS_ON_NODE  # uncomment this line to use gpu
{ kill $WPID && wait $WPID ; } 2>/dev/null
cp -fv * $SLURM_SUBMIT_DIR
```

To write restart data in increments of N=12, the input file should contain:

```
*RESTART, WRITE, OVERLAY, FREQUENCY=12
```

To write restart data for a total of 12 increments, instead enter:

```
*RESTART, WRITE, OVERLAY, NUMBER INTERVAL=12, TIME MARKS=NO
```

To check complete restart information:

```bash
egrep -i "step|start" testst*.com testst*.msg testst*.sta
```


**File: "scriptst2.txt"**

```bash
#!/bin/bash
#SBATCH --account=def-group    # Specify account
#SBATCH --time=00-06:00        # Specify days-hrs:mins
#SBATCH --cpus-per-task=4      # Specify number of cores
#SBATCH --mem=8G               # Specify total memory > 5G
#SBATCH --nodes=1              # Do not change !
##SBATCH --constraint=cascade  # Uncomment to specify node (44 cores)
##SBATCH --gres=gpu:t4:1       # Uncomment to specify gpu
# or
##SBATCH --constraint=rome     # Uncomment to specify node (128 cores)
##SBATCH --gres=gpu:a100:1     # Uncomment to specify gpu
module load StdEnv/2020
# Latest installed version
module load abaqus/2021
# Latest installed version
unset SLURM_GTIDS
export MPI_IC_ORDER='tcp'
echo "LM_LICENSE_FILE= $LM_LICENSE_FILE"
echo "ABAQUSLM_LICENSE_FILE= $ABAQUSLM_LICENSE_FILE"
echo "SLURM_SUBMIT_DIR = "$SLURM_SUBMIT_DIR
echo "SLURM_TMPDIR = "$SLURM_TMPDIR
rm -f testst2* testst1.lck
cp testst1* $SLURM_TMPDIR
mkdir $SLURM_TMPDIR/scratch
cd $SLURM_TMPDIR
while sleep 6h ; do
echo "Saving data due to time limit ..."
cp -fv testst2* $SLURM_SUBMIT_DIR 2>/dev/null
done &
WPID=$!
abaqus job=testst2 oldjob=testst1 input=$SLURM_SUBMIT_DIR/mystd-sim-restart.inp \
scratch=$SLURM_TMPDIR/scratch cpus=$SLURM_CPUS_ON_NODE interactive \
mp_mode=threads memory="$(( ${SLURM_MEM_PER_NODE} - 3072 ))MB" \
#  gpus=$SLURM_GPUS_ON_NODE  # uncomment this line to use gpu
{ kill $WPID && wait $WPID ; } 2>/dev/null
cp -fv testst2* $SLURM_SUBMIT_DIR
```

The input file for restart should contain:

```
*HEADING
*RESTART, READ
```

#### Multiple-Node Script

If you have a license that allows you to run memory- and compute-intensive jobs, the following script will perform the computation with MPI using an arbitrary set of nodes ideally automatically determined by the scheduler. A template script for restarting multiple-node jobs is not provided as its usage presents additional limitations.


**File: "scriptsp1-mpi.txt"**

```bash
!/bin/bash
#SBATCH --account=def-group    # Specify account
#SBATCH --time=00-06:00        # Specify days-hrs:mins
##SBATCH --nodes=2             # Uncomment to specify (optional)
#SBATCH --ntasks=8             # Specify number of cores
#SBATCH --mem-per-cpu=4G       # Specify memory per core
##SBATCH --tasks-per-node=4    # Uncomment to specify (optional)
#SBATCH --cpus-per-task=1      # Do not change !
module load StdEnv/2020
# Latest installed version
module load abaqus/2021
# Latest installed version
unset SLURM_GTIDS
#export MPI_IC_ORDER='tcp'
echo "LM_LICENSE_FILE= $LM_LICENSE_FILE"
echo "ABAQUSLM_LICENSE_FILE= $ABAQUSLM_LICENSE_FILE"
rm -f testsp1-mpi*
unset hostlist
nodes="$( slurm_hl2hl.py --format MPIHOSTLIST | xargs )"
for i in `echo "$nodes" | xargs -n1 | uniq` ; do
hostlist="${hostlist}$(echo "['${i}',$(echo "$nodes" | xargs -n1 | grep $i | wc -l)],")";
done
hostlist="$(echo "$hostlist" | sed 's/,$//g')"
mphostlist="mp_host_list=[$(echo "$hostlist")]"
export $mphostlist
echo "$mphostlist" > abaqus_v6.env

abaqus job=testsp1-mpi input=mystd-sim.inp \
scratch=$SLURM_TMPDIR cpus=$SLURM_NTASKS interactive mp_mode=mpi \
#mp_host_split=1  # number of dmp processes per node >= 1 (uncomment to specify)
```

### Explicit Analysis

Solvers support both thread and MPI parallelization. Scripts for each mode are presented under tabs for single-node and multiple-node usage. Template scripts for restarting a multiple-node job require further testing and are not presented at this time.


#### Single-Node Scripts

*   `/project` directory script
*   `/project` directory restart script
*   Temporary directory script
*   Temporary directory restart script


**File: "scriptep1.txt"**

```bash
#!/bin/bash
#SBATCH --account=def-group    # indiquer le nom du compte
#SBATCH --time=00-06:00        # indiquer la limite de temps (jours-heures:minutes)
#SBATCH --mem=8000M            # indiquer la mémoire totale > 5M
#SBATCH --cpus-per-task=4      # indiquer le nombre de cœurs > 1
#SBATCH --nodes=1              # ne pas modifier
module load StdEnv/2020
module load abaqus/2021
unset SLURM_GTIDS
export MPI_IC_ORDER='tcp'
echo "LM_LICENSE_FILE= $LM_LICENSE_FILE"
echo "ABAQUSLM_LICENSE_FILE= $ABAQUSLM_LICENSE_FILE"
rm -f testep1* testep2*
abaqus job=testep1 input=myexp-sim.inp \
scratch=$SLURM_TMPDIR cpus=$SLURM_CPUS_ON_NODE interactive \
mp_mode=threads memory="$(( ${SLURM_MEM_PER_NODE} - 3072 ))MB"
```

To write restart data for a total of 12 increments, the input file should contain:

```
*RESTART, WRITE, OVERLAY, NUMBER INTERVAL=12, TIME MARKS=NO
```

To check complete restart information:

```bash
egrep -i "step|restart" testep*.com testep*.msg testep*.sta
```


**File: "scriptep2.txt"**

```bash
#!/bin/bash
#SBATCH --account=def-group    # indiquer le nom du compte
#SBATCH --time=00-06:00        # indiquer la limite de temps (jours-heures:minutes)
#SBATCH --mem=8000M            # indiquer la mémoire totale > 5M
#SBATCH --cpus-per-task=4      # indiquer le nombre de cœurs > 1
#SBATCH --nodes=1              # ne pas modifier
module load StdEnv/2020
module load abaqus/2021
unset SLURM_GTIDS
export MPI_IC_ORDER='tcp'
echo "LM_LICENSE_FILE= $LM_LICENSE_FILE"
echo "ABAQUSLM_LICENSE_FILE= $ABAQUSLM_LICENSE_FILE"
rm -f testep2* testep1.lck
for f in testep1* ; do
[[ -f ${f} ]] && cp -a "$f" "testep2${f#testep1}" ; done
abaqus job=testep2 input=myexp-sim.inp recover \
scratch=$SLURM_TMPDIR cpus=$SLURM_CPUS_ON_NODE interactive \
mp_mode=threads memory="$(( ${SLURM_MEM_PER_NODE} - 3072 ))MB"
```

The input file requires no modification for analysis restart.


**File: "scriptet1.txt"**

```bash
#!/bin/bash
#SBATCH --account=def-group    # specify account
#SBATCH --time=00-06:00        # days-hrs:mins
#SBATCH --mem=8000M            # node memory > 5G
#SBATCH --cpus-per-task=4      # number cores > 1
#SBATCH --nodes=1              # do not change
module load StdEnv/2020
module load abaqus/2021
unset SLURM_GTIDS
export MPI_IC_ORDER='tcp'
echo "LM_LICENSE_FILE= $LM_LICENSE_FILE"
echo "ABAQUSLM_LICENSE_FILE= $ABAQUSLM_LICENSE_FILE"
echo "SLURM_SUBMIT_DIR = "$SLURM_SUBMIT_DIR
echo "SLURM_TMPDIR = "$SLURM_TMPDIR
rm -f testet1* testet2*
cd $SLURM_TMPDIR
while sleep 6h ; do
cp -f * $SLURM_SUBMIT_DIR 2>/dev/null
done &
WPID=$!
abaqus job=testet1 input=$SLURM_SUBMIT_DIR/myexp-sim.inp \
scratch=$SLURM_TMPDIR cpus=$SLURM_CPUS_ON_NODE interactive \
mp_mode=threads memory="$(( ${SLURM_MEM_PER_NODE} - 3072 ))MB"
{ kill $WPID && wait $WPID ; } 2>/dev/null
cp -f * $SLURM_SUBMIT_DIR
```

To write restart data for a total of 12 increments, the input file should contain:

```
*RESTART, WRITE, OVERLAY, NUMBER INTERVAL=12, TIME MARKS=NO
```

To check complete restart information:

```bash
egrep -i "step|restart" testet*.com testet*.msg testet*.sta
```


**File: "scriptet2.txt"**

```bash
#!/bin/bash
#SBATCH --account=def-group    # specify account
#SBATCH --time=00-06:00        # days-hrs:mins
#SBATCH --mem=8000M            # node memory > 5G
#SBATCH --cpus-per-task=4      # number cores > 1
#SBATCH --nodes=1              # do not change
module load StdEnv/2020
module load abaqus/2021
unset SLURM_GTIDS
export MPI_IC_ORDER='tcp'
echo "LM_LICENSE_FILE= $LM_LICENSE_FILE"
echo "ABAQUSLM_LICENSE_FILE= $ABAQUSLM_LICENSE_FILE"
echo "SLURM_SUBMIT_DIR = "$SLURM_SUBMIT_DIR
echo "SLURM_TMPDIR = "$SLURM_TMPDIR
rm -f testet2* testet1.lck
for f in testet1* ; do
cp -a "$f" $SLURM_TMPDIR/"testet2${f#testet1}" ; done
cd $SLURM_TMPDIR
while sleep 3h ; do
cp -f * $SLURM_SUBMIT_DIR 2>/dev/null
done &
WPID=$!
abaqus job=testet2 input=$SLURM_SUBMIT_DIR/myexp-sim.inp recover \
scratch=$SLURM_TMPDIR cpus=$SLURM_CPUS_ON_NODE interactive \
mp_mode=threads memory="$(( ${SLURM_MEM_PER_NODE} - 3072 ))MB"
{ kill $WPID && wait $WPID ; } 2>/dev/null
cp -f * $SLURM_SUBMIT_DIR
```

The input file requires no modification for analysis restart.


#### Multiple-Node Script

**File: "scriptep1-mpi.txt"**

```bash
!/bin/bash
#SBATCH --account=def-group    # Specify account
#SBATCH --time=00-06:00        # Specify days-hrs:mins
#SBATCH --ntasks=8             # Specify number of cores
#SBATCH --mem-per-cpu=16000M   # Specify memory per core
# SBATCH --nodes=2             # Specify number of nodes (optional)
#SBATCH --cpus-per-task=1      # Do not change !
module load StdEnv/2020
# Latest installed version
module load abaqus/2021
# Latest installed version
unset SLURM_GTIDS
export MPI_IC_ORDER='tcp'
# uncomment next line when using abaqus/2021
export I_MPI_HYDRA_TOPOLIB=ipl
echo "LM_LICENSE_FILE= $LM_LICENSE_FILE"
echo "ABAQUSLM_LICENSE_FILE= $ABAQUSLM_LICENSE_FILE"
rm -f testep1-mpi*
unset hostlist
nodes="$( slurm_hl2hl.py --format MPIHOSTLIST | xargs )"
for i in `echo "$nodes" | xargs -n1 | uniq` ; do
hostlist="${hostlist}$(echo "['${i}',$(echo "$nodes" | xargs -n1 | grep $i | wc -l)],")";
done
hostlist="$(echo "$hostlist" | sed 's/,$//g')"
mphostlist="mp_host_list=[$(echo "$hostlist")]"
export $mphostlist
echo "$mphostlist" > abaqus_v6.env

abaqus job=testep1-mpi input=myexp-sim.inp \
scratch=$SLURM_TMPDIR cpus=$SLURM_NTASKS interactive mp_mode=mpi \
#mp_host_split=1  # number of dmp processes per node >= 1 (uncomment to specify)
```

## Estimating Memory Requirements

### Single Process

An estimate of the total node memory (`--mem=`) required by Slurm for a simulation to be performed solely in RAM (without being swapped to scratch disk) is found in the Abaqus output file `test.dat`. In the following example, the simulation requires a rather large amount of memory.

```
MEMORY ESTIMATE
PROCESS FLOATING PT MINIMUM MEMORY MEMORY TO
OPERATIONS REQUIRED MINIMIZE I/O
PER ITERATION (MB) (MB)
1 1.89E+14 3612 96345
```

Alternatively, the total memory estimate for a single-node threaded process can be obtained by running the simulation interactively on a compute node, then monitoring memory consumption using the `ps` or `top` commands. The following describes how to proceed in the latter case:

1.  Connect to a cluster via SSH, obtain an allocation on a compute node (such as `gra100`), and start your simulation with:

```bash
[name@server ~] $ module load StdEnv/2020
[name@server ~] $ module load abaqus/2021
[name@server ~] $ unset SLURM_GTIDS
```

2.  Via SSH, connect again to the cluster reserved by `sallco`, then to the compute node and launch `top`.

```bash
[name@server ~] $ ssh gra100
[name@server ~] $ top -u $USER
```

3.  Observe the `VIRT` and `RES` columns until stable maximum memory values are reached.

To fully satisfy the recommended value for `MEMORY TO OPERATIONS REQUIRED MINIMIZE I/O` (MRMIO), at least the same amount of non-swapped physical memory (RES) must be available to Abaqus. Since RES will generally be less than virtual memory (VIRT) by a relatively constant amount for a given simulation, it is necessary to slightly over-allocate the requested node memory `-mem=`. In the example script above, this over-allocation has been hardcoded to a conservative value of 3072MB based on initial tests of the standard Abaqus solver. To avoid long wait times associated with high MRMIO values, it may be worthwhile to investigate the impact on simulation performance associated with reducing the RES memory made available to Abaqus significantly below MRMIO. This can be done by decreasing the `-mem=` value which in turn will set an artificially low `memory=` value in the Abaqus command (found in the last line of the script). In doing so, one must ensure that RES does not drop below `MINIMUM MEMORY REQUIRED` (MMR) otherwise Abaqus will terminate due to insufficient memory (OOM). For example, if your MRMIO is 96GB, try running a series of short test jobs with `#SBATCH --mem=8G, 16G, 32G, 64G` until a minimally acceptable performance impact is found, noting that smaller values will result in an increasingly larger `/scratch` space for temporary files.


### Multiple Processes

To determine the memory required for scripts using multiple nodes, the memory estimates (per compute process) required to minimize I/O are given in the `dat` output file for completed jobs. If `mp_host_split` is not specified (or is set to 1), the total number of compute processes will be equal to the number of nodes. The `mem-per-cpu` value can then be determined approximately by multiplying the largest memory estimate by the number of nodes, then dividing by the number of `ntasks`. If, however, the `mp_host_split` value is specified (greater than 1), the `mem-per-cpu` value can be determined approximately from the largest memory estimate multiplied by the number of nodes, multiplied by the `mp_host_split` value, divided by the number of tasks. Note that the `mp_host_split` value must be less than or equal to the number of cores per node allocated at runtime, otherwise Abaqus will terminate. This scenario can be controlled by uncommenting to specify a value for tasks per node. The following definitive statement is given in each `dat` file and mentioned here for reference:

> THE MAXIMUM MEMORY THAT CAN BE ALLOCATED BY ABAQUS GENERALLY DEPENDS ON THE VALUE OF THE MEMORY PARAMETER AND THE AMOUNT OF PHYSICAL MEMORY AVAILABLE ON THE MACHINE. PLEASE CONSULT THE ABAQUS ANALYSIS USER'S MANUAL FOR MORE DETAILS. THE ACTUAL MEMORY AND DISK SPACE USAGE FOR /SCRATCH DATA WILL DEPEND ON THIS UPPER LIMIT AS WELL AS THE MEMORY REQUIRED TO MINIMIZE I/O. IF THE UPPER MEMORY LIMIT IS GREATER THAN THE MEMORY REQUIRED TO MINIMIZE I/O, THE ACTUAL MEMORY USAGE WILL BE CLOSE TO THE ESTIMATED VALUE OF MEMORY TO MINIMIZE I/O AND THE WORK DISK USAGE WILL BE CLOSE TO ZERO. OTHERWISE, THE ACTUAL MEMORY USED WILL BE CLOSE TO THE UPPER MEMORY LIMIT MENTIONED ABOVE, AND THE /SCRATCH DISK USAGE WILL BE ROUGHLY PROPORTIONAL TO THE DIFFERENCE BETWEEN THE ESTIMATED MEMORY TO MINIMIZE I/O AND THE UPPER MEMORY LIMIT. HOWEVER, IT IS IMPOSSIBLE TO ACCURATELY ESTIMATE THE /SCRATCH DISK SPACE.


## Graphical Mode

Abaqus can be used interactively in graphical mode on a cluster or on `gra-vdi` with VCN.

### On a Cluster

1.  Connect to a compute node (maximum duration 3 hours) with TigerVNC.
2.  Open a new terminal window and enter `module load StdEnv/2020 abaqus/2021`.
3.  Launch the application with `abaqus cae -mesa`.

### On gra-vdi

1.  Connect to a VDI node (maximum duration 24 hours) with TigerVNC.
2.  Open a new terminal window and enter one of the following statements:

    *   `module load StdEnv/2016 abaqus/6.14.1`
    *   `module load StdEnv/2016 abaqus/2020`
    *   `module load StdEnv/2020 abaqus/2021`
3.  Launch the application with `abaqus cae`.

For Abaqus to start in graphical mode, at least one free (unused) CAE license is required. The SHARCNET license has 2 free and 2 reserved licenses. If all 4 are in use according to:

```bash
[gra-vdi3:~] abaqus licensing lmstat -c $ABAQUSLM_LICENSE_FILE -a | grep "Users of cae"
Users of cae:  (Total of 4 licenses issued;  Total of 4 licenses in use)
```

The following error messages will be displayed when attempting to launch `abaqus cae`:

```
[gra-vdi3:~] abaqus cae
ABAQUSLM_LICENSE_FILE=27050@license3.sharcnet.ca
/opt/sharcnet/abaqus/2020/Commands/abaqus cae
No socket connection to license server manager.
Feature:       cae
License path:  27050@license3.sharcnet.ca:
FLEXnet Licensing error:-7,96
For further information, refer to the FLEXnet Licensing documentation,
or contact your local Abaqus representative.
Number of requested licenses: 1
Number of total licenses:     4
Number of licenses in use:    2
Number of available licenses: 2
Abaqus Error: Abaqus/CAE Kernel exited with an error.
```

## Site-Specific Usage

### SHARCNET License

The SHARCNET license is renewed until January 17, 2026. It consists of 2 cae tokens and 35 execution tokens with usage limits imposed at 10 tokens/user and 15 tokens/group. For groups that have purchased dedicated tokens, the usage limits of the free tokens are added to their reservation. Free tokens are available on a first-come, first-served basis and are primarily intended for testing and light usage before deciding whether or not to purchase dedicated tokens. The costs of dedicated tokens (in 2021) were approximately \$110 CAD per compute token and \$400 CAD per graphical interface token: submit a support request to request an official quote. The license can be used by anyone with an account with the Alliance, but only on SHARCNET hardware. Groups that purchase dedicated tokens for execution on the SHARCNET license server can also only use them on SHARCNET hardware, including `gra-vdi` (to run Abaqus in full graphical mode) and the Graham or Dusky clusters (to submit batch compute jobs to the queue). Before you can use the license, you must contact technical support to request access. In your email, 1) mention that it is intended for use on SHARCNET systems and 2) include a copy/paste of the following License Agreement statement with your full name and username entered in the indicated locations. Please note that each user must do this, it cannot be done once for a group; this includes PIs who have purchased their own dedicated tokens.


#### Agreement

```
----------------------------------------------------------------------------------
Subject: Abaqus SHARCNET Academic License User Agreement

This email is to confirm that i "_____________" with username "___________" will
only use “SIMULIA Academic Software” with tokens from the SHARCNET license server
for the following purposes:

1) on SHARCNET hardware where the software is already installed
2) in affiliation with a Canadian degree-granting academic institution
3) for education, institutional or instruction purposes and not for any commercial
   or contract-related purposes where results are not publishable
4) for experimental, theoretical and/or digital research work, undertaken primarily
   to acquire new knowledge of the underlying foundations of phenomena and observable
   facts, up to the point of proof-of-concept in a laboratory    
-----------------------------------------------------------------------------------
```

#### Configuring the License File

Configure your license file as follows (for use only on SHARCNET Graham, `gra-vdi`, and Dusky systems).

```bash
[gra-login1:~] cat ~/.licenses/abaqus.lic
prepend_path (
"LM_LICENSE_FILE",
"27050@license3.sharcnet.ca"
)
prepend_path (
"ABAQUSLM_LICENSE_FILE",
"27050@license3.sharcnet.ca"
)
```

If your jobs terminate abnormally and the scheduler output file contains the error message `*** ABAQUS/eliT_CheckLicense rank 0 terminated by signal 11 (Segmentation fault)`, check if your `abaqus.lic` file contains `ABAQUSLM_LICENSE_FILE` for Abaqus/2020. If the output file contains `License server machine is down or not responding etc.`, check if the `abaqus.lic` file contains `LM_LICENSE_FILE` for Abaqus/6.14.1, as shown. Since the `abaqus.lic` file shown contains both statements, you should not have this problem.


#### Querying the License Server

Connect to Graham, load Abaqus, and run one of the following commands:

```bash
ssh graham.alliancecan.ca
module load StdEnv/2020
module load abaqus
```

I) Check for running and queued jobs for the SHARCNET license server.

```bash
abaqus licensing lmstat -c $LM_LICENSE_FILE -a | egrep "Users|