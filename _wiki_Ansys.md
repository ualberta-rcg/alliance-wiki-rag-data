# Ansys

This page contains changes which are not marked for translation.

Other languages: English, français

Ansys is a software suite for engineering simulation and 3-D design. It includes packages such as Ansys Fluent and Ansys CFX.


## Contents

* [Licensing](#licensing)
    * [Configuring your license file](#configuring-your-license-file)
        * [Local license servers](#local-license-servers)
            * [Ready to use](#ready-to-use)
            * [Setup required](#setup-required)
    * [Checking license](#checking-license)
* [Version compatibility](#version-compatibility)
    * [Platform support](#platform-support)
    * [What's new](#whats-new)
    * [Service packs](#service-packs)
* [Cluster batch job submission](#cluster-batch-job-submission)
    * [Ansys Fluent](#ansys-fluent)
        * [Slurm scripts](#slurm-scripts)
            * [General purpose](#general-purpose)
            * [License requeue](#license-requeue)
            * [Solution restart](#solution-restart)
        * [Journal files](#journal-files)
        * [UDFs](#udfs)
            * [Interpreted](#interpreted)
            * [Compiled](#compiled)
            * [Parallel](#parallel)
            * [DPM](#dpm)
    * [Ansys CFX](#ansys-cfx)
        * [Slurm scripts](#slurm-scripts-1)
    * [Workbench](#workbench)
        * [Slurm scripts](#slurm-scripts-2)
    * [Mechanical](#mechanical)
        * [Slurm scripts](#slurm-scripts-3)
    * [Ansys EDT](#ansys-edt)
        * [Slurm scripts](#slurm-scripts-4)
    * [Ansys ROCKY](#ansys-rocky)
        * [Slurm scripts](#slurm-scripts-5)
* [Graphical use](#graphical-use)
    * [Compute nodes](#compute-nodes)
        * [Fluids](#fluids)
        * [Mapdl](#mapdl)
        * [Workbench](#workbench-1)
        * [Ansys EDT](#ansys-edt-1)
        * [Ensight](#ensight)
        * [Rocky](#rocky)
    * [VDI nodes](#vdi-nodes)
        * [Fluids](#fluids-1)
        * [Mapdl](#mapdl-1)
        * [Workbench](#workbench-2)
        * [Ansys EDT](#ansys-edt-2)
        * [Ensight](#ensight-1)
        * [Rocky](#rocky-1)
    * [SSH issues](#ssh-issues)
* [Site-specific usage](#site-specific-usage)
    * [SHARCNET license](#sharcnet-license)
        * [License server file](#license-server-file)
        * [Query license server](#query-license-server)
        * [Local VDI modules](#local-vdi-modules)
            * [Ansys modules](#ansys-modules)
            * [ansysedt modules](#ansysedt-modules)
* [Additive Manufacturing](#additive-manufacturing)
    * [Enable Additive](#enable-additive)
        * [Download Extension](#download-extension)
        * [Start Workbench](#start-workbench)
        * [Open Extension Manager](#open-extension-manager)
        * [Install Extension](#install-extension)
        * [Load Extension](#load-extension)
        * [Unload Extension](#unload-extension)
    * [Run Additive](#run-additive)
        * [Gra-vdi](#gra-vdi)
        * [Cluster](#cluster)
* [Help resources](#help-resources)
    * [Online documentation](#online-documentation)
    * [Youtube videos](#youtube-videos)
    * [Innovation Space](#innovation-space)


## Licensing

We are a hosting provider for Ansys. This means that we have the software installed on our clusters, but we do not provide a generic license accessible to everyone. However, many institutions, faculties, and departments already have licenses that can be used on our clusters. Once the legal aspects are worked out for licensing, there will be remaining technical aspects. The license server on your end will need to be reachable by our compute nodes. This will require our technical team to get in touch with the technical people managing your license software. In some cases, this has already been done. You should then be able to load the Ansys module, and it should find its license automatically. If this is not the case, please contact our [technical support](<technical support email>) so that they can arrange this for you.

### Configuring your license file

Our module for Ansys is designed to look for license information in a few places. One of those places is your `/home` folder. You can specify your license server by creating a file named `$HOME/.licenses/ansys.lic` consisting of two lines as shown. Customize the file to replacing FLEXPORT, INTEPORT and LICSERVER with appropriate values for your server.

```
FILE: ansys.lic
setenv("ANSYSLMD_LICENSE_FILE", "
FLEXPORT@LICSERVER
")
setenv("ANSYSLI_SERVERS", "
INTEPORT@LICSERVER
")
```

The following table provides established values for the CMC and SHARCNET license servers. To use a different server, locate the corresponding values as explained in [Local license servers](#local-license-servers).

```
TABLE: Preconfigured license servers
License     System/Cluster     LICSERVER             FLEXPORT     INTEPORT     VENDPORT     NOTICES
CMC         beluga             10.20.73.21           6624         2325         n/a          None
CMC         cedar              172.16.0.101          6624         2325         n/a          None
CMC         graham             10.25.1.56           6624         2325         n/a          NewIP Feb21/2025
CMC         narval             10.100.64.10          6624         2325         n/a          None
SHARCNET    beluga/cedar/graham/gra-vdi/narval license3.sharcnet.ca 1055         2325         n/a          None
SHARCNET    niagara            localhost             1055         2325         1793         None
```

Researchers who purchase a CMC license subscription must send their Alliance account username to `<cmcsupport@cmc.ca>` otherwise license checkouts will fail.

Researchers who purchase a CMC license subscription must create a CMC support case to submit their Alliance account username otherwise license checkouts will fail. The number of cores that can be used with a CMC license is described in the *Other Tricks and Tips* sections of the [Ansys Electronics Desktop and Ansys Mechanical/Fluids quick start guides](<link to quick start guides>).


#### Local license servers

Before a local institutional Ansys license server can be used on the Alliance, firewall changes will need to be done on both the server and cluster side. For many Ansys servers this work has already been done and they can be used by following the steps in the "Ready To Use" section below. For Ansys servers that have never used on the Alliance, two additional steps must be done as shown in the "Setup Required" section also below.

##### Ready to use

To use an ANSYS License server that has already been set up for use on the cluster where you will be submitting jobs, contact your local Ansys license server administrator and get the following three pieces (1->3) of information:

1.  the fully qualified hostname (LICSERVER) of the server
2.  the Ansys flex port (FLEXPORT) number commonly 1055
3.  the Ansys licensing interconnect port (INTEPORT) number commonly 2325

Once the three pieces of information are collected, configure your `~/.licenses/ansys.lic` file by plugging the values for LICSERVER, FLEXPORT and INTEPORT into the `FILE: ansys.lic` template above.

##### Setup required

If your local Ansys license server has never been setup for use on the Alliance cluster(s) where you will be submitting jobs, then in addition to (1->3) above, you will ALSO need to get the following items (4,5) from the administrator:

4.  the static vendor port (VENDPORT) number from your local Ansys server administrator
5.  confirmation that `<servername>` will resolve to the same IP as LICSERVER on Alliance clusters where the `<servername>` can be found in the first line of the license file with format "SERVER `<servername>` `<host id>` `<lmgrd port>`". Item (5) is required; otherwise, Ansys license checkouts will not work on any remote cluster at the Alliance. If it turns out `<servername>` does not meet this requirement then request your license administrator to change `<servername>` to either the same fully qualified hostname as LICSERVER or at least to a hostname that will resolve to the same IP address as LICSERVER remotely.

Finally, send (1->4) by email to [technical support](<technical support email>) being sure to mention which Alliance cluster(s) you want to run Ansys jobs on. Alliance system administrators will then proceed to open the firewall so license checkout requests can reach your license server from the specified cluster(s) compute nodes. In return you will then receive a range of IP addresses to forward to your server administrator can then open the local firewall to permit inbound Ansys license connections from the cluster(s) on EACH of the ports defined by FLEXPORT, INTEPORT and VENDPORT.

### Checking license

To test if your `ansys.lic` is configured and working properly copy/paste the following sequence of commands on the cluster you will be submitting jobs to. The only required change would be to specify YOURUSERID. If the software on a remote license server has not been updated then a failure can occur if the latest module version of ansys is loaded to test with. Therefore to be certain the license checkouts will work when jobs are run in the queue, the same ansys module version that you load in your slurm scripts should be specified below.

```bash
[gra-login:~] cd /tmp
[gra-login:~] salloc --time=1:0:0 --mem=1000M --account=def-YOURUSERID
[gra-login:~] module load StdEnv/2023; module load ansys/2023R2
[gra-login:~] $EBROOTANSYS/v$(echo ${EBVERSIONANSYS:2:2}${EBVERSIONANSYS:5:1})/licensingclient/linx64/lmutil lmstat -c $ANSYSLMD_LICENSE_FILE 1> /dev/null && echo Success || echo Fail
```

If `Success` is output license checkouts should work when jobs are submitted to the queue.

If `Fail` is output then jobs will likely fail requiring a problem ticket to be submitted to resolve.


## Version compatibility

Ansys simulations are typically forward compatible but **NOT** backwards compatible. This means that simulations created using an older version of Ansys can be expected to load and run fine with any newer version. For example, a simulation created and saved with ansys/2022R2 should load and run smoothly with ansys/2023R2 but **NOT** the other way around. While it maybe possible to start a simulation running with an older version random error messages or crashing will likely occur. Regarding Fluent simulations, if you cannot recall which version of ansys was used to create your cas file try grepping it as follows to look for clues:

```bash
$ grep -ia fluent combustor.cas
  (0 "fluent15.0.7  build-id: 596")
$ grep -ia fluent cavity.cas.h5
  ANSYS_FLUENT 24.1 Build 1018
```

### Platform support

Ansys provides detailed platform support information describing software/hardware compatibility for the [Current Release](<link to current release>) and [Previous Releases](<link to previous releases>). The *Platform Support by Application / Product* pdf is of special interest since it shows which packages are supported under Windows but not under Linux and thus not on the Alliance such as Spaceclaim.

### What's new

Ansys posts [Product Release and Updates](<link to product releases>) for the latest releases. Similar information for previous releases can generally be pulled up for various application topics by visiting the Ansys [blog](<link to ansys blog>) page and using the FILTERS search bar. For example, searching on "What’s New Fluent 2024 gpu" pulls up a document with title "What’s New for Ansys Fluent in 2024 R1?" containing a wealth of the latest gpu support information. Specifying a version number in the *Press Release* search bar is also a good way to find new release information. At the time of this writing Ansys 2024R2 is the current release and will be installed when interest is expressed or there is evident need to support newer hardware or solver capabilities. To request a new version be installed [submit a ticket](<link to submit ticket>).

### Service packs

Starting with Ansys 2024 a separate Ansys module will appear on the clusters with a decimal and two digits appearing after the release number whenever a service pack is been installed over the initial release. For example, the initial 2024 release with no service pack applied may be loaded with `module load ansys/2024R1` while a module with Service Pack 3 applied may be loaded with `module load ansys/2024R1.03` instead. If a service pack is already available by the time a new release is to be installed, then most likely only a module for that service pack number will be installed unless a request to install the initial release is also received.

Most users will likely want to load the latest module version equipped with the latest installed service pack which can be achieved by simply doing `module load ansys`. While it's not expected service packs will impact numerical results, the changes they make are extensive and so, if computations have already been done with the initial release or an earlier service pack then some groups may prefer to continue using it. Having separate modules for each service pack makes this possible. Starting with Ansys 2024R1 a detailed description of what each service pack does can be found by searching this [link](<link to service pack details>) for *Service Pack Details*. Future versions will presumably be similarly searchable by manually modifying the version number contained in the link.


## Cluster batch job submission

The Ansys software suite comes with multiple implementations of MPI to support parallel computation. Unfortunately, none of them support our [Slurm scheduler](<link to slurm scheduler>). For this reason, we need special instructions for each Ansys package on how to start a parallel job. In the sections below, we give examples of submission scripts for some of the packages. While the slurm scripts should work with on all clusters, Niagara users may need to make some additional changes covered [here](<link to niagara changes>).


### Ansys Fluent

Typically, you would use the following procedure to run Fluent on one of our clusters:

1.  Prepare your Fluent job using Fluent from the Ansys Workbench on your desktop machine up to the point where you would run the calculation.
2.  Export the "case" file with `File > Export > Case…` or find the folder where Fluent saves your project's files. The case file will often have a name like `FFF-1.cas.gz`.
3.  If you already have data from a previous calculation, which you want to continue, export a "data" file as well (`File > Export > Data…`) or find it in the same project folder (`FFF-1.dat.gz`).
4.  Transfer the case file (and if needed the data file) to a directory on the `/project` or `/scratch` filesystem on the cluster. When exporting, you can save the file(s) under a more instructive name than `FFF-1.*` or rename them when they are uploaded.
5.  Now you need to create a "journal" file. Its purpose is to load the case file (and optionally the data file), run the solver and finally write the results. See examples below and remember to adjust the filenames and desired number of iterations.
6.  If jobs frequently fail to start due to license shortages and manual resubmission of failed jobs is not convenient, consider modifying your script to requeue your job (up to 4 times) as shown under the *by node + requeue* tab further below. Be aware that doing this will also requeue simulations that fail due to non-license related issues (such as divergence), resulting in lost compute time. Therefore it is strongly recommended to monitor and inspect each Slurm output file to confirm each requeue attempt is license related. When it is determined that a job is requeued due to a simulation issue, immediately manually kill the job progression with `scancel jobid` and correct the problem.
7.  After running the job, you can download the data file and import it back into Fluent with `File > Import > Data…`.


#### Slurm scripts

##### General purpose

Most Fluent jobs should use the following *by node* script to minimize solution latency and maximize performance over as few nodes as possible. Very large jobs, however, might wait less in the queue if they use a *by core* script. However, the startup time of a job using many nodes can be significantly longer, thus offsetting some of the benefits. In addition, be aware that running large jobs over an unspecified number of potentially very many nodes will make them far more vulnerable to crashing if any of the compute nodes fail during the simulation. The scripts will ensure Fluent uses shared memory for communication when run on a single node or distributed memory (utilizing MPI and the appropriate HPC interconnect) when run over multiple nodes. The two narval tabs maybe be useful to provide a more robust alternative if fluent crashes during the initial auto mesh partitioning phase when using the standard intel based scripts with the parallel solver. The other option would be to manually perform the mesh partitioning in the fluent gui then try to run the job again on the cluster with the intel scripts. Doing so will allow you to inspect the partition statistics and specify the partitioning method to obtain an optimal result. The number of mesh partitions should be an integral multiple of the number of cores; for optimal efficiency, ensure at least 10000 cells per core.

*   Multinode (by node)
*   Multinode (by core)
*   Multinode (by node, narval)
*   Multinode (by core, narval)
*   Multinode (by node, niagara)


```bash
File: script-flu-bynode-intel.sh
#!/bin/bash
#SBATCH --account=def-group   # Specify account name
#SBATCH --time=00-03:00       # Specify time limit dd-hh:mm
#SBATCH --nodes=1             # Specify number of compute nodes (narval 1 node max)
#SBATCH --ntasks-per-node=32  # Specify number of cores per node (graham 32 or 44, cedar 48, beluga 40, narval 64, or less)
#SBATCH --mem=0               # Do not change (allocates all memory per compute node)
#SBATCH --cpus-per-task=1     # Do not change
module load StdEnv/2023
# Do not change
module load ansys/2023R2
# or newer versions (beluga, cedar, graham, narval)
#module load StdEnv/2020      # no longer supported
#module load ansys/2019R3     # or newer versions (narval only)
#module load ansys/2021R2     # or newer versions (beluga, cedar, graham)
MYJOURNALFILE=sample.jou
# Specify your journal file name
MYVERSION=3d
# Specify 2d, 2ddp, 3d or 3ddp
# ------- do not change any lines below --------
if [[ "${CC_CLUSTER}" == narval ]] ; then
    if [ "$EBVERSIONGENTOO" == 2020 ] ; then
        module load intel/2021 intelmpi
        export INTELMPI_ROOT=$I_MPI_ROOT/mpi/latest
        export HCOLL_RCACHE=^ucs
    elif [ "$EBVERSIONGENTOO" == 2023 ] ; then
        module load intel/2023 intelmpi
        export INTELMPI_ROOT=$I_MPI_ROOT
    fi
    unset I_MPI_HYDRA_BOOTSTRAP_EXEC_EXTRA_ARGS
    unset I_MPI_ROOT
fi
slurm_hl2hl.py --format ANSYS-FLUENT > /tmp/machinefile-$SLURM_JOB_ID
NCORES=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE * SLURM_CPUS_PER_TASK))
if [ "$SLURM_NNODES" == 1 ] ; then
    fluent -g $MYVERSION -t $NCORES -affinity=0 -mpi=intel -pshmem -i $MYJOURNALFILE
else
    fluent -g $MYVERSION -t $NCORES -affinity=0 -mpi=intel -pib -cnf=/tmp/machinefile-$SLURM_JOB_ID -i $MYJOURNALFILE
fi
```

```bash
File: script-flu-bycore-intel.sh
#!/bin/bash
#SBATCH --account=def-group   # Specify account
#SBATCH --time=00-03:00       # Specify time limit dd-hh:mm
##SBATCH --nodes=1            # Uncomment to specify (narval 1 node max)
#SBATCH --ntasks=16           # Specify total number of cores
#SBATCH --mem-per-cpu=4G      # Specify memory per core
#SBATCH --cpus-per-task=1     # Do not change
module load StdEnv/2023
# Do not change
module load ansys/2023R2
# or newer versions (beluga, cedar, graham, narval)
#module load StdEnv/2020      # no longer supported
#module load ansys/2019R3     # or newer versions (narval only)
#module load ansys/2021R2     # or newer versions (beluga, cedar, graham)
MYJOURNALFILE=sample.jou
# Specify your journal file name
MYVERSION=3d
# Specify 2d, 2ddp, 3d or 3ddp
# ------- do not change any lines below --------
if [[ "${CC_CLUSTER}" == narval ]] ; then
    if [ "$EBVERSIONGENTOO" == 2020 ] ; then
        module load intel/2021 intelmpi
        export INTELMPI_ROOT=$I_MPI_ROOT/mpi/latest
        export HCOLL_RCACHE=^ucs
    elif [ "$EBVERSIONGENTOO" == 2023 ] ; then
        module load intel/2023 intelmpi
        export INTELMPI_ROOT=$I_MPI_ROOT
    fi
    unset I_MPI_HYDRA_BOOTSTRAP_EXEC_EXTRA_ARGS
    unset I_MPI_ROOT
fi
slurm_hl2hl.py --format ANSYS-FLUENT > /tmp/machinefile-$SLURM_JOB_ID
NCORES=$((SLURM_NTASKS * SLURM_CPUS_PER_TASK))
if [ "$SLURM_NNODES" == 1 ] ; then
    fluent -g $MYVERSION -t $NCORES -affinity=0 -mpi=intel -pshmem -i $MYJOURNALFILE
else
    fluent -g $MYVERSION -t $NCORES -affinity=0 -mpi=intel -pib -cnf=/tmp/machinefile-$SLURM_JOB_ID -i $MYJOURNALFILE
fi
```

```bash
File: script-flu-bynode-openmpi.sh
#!/bin/bash
#SBATCH --account=def-group   # Specify account name
#SBATCH --time=00-03:00       # Specify time limit dd-hh:mm
#SBATCH --nodes=1             # Specify number of compute nodes
#SBATCH --ntasks-per-node=64  # Specify number of cores per node (narval 64 or less)
#SBATCH --mem=0               # Do not change (allocates all memory per compute node)
#SBATCH --cpus-per-task=1     # Do not change
module load StdEnv/2023
# Do not change
module load ansys/2023R2
# or newer versions (narval only)
MYJOURNALFILE=sample.jou
# Specify your journal file name
MYVERSION=3d
# Specify 2d, 2ddp, 3d or 3ddp
# ------- do not change any lines below --------
export OPENMPI_ROOT=$EBROOTOPENMPI
export OMPI_MCA_hwloc_base_binding_policy=core
slurm_hl2hl.py --format ANSYS-FLUENT > /tmp/mf-$SLURM_JOB_ID
for i in `cat /tmp/mf-$SLURM_JOB_ID | uniq` ; do
    echo "${i}:$(cat /tmp/mf-$SLURM_JOB_ID | grep $i | wc -l)" >> /tmp/machinefile-$SLURM_JOB_ID ;
done
NCORES=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE * SLURM_CPUS_PER_TASK))
if [ "$SLURM_NNODES" == 1 ] ; then
    fluent -g $MYVERSION -t $NCORES -affinity=0 -mpi=openmpi -pshmem -i $MYJOURNALFILE
else
    fluent -g $MYVERSION -t $NCORES -affinity=0 -mpi=openmpi -pib -cnf=/tmp/machinefile-$SLURM_JOB_ID -i $MYJOURNALFILE
fi
```

```bash
File: script-flu-bycore-openmpi.sh
#!/bin/bash
#SBATCH --account=def-group   # Specify account name
#SBATCH --time=00-03:00       # Specify time limit dd-hh:mm
##SBATCH --nodes=1            # Uncomment to specify number of compute nodes (optional)
#SBATCH --ntasks=16           # Specify total number of cores
#SBATCH --mem-per-cpu=4G      # Specify memory per core
#SBATCH --cpus-per-task=1     # Do not change
module load StdEnv/2023
# Do not change
module load ansys/2023R2
# or newer versions (narval only)
MYJOURNALFILE=sample.jou
# Specify your journal file name
MYVERSION=3d
# Specify 2d, 2ddp, 3d or 3ddp
# ------- do not change any lines below --------
export OPENMPI_ROOT=$EBROOTOPENMPI
export OMPI_MCA_hwloc_base_binding_policy=core
slurm_hl2hl.py --format ANSYS-FLUENT > /tmp/mf-$SLURM_JOB_ID
for i in `cat /tmp/mf-$SLURM_JOB_ID | uniq` ; do
    echo "${i}:$(cat /tmp/mf-$SLURM_JOB_ID | grep $i | wc -l)" >> /tmp/machinefile-$SLURM_JOB_ID ;
done
NCORES=$((SLURM_NTASKS * SLURM_CPUS_PER_TASK))
if [ "$SLURM_NNODES" == 1 ] ; then
    fluent -g $MYVERSION -t $NCORES -affinity=0 -mpi=openmpi -pshmem -i $MYJOURNALFILE
else
    fluent -g $MYVERSION -t $NCORES -affinity=0 -mpi=openmpi -pib -cnf=/tmp/machinefile-$SLURM_JOB_ID -i $MYJOURNALFILE
fi
```

```bash
File: script-flu-bynode-intel-nia.sh
#!/bin/bash
#SBATCH --account=def-group      # Specify account name
#SBATCH --time=00-03:00          # Specify time limit dd-hh:mm
#SBATCH --nodes=2                # Specify number of compute nodes
#SBATCH --ntasks-per-node=80     # Specify number cores per node (niagara 80 or less)
#SBATCH --mem=0                  # Do not change (allocate all memory per compute node)
#SBATCH --cpus-per-task=1        # Do not change (required parameter)
module load CCEnv StdEnv/2023
# Do not change
module load arch/avx512
module load ansys/2023R2
# or newer versions (niagara only)
MYJOURNALFILE=sample.jou
# Specify your journal file name
MYVERSION=3d
# Specify 2d, 2ddp, 3d or 3ddp
# These settings are used instead of your ~/.licenses/ansys.lic
LICSERVER=license3.sharcnet.ca
# Specify license server hostname
FLEXPORT=1055
# Specify server flex port
INTEPORT=2325
# Specify server interconnect port
VENDPORT=1793
# Specify server vendor port
# ------- do not change any lines below --------
ssh nia-gw -fNL $FLEXPORT:$LICSERVER:$FLEXPORT
# Do not change
ssh nia-gw -fNL $INTEPORT:$LICSERVER:$INTEPORT
# Do not change
ssh nia-gw -fNL $VENDPORT:$LICSERVER:$VENDPORT
# Do not change
export ANSYSLMD_LICENSE_FILE=$FLEXPORT@localhost
# Do not change
export ANSYSLI_SERVERS=$INTEPORT@localhost
# Do not change
slurm_hl2hl.py --format ANSYS-FLUENT > $SLURM_SUBMIT_DIR/machinefile-$SLURM_JOB_ID
NCORES=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE * SLURM_CPUS_PER_TASK))
if [ ! -L "$HOME/.ansys" ] ; then
    echo "ERROR: A link to a writable .ansys directory does not exist."
    echo 'Remove ~/.ansys if one exists and then run: ln -s $SCRATCH/.ansys ~/.ansys'
    echo "Then try submitting your job again. Aborting the current job now!"
elif [ ! -L "$HOME/.fluentconf" ] ; then
    echo "ERROR: A link to a writable .fluentconf directory does not exist."
    echo 'Remove ~/.fluentconf if one exists and run: ln -s $SCRATCH/.fluentconf ~/.fluentconf'
    echo "Then try submitting your job again. Aborting the current job now!"
elif [ ! -L "$HOME/.flrecent" ] ; then
    echo "ERROR: A link to a writable .flrecent file does not exist."
    echo 'Remove ~/.flrecent if one exists and then run: ln -s $SCRATCH/.flrecent ~/.flrecent'
    echo "Then try submitting your job again. Aborting the current job now!"
else
    mkdir -pv $SCRATCH/.ansys
    mkdir -pv $SCRATCH/.fluentconf
    touch $SCRATCH/.flrecent
    if [ "$SLURM_NNODES" == 1 ] ; then
        fluent -g $MYVERSION -t $NCORES -affinity=0 -mpi=intel -pshmem -i $MYJOURNALFILE
    else
        fluent -g $MYVERSION -t $NCORES -affinity=0 -mpi=intel -ssh -pib -cnf=$SLURM_SUBMIT_DIR/machinefile-$SLURM_JOB_ID -i $MYJOURNALFILE
    fi
fi
```

##### License requeue

The scripts in this section should only be used with Fluent jobs that are known to complete normally without generating any errors in the output however typically require multiple requeue attempts to checkout licenses. They are not recommended for Fluent jobs that may 1) run for a long time before crashing 2) run to completion but contain unresolved journal file warnings, since in both cases the simulations will be repeated from the beginning until the maximum number of requeue attempts specified by the `array` value is reached. For these types of jobs, the general purpose scripts above should be used instead.

*   Multinode (by node + requeue)
*   Multinode (by core + requeue)


```bash
File: script-flu-bynode+requeue.sh
#!/bin/bash
#SBATCH --account=def-group   # Specify account
#SBATCH --time=00-03:00       # Specify time limit dd-hh:mm
#SBATCH --nodes=1             # Specify number of compute nodes (narval 1 node max)
#SBATCH --ntasks-per-node=32  # Specify number of cores per node (graham 32 or 44, cedar 48, beluga 40, or less)
#SBATCH --mem=0               # Do not change (allocates all memory per compute node)
#SBATCH --cpus-per-task=1     # Do not change
#SBATCH --array=1-5%1         # Specify number of requeue attempts (2 or more, 5 is shown)
module load StdEnv/2023
# Do not change
module load ansys/2023R2
# Specify version (beluga, cedar, graham, narval)
#module load StdEnv/2020      # no longer supported
#module load ansys/2019R3     # or newer versions (narval only)
#module load ansys/2021R2     # or newer versions (beluga, cedar, graham)
MYJOURNALFILE=sample.jou
# Specify your journal file name
MYVERSION=3d
# Specify 2d, 2ddp, 3d or 3ddp
# ------- do not change any lines below --------
if [[ "${CC_CLUSTER}" == narval ]] ; then
    if [ "$EBVERSIONGENTOO" == 2020 ] ; then
        module load intel/2021 intelmpi
        export INTELMPI_ROOT=$I_MPI_ROOT/mpi/latest
        export HCOLL_RCACHE=^ucs
    elif [ "$EBVERSIONGENTOO" == 2023 ] ; then
        module load intel/2023 intelmpi
        export INTELMPI_ROOT=$I_MPI_ROOT
    fi
    unset I_MPI_HYDRA_BOOTSTRAP_EXEC_EXTRA_ARGS
    unset I_MPI_ROOT
fi
slurm_hl2hl.py --format ANSYS-FLUENT > /tmp/machinefile-$SLURM_JOB_ID
NCORES=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE * SLURM_CPUS_PER_TASK))
if [ "$SLURM_NNODES" == 1 ] ; then
    fluent -g $MYVERSION -t $NCORES -affinity=0 -mpi=intel -pshmem -i $MYJOURNALFILE
else
    fluent -g $MYVERSION -t $NCORES -affinity=0 -mpi=intel -pib -cnf=/tmp/machinefile-$SLURM_JOB_ID -i $MYJOURNALFILE
fi
if [ $? -eq 0 ] ; then
    echo "Job completed successfully! Exiting now."
    scancel $SLURM_ARRAY_JOB_ID
else
    echo "Job attempt $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_COUNT failed due to license or simulation issue!"
    if [ $SLURM_ARRAY_TASK_ID -lt $SLURM_ARRAY_TASK_COUNT ] ; then
        echo "Resubmitting job now …"
    else
        echo "All job attempts failed exiting now."
    fi
fi
```

```bash
File: script-flu