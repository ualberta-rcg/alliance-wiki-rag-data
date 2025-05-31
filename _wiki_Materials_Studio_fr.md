# Materials Studio Installation Guide (French)

This page is a translated version of the page Materials Studio and the translation is 100% complete.

Other languages: [English](link-to-english-page), français

The Alliance does not have permission to centrally install Materials Studio on all clusters. If you have a license, follow these instructions to install the application on your account. Note that these instructions are valid for older software environments; if you are currently using the default 2020 environment, you will need to use a command like `module load StdEnv/2016.4` before starting (see [Standard Software Environments](link-to-standard-software-environments)).


## Installation de Materials Studio 2020

**Note:** These instructions have been tested with Materials Studio 2020.

If you have access to Materials Studio 2020, you need:

* The archive file `BIOVIA_2020.MaterialsStudio2020.tar` containing the installer;
* The IP address (or DNS name) and port of the pre-configured license server you want to connect to.


Upload the `BIOVIA_2020.MaterialsStudio2020.tar` file to your `/home` directory on the cluster you want to use. Then run the commands:

```bash
[name@server ~]$ export MS_LICENSE_SERVER=<port>@<server>
[name@server ~]$ eb MaterialsStudio-2020.eb --sourcepath=$HOME
```

Once the command is complete, log out of the cluster and log back in. You should then be able to load the module with:

```bash
[name@server ~]$ module load materialsstudio/2020
```

To access the license server from a login node, contact [technical support](link-to-technical-support) so that we can configure our firewalls to allow the software to access your license server.


## Installation de Materials Studio 2018

**Note:** These instructions have been tested with Materials Studio 2018.

If you have access to Materials Studio 2018, you will need:

* The archive file (`MaterialsStudio2018.tgz`) containing the installer,
* The IP address (or DNS name) and port of a pre-configured license server you want to connect to.


Download the `MaterialsStudio2018.tgz` file to your `/home` directory on the cluster and run the commands:

```bash
[name@server ~]$ export MS_LICENSE_SERVER=<port>@<server>
[name@server ~]$ eb /cvmfs/soft.computecanada.ca/easybuild/easyconfigs/m/MaterialsStudio/MaterialsStudio-2018.eb --disable-enforce-checksums --sourcepath=$HOME
```

When the operation is complete, log out of the cluster and log back in. You should then be able to load the module with:

```bash
[name@server ~]$ module load materialsstudio/2018
```

To access the license server from a compute node, contact [technical support](link-to-technical-support) so that our firewalls are configured accordingly.


### Installation pour un groupe

If you are a principal investigator and have a license, you only need to install the application once for all users in your group. Since team work is usually saved in the `/project` space, determine which directory in this space you want to use. For example, if it is `~/projects/A_DIRECTORY`, you will need to know these two values:

1. Determine the path for A_DIRECTORY with:

```bash
[name@server ~]$ PI_PROJECT_DIR=$(readlink -f ~/projects/A_DIRECTORY)
[name@server ~]$ echo $PI_PROJECT_DIR
```

2. Determine the group for A_DIRECTORY with:

```bash
[name@server ~]$ PI_GROUP=$(stat -c%G $PI_PROJECT_DIR)
[name@server ~]$ echo $PI_GROUP
```

With these two values, install Materials Studio as follows: Replace the default group with the team's group `def-`.

```bash
[name@server ~]$ newgrp $PI_GROUP
[name@server ~]$ chmod g+rsx $PI_PROJECT_DIR
[name@server ~]$ mkdir $PI_PROJECT_DIR/MatStudio2018
[name@server ~]$ MS_LICENSE_SERVER=<port>@<server> eb MaterialsStudio-2018-dummy-dummy.eb --installpath=$PI_PROJECT_DIR/MatStudio2018 --sourcepath=$HOME
```

Before launching the application:

Run the command:

```bash
[name@server ~]$ module use $PI_PROJECT_DIR/MatStudio2018/modules/2017/Core/
```

Team members can add this to their `~/.bashrc` file.

Load the materialsstudio module:

```bash
[name@server ~]$ module load materialsstudio
```

**NOTE:** Make sure to always replace the `PI_GROUP` and `PI_PROJECT_DIR` variables with the appropriate values.


## Exemples de scripts pour l'ordonnanceur Slurm

The following examples are valid provided you have followed the installation instructions above.

**File: file.txt**

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
module load materialsstudio/2018
# Create a list of nodes to be used for the job
DSD_MachineList="machines.LINUX"
slurm_hl2hl.py --format HP-MPI > $DSD_MachineList
export DSD_MachineList
# Job to run
RunDMol3.sh -np $SLURM_CPUS_PER_TASK Brucite001f
```

The following script uses the Materials Studio `RunCASTEP.sh` command.

**File: file.txt**

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1M
#SBATCH --time=0-12:00
module load materialsstudio/2018
DSD_MachineList="mpd.hosts"
slurm_hl2hl.py --format MPIHOSTLIST > $DSD_MachineList
export DSD_MachineList

RunCASTEP.sh -np $SLURM_CPUS_PER_TASK castepjob
if [ -f castepjob_NMR.param ]; then
  cp castepjob.check castepjob_NMR.check
  RunCASTEP.sh -np $SLURM_CPUS_PER_TASK castepjob_NMR
fi
```


## Installation de versions antérieures

To use a version of Materials Studio older than 2018, you must install it in an [Apptainer](link-to-apptainer) container.

1. Create an Apptainer container in which a compatible Linux distribution is installed.
2. Install Materials Studio in this container.
3. Upload the Apptainer container to your account.

**NOTE:** To access the license server from a compute node, contact [technical support](link-to-technical-support) so that our firewalls are configured accordingly.

Since the MPI version in the container may not be usable on multiple nodes, your jobs may be limited to entire nodes (single node).


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Materials_Studio/fr&oldid=140814](https://docs.alliancecan.ca/mediawiki/index.php?title=Materials_Studio/fr&oldid=140814)"
