# Materials Studio Installation Guide

Other languages: English, français

The Alliance does not have permission to install Materials Studio centrally on all clusters. If you have a license, follow these instructions to install the application in your account. Please note that the current instructions are only valid for older standard software environments.  Before beginning, you will need to use a command like `module load StdEnv/2016.4` if you are using the default 2020 standard software environment.


## Installing Materials Studio 2020

**Note:** These instructions have been tested with Materials Studio 2020.

If you have access to Materials Studio 2020, you will need two things:

1. The archive file containing the installer (named `BIOVIA_2020.MaterialsStudio2020.tar`).
2. The IP address (or DNS name) and port of a configured license server.

Upload `BIOVIA_2020.MaterialsStudio2020.tar` to your `/home` folder on the cluster. Then, run these commands:

```bash
export MS_LICENSE_SERVER=<port>@<server>
eb MaterialsStudio-2020.eb --sourcepath=$HOME
```

After the command completes, log out of the cluster and log back in. You should then be able to load the module with:

```bash
module load materialsstudio/2020
```

To access the license server from compute nodes, you will need to [contact technical support](link-to-support) so the firewall(s) can be configured to allow the connection.


## Installing Materials Studio 2018

**Note:** These instructions have been tested with Materials Studio 2018.

If you have access to Materials Studio 2018, you will need:

1. The archive file containing the installer (named `MaterialsStudio2018.tgz`).
2. The IP address (or DNS name) and port of a configured license server.

Upload `MaterialsStudio2018.tgz` to your `/home` folder on the cluster. Then, run these commands:

```bash
export MS_LICENSE_SERVER=<port>@<server>
eb /cvmfs/soft.computecanada.ca/easybuild/easyconfigs/m/MaterialsStudio/MaterialsStudio-2018.eb --disable-enforce-checksums --sourcepath=$HOME
```

After the command completes, log out of the cluster and log back in. You should then be able to load the module with:

```bash
module load materialsstudio/2018
```

To access the license server from compute nodes, you will need to [contact technical support](link-to-support) so the firewall(s) can be configured to allow the connection.


### Team Installation

If you are a PI holding the Materials Studio license, you can install it once for all your group members.  Since team work is usually stored in the `/project` space, determine which project directory to use.  Suppose it is `~/projects/A_DIRECTORY`. You'll need these values:

1.  The actual path of `A_DIRECTORY`:

    ```bash
    PI_PROJECT_DIR=$(readlink -f ~/projects/A_DIRECTORY)
    echo $PI_PROJECT_DIR
    ```

2.  The group of `A_DIRECTORY`:

    ```bash
    PI_GROUP=$(stat -c%G $PI_PROJECT_DIR)
    echo $PI_GROUP
    ```

With these values, install Materials Studio:

Change the default group to your team's group:

```bash
newgrp $PI_GROUP
```

Open permissions for your project directory:

```bash
chmod g+rsx $PI_PROJECT_DIR
```

Create an install directory:

```bash
mkdir $PI_PROJECT_DIR/MatStudio2018
```

Install the software:

```bash
MS_LICENSE_SERVER=<port>@<server> eb MaterialsStudio-2018-dummy-dummy.eb --installpath=$PI_PROJECT_DIR/MatStudio2018 --sourcepath=$HOME
```

Before running the software:

```bash
module use $PI_PROJECT_DIR/MatStudio2018/modules/2017/Core/
```

Your team members might want to add this to their `~/.bashrc` file.

Load the materialsstudio module:

```bash
module load materialsstudio
```

**NOTE:** Always replace `PI_GROUP` and `PI_PROJECT_DIR` with their appropriate values.


## Examples of Slurm Job Submission Scripts

The following examples assume you've installed Materials Studio 2018 as described above.


**File: file.txt**

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
module load materialsstudio/2018
DSD_MachineList="machines.LINUX"
slurm_hl2hl.py --format HP-MPI > $DSD_MachineList
export DSD_MachineList
RunDMol3.sh -np $SLURM_CPUS_PER_TASK Brucite001f
```

Here's an example using `RunCASTEP.sh`:

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


## Installing Earlier Versions of Materials Studio

For versions older than 2018, install into an Apptainer container. This involves:

1. Creating an Apptainer container with a compatible Linux distribution.
2. Installing Materials Studio into the container.
3. Uploading the container to your account and using it.

**NOTE:** To access the license server from compute nodes, you will need to [contact technical support](link-to-support) to configure the firewall(s).

You might be limited to single-node jobs because the container's MPI version may not work across nodes.


**(Replace `link-to-support` with the actual link to technical support.)**
