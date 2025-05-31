# STAR-CCM+

STAR-CCM+ is a multidisciplinary engineering simulation suite used to model acoustics, fluid dynamics, heat transfer, rheology, multiphase flows, particle flows, solid mechanics, reacting flows, electrochemistry, and electromagnetics. It is developed by Siemens.

## License limitations

We are authorized to host STAR-CCM+ binaries on our servers, but we do *not* provide licenses. You will need your own license to use this software.  A remote POD license can be purchased directly from [Siemens](https://www.siemens.com/global/en.html). Alternatively, a local license hosted at your institution can be used, provided it's accessible through the firewall from the cluster where jobs are run.

## Configuring your account

To configure your account to use a license server with our Star-CCM+ module, create a license file `$HOME/.licenses/starccm.lic` with the following layout:

**File: starccm.lic**

```
SERVER <server>
ANY <port>
USE_SERVER
```

Replace `<server>` with the hostname (or IP address) and `<port>` with the static vendor port of the license server.


### POD license file

Researchers with a POD license purchased from [Siemens](https://www.siemens.com/global/en.html) can specify it by creating a `~/.licenses/starccm.lic` file as follows:

**File: starccm.lic**

```
SERVER flex.cd-adapco.com
ANY 1999
USE_SERVER
```

This works on any cluster (except Niagara).  Also, set `LM_PROJECT` to your CD-ADAPCO PROJECT ID in your Slurm script. Manually setting `CDLMD_LICENSE_FILE="<port>@<server>"` in your Slurm script is no longer required.


## Cluster batch job submission

Select one of the available modules:

*   `starccm` for the double-precision flavor (e.g., `module load starccm/19.04.007-R8`)
*   `starccm-mixed` for the mixed-precision flavor (e.g., `module load starccm-mixed/19.04.007`)

When submitting jobs to a cluster for the first time, you must set up the environment to use your license. If using Siemens' remote pay-on-usage license server, create a `~/.licenses/starccm.lic` file as described in the "Configuring your account - POD license file" section; license checkouts should work immediately.  However, if using an institutional license server, after creating your `~/.licenses/starccm.lic` file, submit a problem ticket to [technical support](<insert_support_link_here>) so they can help coordinate the necessary one-time network firewall changes required for access (assuming the server hasn't been set up for access from the Alliance cluster you'll be using). If you still have problems with licensing, try removing or renaming the file `~/.flexlmrc`, as previous search paths and/or license server settings might be stored there. Note that temporary output files from STAR-CCM+ job runs may accumulate in hidden directories named `~/.star-version_number`, consuming valuable quota space.  These can be removed periodically by running `rm -ri ~/.starccm*` and replying "yes" when prompted.


### Slurm scripts

The following examples show `starccm_job.sh` files for different clusters:

#### Beluga

**File: starccm_job.sh**

```bash
#!/bin/bash
#SBATCH --account=def-group   # Specify some account
#SBATCH --time=00-01:00       # Time limit: dd-hh:mm
#SBATCH --nodes=1             # Specify 1 or more nodes
#SBATCH --cpus-per-task=40    # Request all cores per node
#SBATCH --mem=0               # Request all memory per node
#SBATCH --ntasks-per-node=1   # Do not change this value
#module load StdEnv/2020      # Versions < 18.06.006
module load StdEnv/2023
#module load starccm/18.06.006-R8
module load starccm-mixed/18.06.006
SIM_FILE='mysample.sim'       # Specify your input sim filename
#JAVA_FILE='mymacros.java'    # Uncomment to specify an input java filename
# Comment the next line when using an institutional license server
LM_PROJECT='my22digitpodkey' # Specify your Siemens Power on Demand (PoD) Key
# ------- no changes required below this line --------
slurm_hl2hl.py --format STAR-CCM+ > $SLURM_TMPDIR/machinefile
NCORE=$((SLURM_NNODES * SLURM_CPUS_PER_TASK * SLURM_NTASKS_PER_NODE))
if [ -n "$LM_PROJECT" ]; then
  # Siemens PoD license server
  starccm+ -jvmargs -Xmx4G -jvmargs -Djava.io.tmpdir=$SLURM_TMPDIR -batch -power -podkey $LM_PROJECT -np $NCORE -nbuserdir $SLURM_TMPDIR -machinefile $SLURM_TMPDIR/machinefile $JAVA_FILE $SIM_FILE
else
  # Institutional license server
  starccm+ -jvmargs -Xmx4G -jvmargs -Djava.io.tmpdir=$SLURM_TMPDIR -batch -np $NCORE -nbuserdir $SLURM_TMPDIR -machinefile $SLURM_TMPDIR/machinefile $JAVA_FILE $SIM_FILE
fi
```

#### Cedar

**File: starccm_job.sh**

```bash
#!/bin/bash
#SBATCH --account=def-group   # Specify some account
#SBATCH --time=00-01:00       # Time limit: dd-hh:mm
#SBATCH --nodes=1             # Specify 1 or more nodes
#SBATCH --cpus-per-task=48    # Request all cores per node (32 or 48)
#SBATCH --mem=0               # Request all memory per node
#SBATCH --ntasks-per-node=1   # Do not change this value
#module load StdEnv/2020      # Versions < 18.06.006
module load StdEnv/2023
#module load starccm/18.06.006-R8
module load starccm-mixed/18.06.006
SIM_FILE='mysample.sim'       # Specify your input sim filename
#JAVA_FILE='mymacros.java'    # Uncomment to specify an input java filename
# Comment the next line when using an institutional license server
LM_PROJECT='my22digitpodkey' # Specify your Siemens Power on Demand (PoD) Key
# ------- no changes required below this line --------
slurm_hl2hl.py --format STAR-CCM+ > $SLURM_TMPDIR/machinefile
NCORE=$((SLURM_NNODES * SLURM_CPUS_PER_TASK * SLURM_NTASKS_PER_NODE))
if [ -n "$LM_PROJECT" ]; then
  # Siemens PoD license server
  starccm+ -jvmargs -Xmx4G -jvmargs -Djava.io.tmpdir=$SLURM_TMPDIR -batch -power -podkey $LM_PROJECT -np $NCORE -nbuserdir $SLURM_TMPDIR -machinefile $SLURM_TMPDIR/machinefile $JAVA_FILE $SIM_FILE -mpi intel -fabric psm2
else
  # Institutional license server
  starccm+ -jvmargs -Xmx4G -jvmargs -Djava.io.tmpdir=$SLURM_TMPDIR -batch -np $NCORE -nbuserdir $SLURM_TMPDIR -machinefile $SLURM_TMPDIR/machinefile $JAVA_FILE $SIM_FILE -mpi intel -fabric psm2
fi
```

#### Graham

**File: starccm_job.sh**

```bash
#!/bin/bash
#SBATCH --account=def-group   # Specify some account
#SBATCH --time=00-01:00       # Time limit: dd-hh:mm
#SBATCH --nodes=1             # Specify 1 or more nodes
#SBATCH --cpus-per-task=32    # Request all cores per node (32 or 44)
#SBATCH --mem=0               # Request all memory per node
#SBATCH --ntasks-per-node=1   # Do not change this value
#module load StdEnv/2020      # Versions < 18.06.006
module load StdEnv/2023
#module load starccm/18.06.006-R8
module load starccm-mixed/18.06.006
SIM_FILE='mysample.sim'       # Specify your input sim filename
#JAVA_FILE='mymacros.java'    # Uncomment to specify an input java filename
# Comment the next line when using an institutional license server
LM_PROJECT='my22digitpodkey' # Specify your Siemens Power on Demand (PoD) Key
# ------- no changes required below this line --------
slurm_hl2hl.py --format STAR-CCM+ > $SLURM_TMPDIR/machinefile
NCORE=$((SLURM_NNODES * SLURM_CPUS_PER_TASK * SLURM_NTASKS_PER_NODE))
if [ -n "$LM_PROJECT" ]; then
  # Siemens PoD license server
  starccm+ -jvmargs -Xmx4G -jvmargs -Djava.io.tmpdir=$SLURM_TMPDIR -batch -power -podkey $LM_PROJECT -np $NCORE -nbuserdir $SLURM_TMPDIR -machinefile $SLURM_TMPDIR/machinefile $JAVA_FILE $SIM_FILE -mpi intel -fabric psm2
else
  # Institutional license server
  starccm+ -jvmargs -Xmx4G -jvmargs -Djava.io.tmpdir=$SLURM_TMPDIR -batch -np $NCORE -nbuserdir $SLURM_TMPDIR -machinefile $SLURM_TMPDIR/machinefile $JAVA_FILE $SIM_FILE -mpi intel -fabric psm2
fi
```

#### Narval

**File: starccm_job.sh**

```bash
#!/bin/bash
#SBATCH --account=def-group   # Specify some account
#SBATCH --time=00-01:00       # Time limit: dd-hh:mm
#SBATCH --nodes=1             # Specify 1 or more nodes
#SBATCH --cpus-per-task=64    # Request all cores per node
#SBATCH --mem=0               # Request all memory per node
#SBATCH --ntasks-per-node=1   # Do not change this value
#module load StdEnv/2020      # Versions < 18.06.006
module load StdEnv/2023
#module load starccm/18.06.006-R8
module load starccm-mixed/18.06.006
SIM_FILE='mysample.sim'       # Specify your input sim filename
#JAVA_FILE='mymacros.java'    # Uncomment to specify an input java filename
# Comment the next line when using an institutional license server
LM_PROJECT='my22digitpodkey' # Specify your Siemens Power on Demand (PoD) Key
# ------- no changes required below this line --------
slurm_hl2hl.py --format STAR-CCM+ > $SLURM_TMPDIR/machinefile
NCORE=$((SLURM_NNODES * SLURM_CPUS_PER_TASK * SLURM_NTASKS_PER_NODE))
if [ -n "$LM_PROJECT" ]; then
  # Siemens PoD license server
  starccm+ -jvmargs -Xmx4G -jvmargs -Djava.io.tmpdir=$SLURM_TMPDIR -batch -power -podkey $LM_PROJECT -np $NCORE -nbuserdir $SLURM_TMPDIR -machinefile $SLURM_TMPDIR/machinefile $JAVA_FILE $SIM_FILE -mpi openmpi
else
  # Institutional license server
  starccm+ -jvmargs -Xmx4G -jvmargs -Djava.io.tmpdir=$SLURM_TMPDIR -batch -np $NCORE -nbuserdir $SLURM_TMPDIR -machinefile $SLURM_TMPDIR/machinefile $JAVA_FILE $SIM_FILE -mpi openmpi
fi
```

#### Niagara

**File: starccm_job.sh**

```bash
#!/bin/bash
#SBATCH --account=def-group   # Specify some account
#SBATCH --time=00-01:00       # Time limit: dd-hh:mm
#SBATCH --nodes=1             # Specify 1 or more nodes
#SBATCH --cpus-per-task=40    # Request all cores per node
#SBATCH --mem=0               # Request all memory per node
#SBATCH --ntasks-per-node=1   # Do not change this value
module load CCEnv
#module load StdEnv/2020      # Versions < 18.06.006
module load StdEnv/2023
#module load starccm/18.06.006-R8
module load starccm-mixed/18.06.006
SIM_FILE='mysample.sim'       # Specify input sim filename
#JAVA_FILE='mymacros.java'    # Uncomment to specify an input java filename
# Comment the next line when using an institutional license server
LM_PROJECT='my22digitpodkey' # Specify your Siemens Power on Demand (PoD) Key
# These settings are used instead of your ~/.licenses/starccm.lic
# (settings shown will use the cd-adapco pod license server)
FLEXPORT=1999                # Specify server static flex port
VENDPORT=2099                # Specify server static vendor port
LICSERVER=flex.cd-adapco.com # Specify license server hostname
# ------- no changes required below this line --------
export CDLMD_LICENSE_FILE="$FLEXPORT@127.0.0.1"
ssh nia-gw -L $FLEXPORT:$LICSERVER:$FLEXPORT -L $VENDPORT:$LICSERVER:$VENDPORT -N -f

slurm_hl2hl.py --format STAR-CCM+ > $SLURM_TMPDIR/machinefile
NCORE=$((SLURM_NNODES * SLURM_CPUS_PER_TASK * SLURM_NTASKS_PER_NODE))
# Workaround for license failures:
# until the exit status is equal to 0, we try to get Star-CCM+ to start (here, for at least 5 times).
i=1
RET=-1
while [ $i -le 5 ] && [ $RET -ne 0 ]; do
  [ $i -eq 1 ] || sleep 5
  echo "Attempt number: " $I
  if [ -n "$LM_PROJECT" ]; then
    # Siemens PoD license server
    starccm+ -jvmargs -Xmx4G -jvmargs -Djava.io.tmpdir=$SLURM_TMPDIR -batch -power -podkey $LM_PROJECT -np $NCORE -nbuserdir $SLURM_TMPDIR -machinefile $SLURM_TMPDIR/machinefile $JAVA_FILE $SIM_FILE
  else
    # Institutional license server
    starccm+ -jvmargs -Xmx4G -jvmargs -Djava.io.tmpdir=$SLURM_TMPDIR -batch -np $NCORE -nbuserdir $SLURM_TMPDIR -machinefile $SLURM_TMPDIR/machinefile $JAVA_FILE $SIM_FILE
  fi
  RET=$?
  i=$((i + 1))
done
exit $RET
```

## Remote visualization

### Preparation

To set up your account for remote visualization:

1.  Create `~/.licenses/starccm.lic` as described above.
2.  Users with a POD license should also set `export LM_PROJECT='CD-ADAPCO PROJECT ID'` and add `-power` to the command-line options shown below.


### Compute nodes

Connect with TigerVNC and open a terminal window.

#### STAR-CCM+ 15.04.010 (or newer versions)

```bash
module load StdEnv/2020
module load starccm-mixed/17.02.007
# OR
module load starccm/17.02.007-R8
starccm+
```

#### STAR-CCM+ 14.06.010, 14.04.013, 14.02.012

```bash
module load StdEnv/2016
module load starccm-mixed/14.06.010
# OR
module load starccm/14.06.010-R8
starccm+
```

#### STAR-CCM+ 13.06.012 (or older versions)

```bash
module load StdEnv/2016
module load starccm-mixed/13.06.012
# OR
module load starccm/13.06.012-R8
starccm+ -mesa
```

### VDI nodes

Connect to `gra-vdi.alliancecan.ca` with TigerVNC and log in. Once the Remote Desktop appears, click `Applications -> Systems Tools -> Mate Terminal` to open a terminal window and then specify which STAR-CCM+ version to load as shown below. After loading a `StdEnv`, use the `module avail starccm-mixed` command to see available STAR-CCM+ versions. Currently, only the MESA implementation of OpenGL is usable on `gra-vdi` with STAR-CCM+ due to virtualgl issues that would otherwise provide local GPU hardware acceleration for OpenGL-driven graphics.

#### STAR-CCM+ 18.04.008 (or newer versions)

```bash
module load CcEnv StdEnv/2023
module load starccm-mixed/18.04.008
# OR
module load starccm/18.04.008-R8
starccm+ -rr server
```

#### STAR-CCM+ 15.04.010 --> 18.02.008 (version range)

```bash
module load CcEnv StdEnv/2020
module load starccm-mixed/15.04.010
# OR
module load starccm/15.04.010-R8
starccm+ -mesa
```

#### STAR-CCM+ 13.06.012 (or older versions)

```bash
module load CcEnv StdEnv/2016
module load starccm-mixed/13.06.012
# OR
module load starccm/13.06.012-R8
starccm+ -mesa
```

**(Retrieved from [https://docs.alliancecan.ca/mediawiki/index.php?title=Star-CCM%2B&oldid=178177](https://docs.alliancecan.ca/mediawiki/index.php?title=Star-CCM%2B&oldid=178177))**
