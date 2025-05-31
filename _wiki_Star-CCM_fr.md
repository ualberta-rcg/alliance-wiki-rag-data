# STAR-CCM+

STAR-CCM+ is a simulation software suite used in several engineering specialties. It allows modeling in various fields including acoustics, fluid dynamics, heat transfer, rheology, multiphase flow, particle flow, solid mechanics, reactive fluids, electrochemistry, and electromagnetism.

## License Limits

STAR-CCM+ binaries are installed on our servers, but we do not have a license for general use; therefore, you must have your own license.

You can purchase a Power On Demand (POD) license directly from Siemens.  Alternatively, you can use a local license hosted by your institution, provided that the firewall allows the cluster where the jobs will be executed to access it.

### Configuring Your Account

To configure your account to use a license server, create the file `$HOME/.licenses/starccm.lic` as follows:

**File: starccm.lic**

```
SERVER <server> ANY <port> USE_SERVER
```

where `server` and `port` are replaced by the hostname (or IP address) and the static port of the license server provider, respectively.

### File for a POD License

If you have purchased a POD license from Siemens, you can specify the license file in a text file as shown below. This works on all clusters except Niagara.

**File: starccm.lic**

```
SERVER flex.cd-adapco.com ANY 1999 USE_SERVER
```

In your scheduler script, set `LM_PROJECT` to your CD-ADAPCO PROJECT ID. Note that it is no longer necessary to manually configure `CDLMD_LICENSE_FILE="<port>@<server>"` in the job submission script.


## Submitting Batch Jobs on Our Clusters

Select one of the available modules, depending on your needs:

*   `starccm` for double-precision format:  `module load starccm/19.04.007-R8`
*   `starccm-mixed` for mixed-precision format: `module load starccm-mixed/19.04.007`

When submitting jobs to a cluster for the first time, you will need to configure your environment for license usage. If you are using Siemens' remote pay-on-usage license server, create the file `~/.licenses/starccm.lic` as described above in "File for a POD License"; this should work immediately. However, if you are using a license server from your institution, first create the file `~/.licenses/starccm.lic` and submit a support request to technical support. We will help you coordinate the necessary network firewall changes to access it (assuming the server has never been configured to communicate via the Alliance cluster you want to use). If you still encounter problems getting the license to work, try deleting or renaming the file `~/.flexlmrc` as previous license server paths and/or settings might be stored there. Note that output files from already executed jobs may accumulate in hidden directories named `.star-version_number` and thus consume your quota. These can be periodically removed by periodically running `rm -ri ~/.starccm*` and answering "yes" to the prompt.


### Scheduler Scripts

**Béluga, Cedar, Graham, Narval, Niagara**

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
export STARCCM_TMP="${SCRATCH}/.starccm-${EBVERSIONSTARCCM}"
mkdir -p "$STARCCM_TMP"
slurm_hl2hl.py --format STAR-CCM+ > machinefile-$SLURM_JOB_ID
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

**Another example for Béluga, Cedar, Graham, Narval:**

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
export STARCCM_TMP="${SCRATCH}/.starccm-${EBVERSIONSTARCCM}"
mkdir -p "$STARCCM_TMP"
slurm_hl2hl.py --format STAR-CCM+ > machinefile-$SLURM_JOB_ID
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

**Another example for Béluga, Cedar, Graham, Narval:**

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

**Another example for Niagara:**

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
LICSERVER=flex.cd-adapco.com  # Specify license server hostname
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


## Remote Visualization

### Preparation

Configure your account for remote visualization:

Create the license file `~/.licenses/starccm.lic` as described above.

If you have a Power-on-demand (POD) license, configure `export LM_PROJECT='CD-ADAPCO PROJECT ID'` and add `-power` to the other command-line options, as shown below.


### Compute Nodes

Connect to a node with TigerVNC and open a terminal window, then:

**STAR-CCM+ 15.04.010 (or newer versions)**

```bash
module load StdEnv/2020
module load starccm-mixed/17.02.007
# OR
module load starccm/16.04.007-R8
starccm+
```

**STAR-CCM+ 14.06.010, 14.04.013, 14.02.012**

```bash
module load StdEnv/2016
module load starccm-mixed/14.06.010
# OR
module load starccm/14.06.010-R8
starccm+
```

**STAR-CCM+ 13.06.012 (or older versions)**

```bash
module load StdEnv/2016
module load starccm-mixed/13.06.012
# OR
module load starccm/13.06.012-R8
starccm+ -mesa
```

### VDI Nodes

Connect to `gra-vdi` with TigerVNC and log in. When the remote desktop is displayed, click "open a terminal window" (Applications-->Systems Tools-->Mate Terminal) to open a terminal window and specify the Star-CCM version you want to load (see below). If you have already loaded a StdEnv, you can display the available versions with the command `module avail starccm-mixed`. Currently, only the MESA implementation of OpenGL can be used on `gra-vdi` with starccm due to problems with virtualgl which provides local GPU acceleration for OpenGL graphics.

**STAR-CCM+ 18.04.008 (or newer versions)**

```bash
module load CcEnv StdEnv/2023
module load starccm-mixed/18.04.008
# OR
module load starccm/18.04.008-R8
starccm+ -rr server
```

**STAR-CCM+ 15.04.010 --> 18.02.008 (from version X to version Y)**

```bash
module load CcEnv StdEnv/2020
module load starccm-mixed/15.04.010
# OR
module load starccm/15.04.010-R8
starccm+ -mesa
```

**STAR-CCM+ 13.06.012 (or older versions)**

```bash
module load CcEnv StdEnv/2016
module load starccm-mixed/13.06.012
# OR
module load starccm/13.06.012-R8
starccm+ -mesa
```
