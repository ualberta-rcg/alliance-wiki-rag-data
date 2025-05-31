# AMS (Amsterdam Modeling Suite) on Graham Cluster

## Introduction

AMS (Amsterdam Modeling Suite), originally named ADF (Amsterdam Density Functional), is the SCM Software for Chemistry and Materials. AMS offers powerful computational chemistry tools for many research areas such as homogeneous and heterogeneous catalysis, inorganic chemistry, heavy element chemistry, various types of spectroscopy, and biochemistry.

The full SCM module products are available:

* ADF
* ADF-GUI
* BAND
* BAND-GUI
* DFTB
* ReaxFF
* COSMO-RS
* QE-GUI
* NBO6

## Running AMS on Graham

The `ams` module is installed on Graham only due to license restrictions. The license is an Academic Computing Center license owned by SHARCNET. You may not use the Software for consulting services and for purposes that have a commercial nature. To check what versions are available, use the `module spider` command as follows:

```bash
[name@server $] module spider ams
```

For module commands, please see [Using modules](link-to-modules-doc).


### Job Submission

Graham uses the Slurm scheduler; for details about submitting jobs, see [Running jobs](link-to-running-jobs-doc).


### Example Scripts for an ADF Job

This `H2O_adf.sh` script is for a whole-node job.

**File: H2O_adf.sh**

```bash
#!/bin/bash
#SBATCH --account=def-pi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32          # 1 node with all 32 cpus, MPI job
#SBATCH --mem=0                       # request all memory on node
#SBATCH --time=00-01:00               # time (DD-HH:MM)
#SBATCH --output=H2O_adf-%j.log       # output .log file
module unload openmpi
module load ams/2024.102
export SCM_TMPDIR=$SLURM_TMPDIR
# use the local disk
bash H2O_adf.run
# run the input script
```

This is the input file used in the script:

**File: H2O_adf.run**

```sh
#!/bin/sh
# This is a shell script for AMS
# You should use '$AMSBIN/ams' instead of '$ADFBIN/adf'

AMS_JOBNAME=H2O_adf $AMSBIN/ams <<eor
   # Input options for the AMS driver:
   System
      Atoms
         O             0.000000     0.000000     0.000000
         H             0.000000    -0.689440    -0.578509
         H             0.000000     0.689440    -0.578509
      End
   End
   Task GeometryOptimization
   GeometryOptimization
      Convergence gradients=1e-4
   End

   # The input options for ADF, which are described in this manual,
   # should be specified in the 'Engine ADF' block:

   Engine ADF
      Basis
         Type TZP
      End
      XC
         GGA PBE
      End
   EndEngine
eor
```


### Example Scripts for a Band Job

**File: SnO_EFG_band.run**

```sh
#!/bin/sh
# The calculation of the electric field gradient is invoked by the EFG key block
# Since Sn is quite an heavy atom we use the scalar relativistic option.
$AMSBIN/ams <<eor
Task SinglePoint
System
FractionalCoords True
Lattice
3.8029  0.0  0.0
0.0  3.8029  0.0
0.0  0.0  4.8382
End
Atoms
O   0.0  0.0  0.0
O   0.5  0.5  0.0
Sn  0.0  0.5  0.2369
Sn  0.5  0.0 -0.2369
End
End
Engine Band
Title SnO EFG
NumericalQuality Basic      ! Only for speed
Tails bas=1e-8              ! Only for reproducibility with nr. of cores
! useful for Moessbauer spectroscopy: density and coulomb pot. at nuclei
PropertiesAtNuclei
End
EFG
Enabled True
End
Basis
Type DZ
Core none
End
EndEngine
eor
```

The following slurm script is similar to the one used for a single adf run (H2O_adf.sh), except it's not a whole-node job.

**File: SnO_EFG_band.sh**

```bash
#!/bin/bash
#SBATCH --account=def-pi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16     # a 16 cpus MPI job
#SBATCH --mem-per-cpu=3G         # memory; 3G per cpu in this example
#SBATCH --time=00-10:00          # time (DD-HH:MM)
#SBATCH --output=SnO_EFG_band-%j.log
module unload openmpi
module load ams/2024.102
export SCM_TMPDIR=$SLURM_TMPDIR
# use the local disk
bash SnO_EFG_band.run
# run the input file
```

### Notes

* The input for AMS is different from ADF; the previous ADF input file will not run for the new AMS. Some examples can be found in `/opt/software/ams/2020.102/examples/`.
* Except for the output `.log` file, other files are all saved in a subdirectory `AMS_JOBNAME.results`. If `AMS_JOBNAME` is not defined in the input `.run` file, the default name is `ams.results`.
* The restart file name is `ams.rkf` instead of `TAPE13` in previous ADF versions.
* You can watch a recorded webinar/tutorial: [An Update on ADF/AMS software on Graham](link-to-webinar).
* For more usage information, please check the manuals in [SCM Support](link-to-scm-support).


## Running AMS-GUI

Rendering over an SSH connection with X11 forwarding is very slow for GUI applications such as AMS-GUI. We recommend you use VNC to connect if you will be running AMS-GUI.


### Graham

AMS can be run interactively in graphical mode on a Graham compute node (3hr time limit) over TigerVNC with these steps:

1. Install a TigerVNC client on your desktop.
2. Connect to a compute node with `vncviewer`.
3. `module load ams`
4. `amsinput`


### Gra-vdi

AMS can be run interactively in graphical mode on gra-vdi (no connection time limit) over TigerVNC with these steps:

1. Install a TigerVNC client on your desktop.
2. Connect to `gra-vdi.computecanada.ca` with `vncviewer`.
3. `module load SnEnv`
4. `module load clumod`
5. `module load ams`
6. `amsinput`

**(Remember to replace `link-to-modules-doc`, `link-to-running-jobs-doc`, `link-to-webinar`, and `link-to-scm-support` with the actual links.)**
