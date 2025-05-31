# AMS (Amsterdam Modeling Suite)

This page is a translated version of the page AMS and the translation is 100% complete.

Other languages: [English](link-to-english-page), [français](current-page-url)


## Introduction

AMS (Amsterdam Modeling Suite), is the new name for ADF (Amsterdam Density Functional) and is part of the SCM Software for Chemistry and Materials suite. AMS offers high-performance tools for research in computational chemistry, particularly in the fields of (homogeneous and heterogeneous) catalysis, inorganic chemistry, heavy element chemistry, biochemistry, and various types of spectroscopy.

All products of the SCM module are available:

* ADF
* ADF-GUI
* BAND
* BAND-GUI
* DFTB
* ReaxFF
* COSMO-RS
* QE-GUI
* NBO6


## Using AMS on Graham

The `ams` module is only installed on Graham due to licensing restrictions. SHARCNET owns this license, which is reserved for university computing centers; this license cannot be used for consulting services or any other commercial use. To see the available versions, run the command:

```bash
[name@server $] module spider ams
```

For commands related to modules, see [Using Modules](link-to-modules-page).


### Submitting a Job

Jobs submitted on Graham are scheduled by Slurm; for details, see [Running Jobs](link-to-running-jobs-page).


#### Examples of scripts for an adf task

The `H2O_adf.sh` script uses an entire node.

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

The input file below is used in the script.

**File: H2O_adf.run**

```bash
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


#### Examples of scripts for a band task

**File: SnO_EFG_band.run**

```bash
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

The following script is similar to `H2O_adf.sh`, except that it does not use an entire node.

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


### Remarks

The input file for AMS is different from that for ADF; the previous input file for ADF will not work with the new AMS. You will find examples in `/opt/software/ams/2020.102/examples/`.

Except for the `.log` output file, all files are saved in the `AMS_JOBNAME.results` subdirectory. If `AMS_JOBNAME` is not defined in the `.run` input file, the default name will be `ams.results`.

The checkpoint file name is `ams.rkf` rather than `TAPE13` in previous ADF versions.

See the tutorial [An Update on ADF/AMS software on Graham](link-to-tutorial).

For more information, see [SCM Support](link-to-scm-support).


## Using AMS-GUI

With applications like AMS-GUI, X11 redirection via an SSH connection takes a long time to produce renderings. We recommend connecting with VNC.


### Graham

On a Graham compute node, AMS can be used interactively in graphical mode with TigerVNC for a maximum duration of 3 hours.

Install a TigerVNC client on your computer.

Connect to a compute node with `vncviewer`.

```bash
module load ams
amsinput
```


### Gra-vdi

On gra-vdi, AMS can be used interactively in graphical mode with TigerVNC, without a time limit.

Install a TigerVNC client on your computer.

Connect to `gra-vdi.computecanada.ca` with `vncviewer`.

```bash
module load SnEnv
module load clumod
module load ams
amsinput
```

Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=AMS/fr&oldid=159246](https://docs.alliancecan.ca/mediawiki/index.php?title=AMS/fr&oldid=159246)"
