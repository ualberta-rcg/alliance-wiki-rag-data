# ADF (now AMS)

**Note:** The ADF suite has been renamed AMS since the 2020 version. This new version includes significant changes, particularly in input and output formats. For more information, see [AMS](link_to_AMS_page).


The SCM (Software for Chemistry and Materials) software suite, originally the ADF suite for Amsterdam Density Functional, offers high-performance applications for research in computational chemistry, particularly in the fields of (homogeneous and heterogeneous) catalysis, inorganic chemistry, heavy element chemistry, biochemistry, and various types of spectroscopy.

The following products are available:

* ADF
* ADF-GUI
* BAND
* BAND-GUI
* DFTB
* ReaxFF
* COSMO-RS
* QE-GUI
* NBO6


## Using SCM on Graham

The `adf` module is only installed on Graham due to licensing restrictions. To see the available versions, run the command:

```bash
[name@server $] module spider adf
```

For commands related to modules, see [Using Modules](link_to_modules_page).


### Submitting a Job

Jobs submitted on Graham are scheduled by Slurm; for details, see [Running Jobs](link_to_running_jobs_page).


#### Single Job

The following script uses an entire node; the second to last line loads version 2019.305 and the last line calls ADF directly.

**File: mysub.sh**

```bash
#!/bin/bash
#SBATCH --nodes=1 --ntasks-per-node=32  # 1 node with 32 cpus, you can modify it
#SBATCH --mem=0                         # request all memory on node
#SBATCH --time=00-03:00                 # time (DD-HH:MM)
#SBATCH --output=adf_test-%j.log        # output file
module unload openmpi
module load adf/2019.305
ADF adf_test.inp
```

The input file below is used in the script.

**File: adf_test.inp**

```
Title WATER Geometry Optimization with Delocalized Coordinates

 Atoms
    O             0.000000     0.000000     0.000000
    H             0.000000    -0.689440    -0.578509
    H             0.000000     0.689440    -0.578509
 End

 Basis
 Type TZP
 Core Small
 End

 Geometry
  Optim Deloc
  Converge 0.0000001
 End

 End Input
```


#### Multiple Jobs with ADF or BAND

Several calculations can be grouped in the same job with a script similar to this one:

**File: GO_H2O.run**

```bash
#!/bin/bash
if test -z "$SCM_TESTOUTPUT"; then SCM_TESTOUTPUT=GO_H2O.out; fi
$ADFBIN/adf << eor > $SCM_TESTOUTPUT
Title WATER Geometry Optimization with Delocalized Coordinates
Atoms
O             0.000000     0.000000     0.000000
H             0.000000    -0.689440    -0.578509
H             0.000000     0.689440    -0.578509
End
Basis
Type TZP
Core Small
End
Geometry
Optim Deloc
Converge 0.0000001
End
End Input
eor
rm TAPE21 logfile
$ADFBIN/adf << eor >> $SCM_TESTOUTPUT
Title WATER Geometry Optimization in Cartesians with new optimizer
Atoms
O             0.000000     0.000000     0.000000
H             0.000000    -0.689440    -0.578509
H             0.000000     0.689440    -0.578509
End
Basis
Type TZP
Core Small
End
Geometry
Optim Cartesian
Branch New
Converge 0.0000001
End
End Input
eor
rm TAPE21 logfile
$ADFBIN/adf << eor >> $SCM_TESTOUTPUT
Title WATER Geometry Optimization with Internal Coordinates
Atoms    Z-Matrix
1. O   0 0 0
2. H   1 0 0   rOH
3. H   1 2 0   rOH  theta
End
Basis
Type TZP
Core Small
End
GeoVar
rOH=0.9
theta=100
End
Geometry
Converge 0.0000001
End
End Input
eor
rm TAPE21 logfile
$ADFBIN/adf << eor >> $SCM_TESTOUTPUT
Title WATER   optimization with (partial) specification of Hessian
Atoms    Z-Matrix
1. O   0 0 0
2. H   1 0 0   rOH
3. H   1 2 0   rOH  theta
End
GeoVar
rOH=0.9
theta=100
End
HessDiag  rad=1.0  ang=0.1
Fragments
H   t21.H
O   t21.O
End
Geometry
Converge 0.0000001
End
End Input
eor
rm TAPE21 logfile
$ADFBIN/adf << eor >> $SCM_TESTOUTPUT
Title WATER Geometry Optimization in Cartesians
Geometry
Optim Cartesian
Converge 0.0000001
End
Define
rOH=0.9
theta=100
End
Atoms    Z-Matrix
1. O   0 0 0
2. H   1 0 0   rOH
3. H   1 2 0   rOH theta
End
Fragments
H   t21.H
O   t21.O
End
End Input
eor
mv TAPE21 H2O.t21
```

The following script is identical to the one used for a single job (mysub.sh), except for the last line which calls the GO_H2O.run script instead of calling ADF directly.

**File: GO_H2O.sh**

```bash
#!/bin/bash
#SBATCH --nodes=1 --ntasks-per-node=32  # 1 node with 32 cpus, you can modify it
#SBATCH --mem=0                         # request all memory on node
#SBATCH --time=00-03:00                 # time (DD-HH:MM)
#SBATCH --output=GO_H2O_%j.log          # output file
module unload openmpi
module load adf/2019.305
bash GO_H2O.run # run the shell script
```


### Examples

For input/output examples for ADF, see on Graham `/home/jemmyhu/tests/test_ADF/2019.305/test_adf/`

For examples of .inp and .sh files with BAND, see on Graham `/home/jemmyhu/tests/test_ADF/2019.305/test_band`


## Using SCM-GUI

With applications like ADF-GUI, X11 forwarding via an SSH connection takes a long time to produce renderings. We recommend connecting with VNC.


### Graham

On a Graham compute node, ADF can be used interactively in graphical mode with TigerVNC for a maximum duration of 3 hours.

Install a TigerVNC client on your computer. Connect to a compute node with `vncviewer`.

```bash
module load adf
adfinput
```


### Gra-vdi

On gra-vdi, ADF can be used interactively in graphical mode, without a time limit.

Install a TigerVNC client on your computer. Connect to `gra-vdi.computecanada.ca` with `vncviewer`.

```bash
module load clumod
module load adf
adfinput
```

See [this tutorial](link_to_tutorial) on how to use ADF-GUI with TigerVNC on gra-vdi.


### Using ADF-GUI Locally

SCM offers a separate license to use ADF-GUI on a local desktop computer; to acquire your own license, contact `license@scm.com`.

