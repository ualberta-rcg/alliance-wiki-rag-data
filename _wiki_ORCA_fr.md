# ORCA

This page is a translated version of the page ORCA and the translation is 100% complete.

Other languages: [English](link-to-english-page), [français](current-page-url)


## Introduction

ORCA is a general-purpose quantum chemistry software package offering flexibility, efficiency, and ease of use.  It is particularly useful for modeling the spectroscopic properties of molecules with open-shell valence configurations. ORCA allows the use of a large number of methods including Density Functional Theory (DFT) and other semi-empirical methods as well as single and multi-reference *ab initio* correlation methods. It also handles environmental and relativistic effects.


## Usage Rights

To use the pre-built ORCA executables:

1. Fill out the registration form found at [https://orcaforum.kofo.mpg.de/](https://orcaforum.kofo.mpg.de/).
2. You will receive a first email to confirm your email address and activate your account; follow the instructions in this email.
3. Once your registration is complete, you will receive a second email with the message "registration for ORCA download and usage has been completed".
4. Send a copy of the second email to technical support.


## Versions

### ORCA 6

The module `orca/6.0.1` is available in the `StdEnv/2023` environment; to load it, run:

```bash
module load StdEnv/2023  gcc/12.3  openmpi/4.1.5 orca/6.0.1
```

There is also the module `orca/6.0.0`. However, the newer version `orca/6.0.1` fixed bugs present in version `6.0.0`.

**Note:** This version of ORCA includes xtb 6.7.1.


### ORCA 5

Versions 5.0.1 to 5.0.3 contained bugs that were eliminated in version 5.0.4, notably a problem affecting D4 dispersion gradients. We therefore recommend using version 5.0.4 instead of older 5.0.x versions. Versions 5.0.1, 5.0.2, and 5.0.3 are in our software stack but might eventually be removed.

Load version 5.0.4 with:

```bash
module load StdEnv/2020  gcc/10.3.0  openmpi/4.1.1 orca/5.0.4
```


### ORCA 4

Load version 4.2.1 with:

```bash
module load StdEnv/2020  gcc/9.3.0  openmpi/4.0.3 orca/4.2.1
```

or

```bash
module load nixpkgs/16.09  gcc/7.3.0  openmpi/3.1.4 orca/4.2.1
```


## Input File Configuration

In addition to the keywords required to run a simulation, ensure you configure the following parameters:

* CPU quantity
* `maxcore`


## Usage

To see available versions, run:

```bash
module spider orca
```

For details related to a specific module (including directives for the order in which to load required modules), use the full module name, for example:

```bash
module spider orca/4.0.1.2
```

For general directives, see [Using Modules](link-to-using-modules-page).


### Submitting Jobs

For general directives, see [Running Jobs](link-to-running-jobs-page).


### Notes

If some ORCA executables have problems with MPI, you can try setting the following variables:

```bash
export OMPI_MCA_mtl='^mxm'
export OMPI_MCA_pml='^yalla'
```

The following script uses MPI. Note that, unlike most MPI programs, ORCA is not started with a parallel command such as `mpirun` or `srun`, but requires the full path to the program, which is indicated by `$EBROOTORCA`.

**File:** `run_orca.sh`

```bash
#!/bin/bash
#SBATCH --account=def-youPIs
#SBATCH --ntasks=8                 # cpus, the nprocs defined in the input file
#SBATCH --mem-per-cpu=3G           # memory per cpu
#SBATCH --time=00-03:00            # time (DD-HH:MM)
#SBATCH --output=benzene.log       # output .log file

module load StdEnv/2020 gcc/9.3.0 openmpi/4.0.3
module load orca/4.2.1
$EBROOTORCA/orca benzene.inp
```

Here is an example of the input file `benzene.inp`:

**File:** `benzene.inp`

```
# Benzene RHF Opt Calculation
%pal nprocs 8 end
! RHF TightSCF PModel
! opt

* xyz 0 1
     C    0.000000000000     1.398696930758     0.000000000000
     C    0.000000000000    -1.398696930758     0.000000000000
     C    1.211265339156     0.699329968382     0.000000000000
     C    1.211265339156    -0.699329968382     0.000000000000
     C   -1.211265339156     0.699329968382     0.000000000000
     C   -1.211265339156    -0.699329968382     0.000000000000
     H    0.000000000000     2.491406946734     0.000000000000
     H    0.000000000000    -2.491406946734     0.000000000000
     H    2.157597486829     1.245660462400     0.000000000000
     H    2.157597486829    -1.245660462400     0.000000000000
     H   -2.157597486829     1.245660462400     0.000000000000
     H   -2.157597486829    -1.245660462400     0.000000000000
*
```


### Notes

For the program to run efficiently and use all the resources or cores required by your job, add the line `%pal nprocs <ncores> end` to the output file, as in the example above. Replace `<ncores>` with the number of cores you specified in your script.

If you want to restart a calculation, delete the `*.hostnames` file (e.g., `benzene.hostnames` in the example above) before submitting the next job; otherwise, the job will likely fail, producing the error message "All nodes which are allocated for this job are already filled."


### (2019-09-06) Temporary Fix Regarding OpenMPI Version Inconsistency

In some types of calculations (especially DLPNO-STEOM-CCSD), critical errors may occur. This might happen if you use an older version of OpenMPI (e.g., 3.1.2 as suggested by `module` for orca/4.1.0 and 4.2.0) than the officially recommended one (3.1.3 for orca/4.1.0 and 3.1.4 for orca/4.2.0). To solve this, you can customize the OpenMPI version.

The following two commands customize openmpi/3.1.4 for orca/4.2.0:

```bash
module load gcc/7.3.0
eb OpenMPI-3.1.2-GCC-7.3.0.eb --try-software-version=3.1.4
```

Once this is done, load openmpi with:

```bash
module load openmpi/3.1.4
```

You can now manually install the orca/4.2.0 binaries from the official forum in the `/home` directory, after registering in the official ORCA forum and obtaining access to the ORCA application on our clusters.

**Other notes from the author:** This fix can be applied while waiting for the official update of OpenMPI on our clusters. Once this update has been made, remember to remove the manually installed binaries. The compilation command does not seem to apply to openmpi/2.1.x.


## Using NBO

You must have access to NBO to use it with ORCA. NBO is not a separate module on our clusters, but it is accessible via the Gaussian modules installed on Cedar and Graham. To use NBO with ORCA, you need access to both ORCA and Gaussian.


### Example Script

The input file name (in the next example `orca_input.inp`) must contain the keyword `NBO`.

**File:** `run_orca-nbo.sh`

```bash
#!/bin/bash
#SBATCH --account=def-youPIs
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem-per-cpu=4000
#SBATCH --time=0-3:00:00

# Load the modules:
module load StdEnv/2020 gcc/10.3.0 openmpi/4.1.1 orca/5.0.4
module load gaussian/g16.c01

export GENEXE=`which gennbo.i4.exe`
export NBOEXE=`which nbo7.i4.exe`

$EBROOTORCA/orca orca_input.inp > orca_output.out
```


## References

* [ORCA tutorials](link-to-orca-tutorials)
* [ORCA Forum](https://orcaforum.kofo.mpg.de/)


**(Retrieved from "https://docs.alliancecan.ca/mediawiki/index.php?title=ORCA/fr&oldid=174100")**
