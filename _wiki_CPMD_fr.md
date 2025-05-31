# CPMD

CPMD is an *ab initio* molecular dynamics simulation program based on Density Functional Theory (DFT) for plane waves/pseudopotentials.

## License Limitations

In the past, you had to register and wait for confirmation from the development team, but registration is no longer required. However, the modules installed on our clusters are protected by a POSIX group.

To use CPMD on our clusters, contact technical support to be added to the POSIX group.

## Module

To load the module, run:

```bash
module load StdEnv/2020
module load intel/2020.1.217 openmpi/4.0.3 cpmd/4.3
```

## Installing CPMD Locally

The response from the CPMD administrators can take a few weeks, even months. As a registered user, you have access to the CPMD source files; you can therefore build the application in your `/home` directory with our EasyBuild environment using the same recipe we use for a central installation.

For CPMD 4.3 in your account on one of our clusters, follow these instructions:

First, create a local directory:

```bash
$ mkdir -p ~/.local/easybuild/sources/c/CPMD
```

Place the tarballs and patches in this directory:

```bash
$ ls -al ~/.local/easybuild/sources/c/CPMD
cpmd2cube.tar.gz
cpmd2xyz-scripts.tar.gz
cpmd-v4.3.tar.gz
fourier.tar.gz
patch.to.4612
patch.to.4615
patch.to.4616
patch.to.4621
patch.to.4624
patch.to.4627
```

Then run the EasyBuild command:

```bash
$ eb CPMD-4.3-iomkl-2020a.eb --rebuild
```

The `--rebuild` option ensures that EasyBuild uses the installation located in your `/home` directory rather than the central location.

Once the application is installed, log out of the cluster and log back in again.  The command `module load cpmd` will find the application in your `/home` directory.

```bash
$ module load StdEnv/2020
$ module load intel/2020.1.217 openmpi/4.0.3 cpmd/4.3
$ which cpmd.x
~/.local/easybuild/software/2020/avx2/MPI/intel2020/openmpi4/cpmd/4.3/bin/cpmd.x
```

You can now use it in a job submission script.

## Script Examples

To run a job, you need to set up an input file and access to the pseudopotentials.

If the input file and pseudopotentials are in the same directory, the following command runs the program in parallel:

```bash
srun cpmd.x <input files> > <output file>
```

(as in script 1)

If the pseudopotentials are in a different directory, the command is:

```bash
srun cpmd.x <input files> <path to pseudo potentials location> > <output file>
```

(as in script 2)


### Input File

**File: 1-h2-wave.inp**

```
&INFO
isolated hydrogen molecule.
single point calculation.
&END

&CPMD
 OPTIMIZE WAVEFUNCTION
 CONVERGENCE ORBITALS
  1.0d-7
 CENTER MOLECULE ON
 PRINT FORCES ON
&END

&SYSTEM
 SYMMETRY
  1
 ANGSTROM
 CELL
  8.00 1.0 1.0  0.0  0.0  0.0
 CUTOFF
  70.0
&END

&DFT
 FUNCTIONAL LDA
&END

&ATOMS
*H_MT_LDA.psp
 LMAX=S
  2
 4.371   4.000   4.000
 3.629   4.000   4.000
&END
```

### Script 1

**File: run-cpmd.sh**

```bash
#!/bin/bash
#SBATCH --account=def-someacct
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=2500M
#SBATCH --time=0-1:00
# Load the modules:
module load StdEnv/2020
module load intel/2020.1.217 openmpi/4.0.3 cpmd/4.3
echo "Starting run at: `date`"
CPMD_INPUT="1-h2-wave.inp"
CPMD_OUTPUT="1-h2-wave_output.txt"
srun cpmd.x ${CPMD_INPUT} > ${CPMD_OUTPUT}
echo "Program finished with exit code $? at: `date`"
```

### Script 2

**File: run-cpmd.sh**

```bash
#!/bin/bash
#SBATCH --account=def-someacct
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=2500M
#SBATCH --time=0-1:00
# Load the modules:
module load StdEnv/2020
module load intel/2020.1.217 openmpi/4.0.3 cpmd/4.3
echo "Starting run at: `date`"
CPMD_INPUT="1-h2-wave.inp"
CPMD_OUTPUT="1-h2-wave_output.txt"
PP_PATH=<path to the location of pseudo-potentials>

srun cpmd.x ${CPMD_INPUT} ${PP_PATH} > ${CPMD_OUTPUT}
echo "Program finished with exit code $? at: `date`"
```

## Reference

Website


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=CPMD/fr&oldid=153167](https://docs.alliancecan.ca/mediawiki/index.php?title=CPMD/fr&oldid=153167)"
