# OpenFOAM CFD Toolbox

The OpenFOAM (Open Field Operation and Manipulation) CFD Toolbox is a free, open-source software package for computational fluid dynamics. OpenFOAM has an extensive range of features to solve anything from complex fluid flows involving chemical reactions, turbulence, and heat transfer, to solid dynamics and electromagnetics.

## Module Files

To load the recent version, run:

```bash
module load openfoam
```

The OpenFOAM development community consists of:

*   The OpenFOAM Foundation Ltd., with websites [openfoam.org](openfoam.org) and [cfd.direct](cfd.direct)
*   OpenCFD Ltd., with website [openfoam.com](openfoam.com)

Up to version 2.3.1, released in December 2014, the release histories appear to be the same. On our clusters, module names after 2.3.1 which begin with "v" are derived from the .com branch (for example, `openfoam/v1706`); those beginning with a digit are derived from the .org branch (for example, `openfoam/4.1`).

See [Using modules](link-to-modules-documentation-if-available) for more on module commands.


## Documentation

*   [OpenFOAM.com documentation](link-to-openfoam-com-docs-if-available)
*   [CFD Direct user guide](link-to-cfd-direct-guide-if-available)


## Usage

OpenFOAM requires substantial preparation of your environment. In order to run OpenFOAM commands (such as `paraFoam`, `blockMesh`, etc.), you must load a module file.

Here is an example of a serial submission script for OpenFOAM 5.0:

**File:** `submit.sh`

```bash
#!/bin/bash
#SBATCH --time=00:01:00
#SBATCH --account=def-someuser
module purge
module load openfoam/5.0

blockMesh
icoFoam
```

Here is an example of a parallel submission script:

**File:** `submit.sh`

```bash
#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --ntasks=4               # number of MPI processes
#SBATCH --mem-per-cpu=1024M      # memory; default unit is megabytes
#SBATCH --time=0-00:10           # time (DD-HH:MM)
module purge
module load openfoam/5.0

blockMesh
setFields
decomposePar
srun interFoam -parallel
```

Mesh preparation (`blockMesh`) may be fast enough to be done at the command line (see [Running jobs](link-to-running-jobs-documentation-if-available)). The solver (`icoFoam` and others) is usually the most expensive step and should always be submitted as a Slurm job except in very small test cases or tutorials.


## Segfaults with OpenMPI 3.1.2

Users have reported random segfaults on Cedar when using OpenFOAM versions compiled for OpenMPI 3.1.2 in single-node jobs (shared memory communication). These issues seem not to happen with other versions of OpenMPI. If you experience such problems, first try to use an OpenMPI 2.1.1-based toolchain. For example:

```bash
module load gcc/5.4.0
module load openmpi/2.1.1
module load openfoam/7
```


## Performance

OpenFOAM can emit a lot of debugging information in very frequent small writes (e.g., hundreds per second). This may lead to poor performance on our shared filesystems. If you are in stable production and don't need the debug output, you can reduce or disable it with:

```bash
mkdir -p $HOME/.OpenFOAM/$WM_PROJECT_VERSION
cp $WM_PROJECT_DIR/etc/controlDict $HOME/.OpenFOAM/$WM_PROJECT_VERSION/
```

There are a variety of other parameters which can be used to reduce the amount of output that OpenFOAM writes to disk as well as the frequency; these run-time parameters are documented for [version 6](link-to-version-6-docs-if-available) and [version 7](link-to-version-7-docs-if-available).

For example, the `debugSwitches` dictionary in `$HOME/.OpenFOAM/$WM_PROJECT_VERSION/controlDict` can be altered to change the flags from values greater than zero to zero. Another solution would be to make use of the local scratch (`$SLURM_TMPDIR`), a disk attached directly to the compute node, discussed [here](link-to-scratch-discussion-if-available).


**(Retrieved from "https://docs.alliancecan.ca/mediawiki/index.php?title=OpenFOAM&oldid=175231")**
