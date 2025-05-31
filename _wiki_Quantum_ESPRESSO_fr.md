# Quantum ESPRESSO

Quantum ESPRESSO is an open-source suite of codes for electronic-structure calculations and materials modeling at the atomic and microscopic scales.  The codes are based on density functional theory, plane waves, and pseudopotentials.

The independent and interoperable codes are distributed under the open-source model.  A set of routines or libraries for performing more advanced tasks is added to the core set of original components, in addition to a few packages produced by other contributors.


## Usage

To use the Quantum ESPRESSO suite, you must load a module (see [Using a module](link-to-using-a-module-page)).

Use `module avail quantumespresso` or `module spider quantumespresso` to see the available versions.

Load the module with, for example, `module load quantumespresso/6.6`.


## Example: `qe_ex1.sh`

```bash
#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --time=0-1:00           # DD-HH:MM
#SBATCH --nodes=1
#SBATCH --tasks-per-node=32     # MPI tasks
#SBATCH --mem=0                 # all memory on node

module load StdEnv/2020 intel/2020.1.217 openmpi/4.0.3
module load quantumespresso/6.6

srun pw.x < si.scf.in > si.scf.out
```

This example requests 32 processes, which is more than necessary in the case of the silicon tutorial. Remember that determining the number of processes to request is complicated, but you must choose an appropriate number. See also [Controlling scheduling with MPI](link-to-mpi-scheduling-page).


## Known Issues

### Absence of Pseudopotential Files

Our clusters do not have any pseudopotential repositories for Quantum ESPRESSO. You must find or create your own files and save them yourself.


### `segfault` with OpenMPI 3.1.2

Users have reported random crashes (`segfault`) on Cedar when using versions of Quantum ESPRESSO compiled with OpenMPI 3.1.2 for single-node jobs (shared memory communication). These problems seem specific to this version. If you get a similar error, try first using a compilation chain based on OpenMPI 2.1.1. For example:

```bash
[name@server ~]$ module load gcc/5.4.0
[name@server ~]$ module load openmpi/2.1.1
[name@server ~]$ module load quantumespresso/6.3
```


### Parameter Error with Grimme-D3

Incorrect results may be obtained when using Grimme-3 with Barium (Ba). This error is due to an incorrect value for one of Barium's coefficients, namely the `r2r4` parameter in the source code file `dft-d3/core.f90`.  The value should be 0.15679528, not 10.1567952. This error is confirmed in Quantum ESPRESSO versions 6.2.1 to 7.1.

[1] "Wrong r2r4 value for Ba in the dft-d3 code", Quantum ESPRESSO mailing list, July 9, 2022.


**(Remember to replace `link-to-using-a-module-page` and `link-to-mpi-scheduling-page` with the actual links to those pages.)**
