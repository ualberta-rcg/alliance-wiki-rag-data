# Quantum ESPRESSO

Quantum ESPRESSO is an integrated suite of Open-Source computer codes for electronic-structure calculations and materials modeling at the nanoscale. It is based on density-functional theory, plane waves, and pseudopotentials.

Quantum ESPRESSO has evolved into a distribution of independent and inter-operable codes in the spirit of an open-source project. The Quantum ESPRESSO distribution consists of a “historical” core set of components, and a set of plug-ins that perform more advanced tasks, plus a number of third-party packages designed to be inter-operable with the core components.<sup>[1]</sup>


## Usage

To use Quantum ESPRESSO, you need to load a module (see [Using modules]). You can see available versions using `module avail quantumespresso` or `module spider quantumespresso`, and load one with (for example), `module load quantumespresso/6.6`.

### Example Script: `qe_ex1.sh`

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

The above example requests 32 processes, which is more than needed for the silicon tutorial case. Please be aware that suitable selection of a process count is complicated, but it is your responsibility to choose an efficient number. See also [Advanced MPI scheduling].


## Known Problems

### No Pseudopotential Files

There is no system-wide repository of pseudopotentials for Quantum ESPRESSO on our clusters. You must find or create and store your own pseudopotential files.

### Segfaults with OpenMPI 3.1.2

Users have reported random segfaults on Cedar when using Quantum ESPRESSO versions compiled for OpenMPI 3.1.2 in single-node jobs (shared memory communication). These issues seem not to happen with other versions of OpenMPI. If you experience such problems, first try to use an OpenMPI 2.1.1-based toolchain. For example:

```bash
[name@server ~]$ module load gcc/5.4.0
[name@server ~]$ module load openmpi/2.1.1
[name@server ~]$ module load quantumespresso/6.3
```

### Parameter Error in Grimme-D3

Incorrect results may be obtained when running Grimme-D3 with the element barium (Ba). The error comes from an incorrect value for one of the coefficients for barium, specifically, the `r2r4` parameter in the source code file `dft-d3/core.f90`. The correct value should be 10.15679528, not 0.15679528. The error has been confirmed by the QE developers to exist in all versions from 6.2.1 to 7.1.<sup>[2]</sup>


<sup>[1]</sup> Quantum ESPRESSO web site.

<sup>[2]</sup> "Wrong r2r4 value for Ba in the dft-d3 code", Quantum ESPRESSO mailing list, 2022 July 9.

