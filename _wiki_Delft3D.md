# Delft3D

Delft3D is a 3D modeling suite used to investigate hydrodynamics, sediment transport and morphology, and water quality for fluvial, estuarine, and coastal environments.

## Examples

Delft3D includes several `run_*` scripts designed for use with the Sun Grid Engine job scheduler and the MPICH library.  The Alliance uses SLURM as its job scheduler and Open MPI as its default MPI implementation.  To demonstrate how to run Delft3D under SLURM, we've provided submission scripts for the computational examples included with the software.

To copy the examples to your home directory, follow these steps:

```bash
$ module load StdEnv/2020  intel/2020.1.217  openmpi/4.0.3 delft3d
$ cp -a $EBROOTDELFT3D/examples ~/
```

The `~/examples/` directory contains `start-slurm.sh` scripts. You can run these scripts with SLURM using a command like this:

```bash
$ sbatch start-slurm.sh
```

The `~/examples/readme.examples` file summarizes the results.
