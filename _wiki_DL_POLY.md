# DL_POLY

## General

DL_POLY is a general-purpose classical molecular dynamics (MD) simulation software. It provides scalable performance from a single-processor workstation to a high-performance parallel computer. DL_POLY_4 offers fully parallel I/O as well as a NetCDF alternative to the default ASCII trajectory file.

There is a mailing list [here](link-to-mailing-list-needed).


## License Limitations

DL-POLY is now open source and does not require registration. A new module `dl_poly4/5.1.0` is already installed under `StdEnv/2023` and is accessible to all users. However, if you would like to use the previous versions (`dl_poly4/4.10.0` and/or `dl_poly4/4.08`), you should contact support and ask to be added to the POSIX group that controls access to DL_POLY4. There is no need to register on the DL-POLY website.


## Modules

To see which versions of DL-POLY are installed on our systems, run `module spider dl_poly4`. See [Using modules](link-to-modules-doc-needed) for more about `module` subcommands.

To load version 5.x, use:

```bash
module load StdEnv/2023 intel/2023.2.1 openmpi/4.1.5 dl_poly4/5.1.0
```

To load the previous version 4.10.0, use:

```bash
module load StdEnv/2023 intel/2020.1.217 openmpi/4.0.3 dl_poly4/4.10.0
```

Note that this version requires being added to a POSIX group as explained above in [License limitations](#license-limitations).

We do not currently provide a module for the Java GUI interface.


## Scripts and Examples

The input files shown below (`CONTROL` and `FIELD`) were taken from example TEST01 that can be downloaded from the page of [DL-POLY examples](link-to-examples-needed).

To start a simulation, one must have at least three files:

*   `CONFIG`: simulation box (atomic coordinates)
*   `FIELD`: force field parameters
*   `CONTROL`: simulation parameters (time step, number of MD steps, simulation ensemble, ...etc.)

### CONTROL

```
SODIUM CHLORIDE WITH (27000 IONS)

restart scale
temperature           500.0
equilibration steps   20
steps                 20
timestep              0.001

cutoff                12.0
rvdw                  12.0
ewald precision       1d-6  

ensemble nvt berendsen 0.01

print every           2
stats every           2
collect
job time              100
close time            10

finish
```

### FIELD

```
SODIUM CHLORIDE WITH EWALD SUM (27000 IONS)
units internal
molecular types 1
SODIUM CHLORIDE
nummols 27
atoms 1000
Na+          22.9898         1.0  500
Cl-           35.453        -1.0  500
finish
vdw    3 
Na+     Na+     bhm      2544.35      3.1545      2.3400   1.0117e+4   4.8177e+3
Na+     Cl-     bhm      2035.48      3.1545      2.7550   6.7448e+4   8.3708e+4
Cl-     Cl-     bhm      1526.61      3.1545      3.1700   6.9857e+5   1.4032e+6
close
```

### run_serial_dlp.sh

```bash
#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2500M      # memory; default unit is megabytes.
#SBATCH --time=0-00:30           # time (DD-HH:MM).
# Load the module:
module load StdEnv/2023
module load intel/2023.2.1
module load openmpi/4.1.5
module load dl_poly4/5.1.0
echo "Starting run at: `date`"
dlp_exec=DLPOLY.Z
$dlp_exec
echo "Program finished with exit code $? at: `date`"
```

### run_mpi_dlp.sh

```bash
#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=2500M      # memory; default unit is megabytes.
#SBATCH --time=0-00:30           # time (DD-HH:MM).
# Load the module:
module load StdEnv/2023
module load intel/2023.2.1
module load openmpi/4.1.5
module load dl_poly4/5.1.0
echo "Starting run at: `date`"
dlp_exec=DLPOLY.Z

srun $dlp_exec
echo "Program finished with exit code $? at: `date`"
```


## Related Software

*   VMD
*   LAMMPS


**(Retrieved from "https://docs.alliancecan.ca/mediawiki/index.php?title=DL_POLY&oldid=173038")**
