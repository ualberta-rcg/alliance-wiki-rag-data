# LAMMPS Molecular Dynamics Code

## General

LAMMPS (Large-scale Atomic/Molecular Massively Parallel Simulator) is a classical molecular dynamics code distributed by Sandia National Laboratories.

**Project website:** [http://lammps.sandia.gov/](http://lammps.sandia.gov/)

**Documentation:** Online Manual

**Mailing list:** [http://lammps.sandia.gov/mail.html](http://lammps.sandia.gov/mail.html)

LAMMPS is parallelized with MPI and OpenMP and can run on GPUs.


## Force Fields

All supported force fields are listed on the [package website](http://lammps.sandia.gov/), classified by functional form (e.g., pairwise potentials, many-body potentials, etc.).  The large number of supported force fields makes LAMMPS suitable for many application areas:

*   **Biomolecules:** CHARMM, AMBER, OPLS, COMPASS (class 2), long-range Coulombics via PPPM, point dipoles, ...
*   **Polymers:** all-atom, united-atom, coarse-grain (bead-spring FENE), bond-breaking, …
*   **Materials:** EAM and MEAM for metals, Buckingham, Morse, Yukawa, Stillinger-Weber, Tersoff, EDIP, COMB, SNAP, ...
*   **Reactions:** AI-REBO, REBO, ReaxFF, eFF
*   **Mesoscale:** granular, DPD, Gay-Berne, colloidal, peri-dynamics, DSMC...

Combinations of potentials can be used for hybrid systems, e.g., water on metal, polymer/semiconductor interfaces, colloids in solution, ...


## Versions and Packages

To see which LAMMPS versions are installed, run `module spider lammps`.  See [Using modules](link-to-using-modules-page-if-available) for more about `module` subcommands.

LAMMPS version numbers are based on release dates (YYYYMMDD). Use `module avail lammps` to see all installed releases.

For example, the March 31, 2017, release has three modules:

*   Built with MPI: `lammps/20170331`
*   Built with USER-OMP support: `lammps-omp/20170331`
*   Built with USER-INTEL support: `lammps-user-intel/20170331`

GPU-enabled versions are also available. Load the CUDA module before the LAMMPS module:

```bash
$ module load cuda
$ module load lammps-omp/20170331
```

The executable name may differ between versions.  Prebuilt versions on our clusters have a symbolic link called `lmp`.  To see the original executable name for a given module, list the files in the `${EBROOTLAMMPS}/bin` directory:

```bash
$ module load lammps-omp/20170331
$ ls ${EBROOTLAMMPS}/bin/
```

Recent LAMMPS versions contain about 60 packages that can be enabled or disabled during compilation. Not all packages can be enabled in a single executable. All packages are documented on the official webpage.  If a simulation doesn't work with one module, a necessary package might be disabled.

For some LAMMPS modules, a `list-packages.txt` file lists enabled ("Supported") and disabled ("Not Supported") packages. After loading a module, run:

```bash
cat ${EBROOTLAMMPS}/list-packages.txt
```

If `list-packages.txt` is not found, examine the EasyBuild recipe file: `${EBROOTLAMMPS}/easybuild/LAMMPS*.eb`. The list of enabled packages appears in the `general_packages` block.


## Example of Input File

The following input file can be used with the example job scripts:

**File: `lammps.in`**

```
# 3d Lennard-Jones melt

units           lj
atom_style      atomic

lattice         fcc 0.8442
region          box block 0 15 0 15 0 15
create_box      1 box
create_atoms    1 box
mass            1 1.0

velocity        all create 1.44 87287 loop geom

pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0 2.5
neighbor        0.3 bin
neigh_modify    delay 5 every 1

fix             1 all nve
thermo          5
run             10000
write_data     config.end_sim

# End of the Input file.
```

**File: `run_lmp_serial.sh`**

```bash
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2500M
#SBATCH --time=0-00:30
module load StdEnv/2020 intel/2020.1.217 openmpi/4.0.3 lammps-omp/20210929

lmp < lammps.in > lammps_output.txt
```

**File: `run_lmp_mpi.sh`**

```bash
#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=2500M
#SBATCH --time=0-00:30
module load StdEnv/2020 intel/2020.1.217 openmpi/4.0.3 lammps-omp/20210929

srun lmp < lammps.in > lammps_output.txt
```


## Performance

Most CPU time in molecular dynamics simulations is spent computing pair interactions. LAMMPS uses domain decomposition to split the work among processors. Communication between processors is required, increasing with the number of processors, eventually leading to low CPU efficiency.

Before running extensive simulations, test performance with different core counts to maximize efficiency.

**Example 1 (low efficiency):**  A simulation of 4000 particles using 12 MPI tasks showed 46.45% of time spent on pair interactions and 44.5% on communication.

Loop time of 15.4965 on 12 procs for 25000 steps with 4000 atoms.
Performance: 696931.853 tau/day, 1613.268 timesteps/s.
90.2% CPU use with 12 MPI tasks x 1 OpenMP threads.

| Section | min time | avg time | max time | %varavg | %total |
|---|---|---|---|---|---|
| Pair | 6.6964 | 7.1974 | 7.9599 | 14.8 | 46.45 |
| Neigh | 0.94857 | 1.0047 | 1.0788 | 4.3 | 6.48 |
| Comm | 6.0595 | 6.8957 | 7.4611 | 17.1 | 44.50 |
| Output | 0.01517 | 0.01589 | 0.019863 | 1.0 | 0.10 |
| Modify | 0.14023 | 0.14968 | 0.16127 | 1.7 | 0.97 |
| Other | -- | 0.2332 | -- | -- | 1.50 |


**Example 2 (scaling with system size):**  This compares communication and pair interaction time for different system sizes and core counts.

| Atoms | Cores | Pairs | Comm | Pairs | Comm | Pairs | Comm | Pairs | Comm |
|---|---|---|---|---|---|---|---|---|---|
| 2048 | 1 | 73.68 | 1.36 |  |  |  |  |  |  |
|  | 2 | 70.35 | 5.19 |  |  |  |  |  |  |
|  | 4 | 62.77 | 13.98 |  |  |  |  |  |  |
|  | 8 | 58.36 | 20.14 |  |  |  |  |  |  |
|  | 16 | 56.69 | 20.18 |  |  |  |  |  |  |
| 4000 | 1 | 73.70 | 1.28 |  |  |  |  |  |  |
|  | 2 | 70.77 | 4.68 |  |  |  |  |  |  |
|  | 4 | 64.93 | 12.19 |  |  |  |  |  |  |
|  | 8 | 61.78 | 15.58 |  |  |  |  |  |  |
|  | 16 | 56.70 | 20.18 |  |  |  |  |  |  |
| 6912 | 1 | 73.66 | 1.27 |  |  |  |  |  |  |
|  | 2 | 70.51 | 5.11 |  |  |  |  |  |  |
|  | 4 | 67.52 | 8.99 |  |  |  |  |  |  |
|  | 8 | 64.10 | 12.86 |  |  |  |  |  |  |
|  | 16 | 56.97 | 19.80 |  |  |  |  |  |  |
| 13500 | 1 | 73.72 | 1.29 |  |  |  |  |  |  |
|  | 2 | 67.80 | 8.77 |  |  |  |  |  |  |
|  | 4 | 67.74 | 8.71 |  |  |  |  |  |  |
|  | 8 | 62.06 | 8.71 |  |  |  |  |  |  |
|  | 16 | 56.41 | 20.38 |  |  |  |  |  |  |


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=LAMMPS&oldid=173845](https://docs.alliancecan.ca/mediawiki/index.php?title=LAMMPS&oldid=173845)"
