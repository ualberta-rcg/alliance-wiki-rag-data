# LAMMPS

This page is a translated version of the page LAMMPS and the translation is 100% complete.

Other languages: [English](link-to-english-page), [français](current-page-url)

## Généralités

LAMMPS (for *large-scale atomic/molecular massively parallel simulator*) is a classical molecular dynamics software distributed by Sandia National Laboratories of the United States Department of Energy.

Project website: [http://lammps.sandia.gov/](http://lammps.sandia.gov/)

Documentation

Mailing list

Parallelization is done with MPI and OpenMP, and LAMMPS can be run on GPUs.


## Champs de force

The available force fields are listed in the [Interatomic potentials](link-to-interatomic-potentials-page) section of the website, classified according to their functional form (pairwise, N-body, etc.).  Because LAMMPS can handle a large number of force fields, it can be used for modeling in several application areas, for example:

*   **Biomolecules:** CHARMM, AMBER, OPLS, COMPASS (class 2), long-range Coulombics via PPPM, dipole moments, etc.
*   **Polymers:** atom bonding, atom union, coarse-grained (FENE globular chains), bond-breaking, etc.
*   **Materials:** EAM and MEAM for metals, Buckingham, Morse, Yukawa, Stillinger-Weber, Tersoff, EDIP, COMB, SNAP, etc.
*   **Reactions:** AI-REBO, REBO, ReaxFF, eFF
*   **Mesoscopic scale:** granular, DPD, Gay-Berne, colloidal, peridynamics, DSMC, etc.

Potentials can also be combined in hybrid systems, for example water on metal, polymer/semiconductor interfaces, colloids in solution, etc.


## Versions et paquets

To find out which versions are available, run `module spider lammps` (see [Using modules](link-to-using-modules-page)).

LAMMPS version numbers include the release date in YYYYMMDD format. Run `module avail lammps` to see the installed versions and select the one you want to use.

There may be several modules for the same version. For example, the March 31, 2017 version has the following three modules:

*   `lammps/20170331` developed under MPI
*   `lammps-omp/20170331` USER-OMP (OpenMP compatible)
*   `lammps-user-intel/20170331` USER-INTEL

These versions also work with GPUs; the CUDA module must be loaded before the LAMMPS module.

```bash
$ module load cuda
$ module load lammps-omp/20170331
```

The executable name may differ depending on the version. All versions installed on our clusters have the symbolic link `lmp`; you can therefore run LAMMPS by calling `lmp` regardless of the module you use.

To find out the original name of the executable for a particular module, list the files in the `${EBROOTLAMMPS}/bin` directory with, for example:

```bash
$ module load lammps-omp/20170331
$ ls ${EBROOTLAMMPS}/bin/
lmp lmp_icc_openmpi
```

where the executable is `lmp_icc_openmpi` and `lmp` is the associated symbolic link.

Different modules exist for the same version, depending on the packages included.  The most recent LAMMPS versions include about 60 different packages that can be enabled or disabled when compiling the program. Not all packages can be enabled in the same executable. See the [package documentation](link-to-package-documentation). If your simulation doesn't work with a module, it's possible that a necessary package hasn't been enabled.

For some LAMMPS modules, we provide the `list-packages.txt` file which lists the enabled (`Supported`) and disabled (`Not Supported`) packages. Once you have loaded a module, run `cat ${EBROOTLAMMPS}/list-packages.txt` to see its contents.

If `list-packages.txt` is not found, you may be able to determine which packages are available by opening the EasyBuild recipe file with `${EBROOTLAMMPS}/easybuild/LAMMPS*.eb`. The available packages are in the `general_packages` block.


## Exemples de fichiers d'entrée

The file below can be used with either of the example job scripts.

**Input File:** `lammps.in`

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

**Sequential Job Script:** `run_lmp_serial.sh`

```bash
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2500M
#SBATCH --time=0-00:30
module load StdEnv/2020 intel/2020.1.217 openmpi/4.0.3 lammps-omp/20210929

lmp < lammps.in > lammps_output.txt
```

**MPI Job Script:** `run_lmp_mpi.sh`

```bash
#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=2500M
#SBATCH --time=0-00:30
module load StdEnv/2020 intel/2020.1.217 openmpi/4.0.3 lammps-omp/20210929

srun lmp < lammps.in > lammps_output.txt
```


## Performance

In the case of molecular dynamics simulations, the calculation of pair interactions between particles occupies the largest part of the CPU time. LAMMPS uses the domain decomposition method to distribute the work to the available processors by assigning each a part of the simulation box. It is necessary for the processors to communicate with each other during the calculation of interactions between particles. For a given number of particles, the higher the number of processors, the more parts of the simulation box exchange information. Thus, the more processors there are, the longer the communication time, which eventually causes low CPU efficiency.

Before running simulations for problems of a certain size or with multi-part boxes, perform tests to see the impact of the number of cores on the program's performance. Perform short tests with a different number of cores to identify the number of cores likely to offer the best efficiency; however, the results remain approximate.

The following table shows the duration for the simulation of a 4000-particle system with 12 MPI tasks. Using 12 cores, the 4000-atom system is distributed over 12 small boxes and the efficiency is very low. The calculation of pair interactions occupies 46.45% of the time and communication between processors 44.5%. The significant proportion of communication time is due to the fact that such a small system uses a large number of small boxes.

Duration of loop 15.4965 for 12 processes of 25000 steps with 4000 atoms.

Performance: 696931.853 tau/day, 1613.268 timesteps/s.

CPU used at 90.2% with 12 MPI tasks x 1 OpenMP thread.


| SECTION       | durée minimale | durée moyenne | durée maximale | variation moyenne (%) | total (%) |
|---------------|-----------------|-----------------|-----------------|-----------------------|-----------|
| pairs         | 6.6964          | 7.1974          | 7.9599          | 14.8                   | 46.45     |
| neighbors     | 0.94857         | 1.0047          | 1.0788          | 4.3                    | 6.48      |
| communication | 6.0595          | 6.8957          | 7.4611          | 17.1                   | 44.50     |
| output        | 0.01517         | 0.01589         | 0.019863        | 1.0                    | 0.10      |
| modification  | 0.14023         | 0.14968         | 0.16127         | 1.7                    | 0.97      |
| other         | --              | 0.2332          | --              | --                     | 1.50      |


In the last table, the communication time is compared to the pair calculation time for different numbers of cores.

| Atoms       | Cores | pairs | comm. | pairs | comm. | pairs | comm. | pairs | comm. |
|-------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| 2048        | 1     | 73.68 | 1.36  |       |       |       |       |       |       |
|             | 2     | 70.35 | 5.19  |       |       |       |       |       |       |
|             | 4     | 62.77 | 13.98 |       |       |       |       |       |       |
|             | 8     | 58.36 | 20.14 |       |       |       |       |       |       |
|             | 16    | 56.69 | 20.18 |       |       |       |       |       |       |
| 4000        | 1     | 73.70 | 1.28  |       |       |       |       |       |       |
|             | 2     | 70.77 | 4.68  |       |       |       |       |       |       |
|             | 4     | 64.93 | 12.19 |       |       |       |       |       |       |
|             | 8     | 61.78 | 15.58 |       |       |       |       |       |       |
|             | 16    | 56.70 | 20.18 |       |       |       |       |       |       |
| 6912        | 1     | 73.66 | 1.27  |       |       |       |       |       |       |
|             | 2     | 70.51 | 5.11  |       |       |       |       |       |       |
|             | 4     | 67.52 | 8.99  |       |       |       |       |       |       |
|             | 8     | 64.10 | 12.86 |       |       |       |       |       |       |
|             | 16    | 56.97 | 19.80 |       |       |       |       |       |       |
| 13500       | 1     | 73.72 | 1.29  |       |       |       |       |       |       |
|             | 2     | 67.80 | 8.77  |       |       |       |       |       |       |
|             | 4     | 67.74 | 8.71  |       |       |       |       |       |       |
|             | 8     | 62.06 | 8.71  |       |       |       |       |       |       |
|             | 16    | 56.41 | 20.38 |       |       |       |       |       |       |


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=LAMMPS/fr&oldid=173854](https://docs.alliancecan.ca/mediawiki/index.php?title=LAMMPS/fr&oldid=173854)"
