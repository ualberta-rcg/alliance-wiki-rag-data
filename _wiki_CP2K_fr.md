# CP2K

CP2K is a software package for quantum chemistry and solid-state physics that enables atomistic simulations of solid, liquid, molecular, periodic, material, crystalline, and biological systems.


## Versions

The most recently installed version is CP2K 8.2. To load the module compiled with GCC, run the command:

```bash
module load StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 cp2k/8.2
```

You can also use the version compiled with Intel, but it seems less stable as it occasionally crashes for unknown reasons:

```bash
module load StdEnv/2020 intel/2020.1.217 openmpi/4.0.3 cp2k/8.2
```


## Example Task

Here we use the static calculation example from the [CP2K website](https://www.cp2k.org/).

Log in to a cluster and download the required files with `wget`:

```bash
wget https://www.cp2k.org/_media/static_calculation.tgz
tar xvfz static_calculation.tgz
cd static_calculation/sample_output_no_smearing
```

In this directory, create the following job script using your account name.

**File: `mpi_job.sh`**

```bash
#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --ntasks=4               # number of MPI processes
#SBATCH --mem-per-cpu=4G      # memory; default unit is megabytes
#SBATCH --time=0-00:15           # time (DD-HH:MM)
module load StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 cp2k/8.2
srun cp2k.popt -o Si_bulk8.out Si_bulk8.inp
```

To submit this job, run:

```bash
sbatch mpi_job.sh
```

To check if the job is finished, run:

```bash
sq
```

Your job is finished if it does not appear in the list.  The CP2K result will be in the file `Si_bulk8.out`. There will also be a results file named `slurm-*.out` which will be empty if the calculation ran without errors.


## Threads and MPI

Starting with version 8.2, the CP2K installation provides the executable `cp2k.popt` and the OpenMP/MPI executable `cp2k.psmp` which can improve the performance of some calculations. With our test, we obtained a 10% improvement with the QS/H2O-512.inp test using 2 threads per MPI process, compared to running `cp2k.popt` in MPI alone; in both cases, the total CPU cores were identical.

The example below is an OpenMP/MPI file for submitting a job on Beluga. On other clusters, modify the number of tasks to match the number of cores available on the nodes of each cluster. The performance difference with the use of threads depends on the problem being addressed. In some cases, the `cp2k.psmp` executable may take longer, and it is important to test with your code to choose the best option.

**File: `openmp_mpi_job.sh`**

```bash
#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --nodes=2
#SBATCH --ntasks=40               # number of MPI processes
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3G          # memory (in MB by default)
#SBATCH --time=0-00:59            # computation time (DD-HH:MM)
module load StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 cp2k/8.2
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun --cpus-per-task=$OMP_NUM_THREADS cp2k.psmp -o H2O-512.out H2O-512.inp
```

Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=CP2K/fr&oldid=134800](https://docs.alliancecan.ca/mediawiki/index.php?title=CP2K/fr&oldid=134800)"
