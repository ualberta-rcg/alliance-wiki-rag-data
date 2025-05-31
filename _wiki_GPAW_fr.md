# GPAW

This page is a translated version of the page GPAW and the translation is 100% complete.

Other languages: [English](link-to-english-page), [français](current-page-link)


## Description

GPAW is a Python-based density functional theory (DFT) code based on the projector augmented wave (PAW) method and the atomic simulation environment (ASE).


## Creating a GPAW Virtual Environment

We offer pre-compiled Python wheels for GPAW that can be installed in a Python virtual environment.

1. Check which versions are available.

```bash
[name@server ~] avail_wheels gpaw
name    version python  arch
------ --------- -------- ------
gpaw    22.8.0   cp39    avx2
gpaw    22.8.0   cp38    avx2
gpaw    22.8.0   cp310   avx2
```

2. Load a Python module (here python/3.10)

```bash
(ENV) [name@server ~] module load python/3.10
```

3. Create a new virtual environment.

```bash
[name@server ~] virtualenv --no-download venv_gpaw
created virtual environment CPython3.10.2.final.0-64 in 514ms
[...]
```

4. Activate the virtual environment (venv).

```bash
[name@server ~] source venv_gpaw/bin/activate
```

5. Install gpaw in venv.

```bash
(venv_gpaw) [name@server ~] pip install --no-index gpaw
[...]
Successfully installed ... gpaw-22.8.0+computecanada ...
```

6. Download and install the data in the SCRATCH filesystem.

```bash
(venv_gpaw) [name@server ~] gpaw install-data $SCRATCH
Available setups and pseudopotentials [*] https://wiki.fysik.dtu.dk/gpaw-files/gpaw-setups-0.9.20000.tar.gz
[...]
Setups installed into /scratch/name/gpaw-setups-0.9.20000.
Register this setup path in /home/name/.gpaw/rc.py? [y/n] n
As you wish.
[...]
Installation complete.
```

7. Configure GPAW_SETUP_PATH to point to the data directory.

```bash
(venv_gpaw) [name@server ~] export GPAW_SETUP_PATH=$SCRATCH/gpaw-setups-0.9.20000
```

8. Run the tests, which are very fast.

```bash
(venv_gpaw) [name@server ~] gpaw test
------------------------------------------------------------------------------------------------------------
| python-3.10.2 /home/name/venv_gpaw/bin/python                                                          |
| gpaw-22.8.0   /home/name/venv_gpaw/lib/python3.10/site-packages/gpaw/                                   |
| ase-3.22.1    /home/name/venv_gpaw/lib/python3.10/site-packages/ase/                                    |
| numpy-1.23.0  /home/name/venv_gpaw/lib/python3.10/site-packages/numpy/                                  |
| scipy-1.9.3   /home/name/venv_gpaw/lib/python3.10/site-packages/scipy/                                   |
| libxc-5.2.3   yes                                                                                       |
| _gpaw         /home/name/venv_gpaw/lib/python3.10/site-packages/_gpaw.cpython-310-x86_64-linux-gnu.so     |
| MPI enabled   yes                                                                                       |
| OpenMP enabled yes                                                                                       |
| scalapack     yes                                                                                       |
| Elpa          no                                                                                        |
| FFTW          yes                                                                                       |
| libvdwxc      no                                                                                        |
| PAW-datasets  (1) /scratch/name/gpaw-setups-0.9.20000                                                 |
------------------------------------------------------------------------------------------------------------
Doing a test calculation (cores: 1) : ... Done
Test parallel calculation with "gpaw -P 4 test".
(venv_gpaw) [name@server ~] gpaw -P 4 test
------------------------------------------------------------------------------------------------------------
| python-3.10.2 /home/name/venv_gpaw/bin/python                                                          |
| gpaw-22.8.0   /home/name/venv_gpaw/lib/python3.10/site-packages/gpaw/                                   |
| ase-3.22.1    /home/name/venv_gpaw/lib/python3.10/site-packages/ase/                                    |
| numpy-1.23.0  /home/name/venv_gpaw/lib/python3.10/site-packages/numpy/                                  |
| scipy-1.9.3   /home/name/venv_gpaw/lib/python3.10/site-packages/scipy/                                   |
| libxc-5.2.3   yes                                                                                       |
| _gpaw         /home/name/venv_gpaw/lib/python3.10/site-packages/_gpaw.cpython-310-x86_64-linux-gnu.so     |
| MPI enabled   yes                                                                                       |
| OpenMP enabled yes                                                                                       |
| scalapack     yes                                                                                       |
| Elpa          no                                                                                        |
| FFTW          yes                                                                                       |
| libvdwxc      no                                                                                        |
| PAW-datasets  (1) /scratch/name/gpaw-setups-0.9.20000                                                 |
------------------------------------------------------------------------------------------------------------
Doing a test calculation (cores: 4) : ... Done
```

The results of the last test are in the `test.txt` file, which will be in the current directory.


## Example Script

The following script is an example of hybrid OpenMP and MPI parallelization. Here, virtualenv is in your $HOME directory and the datasets are in $SCRATCH as above.


**File: job_gpaw.sh**

```bash
#!/bin/bash
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=0-01:00
module load gcc/9.3.0 openmpi/4.0.3
source ~/venv_gpaw/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export GPAW_SETUP_PATH=/scratch/$USER/gpaw-setups-0.9.20000

srun --cpus-per-task=$OMP_NUM_THREADS gpaw python my_gpaw_script.py
```

The script uses a single node with 8 MPI ranks (ntasks) and 4 OpenMP threads per MPI rank for a total of 32 CPUs.  You will likely want to adjust these values so that the product matches the number of cores of a whole node (either 32 on Graham, 40 on Beluga and Niagara, 48 on Cedar, or 64 on Narval).

Configuring `OMP_NUM_THREADS` as shown above ensures it always has the same value as `cpus-per-task` or 1 when `cpus-per-task` is not defined. Loading the `gcc/9.3.0` and `openmpi/4.0.3` modules ensures that the correct MPI library is used for the job, the same one that was used to build the wheels.


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=GPAW/fr&oldid=134553](https://docs.alliancecan.ca/mediawiki/index.php?title=GPAW/fr&oldid=134553)"
