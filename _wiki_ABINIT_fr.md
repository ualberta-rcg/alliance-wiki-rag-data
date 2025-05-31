# ABINIT

ABINIT is a software suite for calculating the optical, mechanical, vibrational, and other observable properties of materials. Using the Density Functional Theory (DFT) equations, it's possible to move towards more advanced applications with DFT-based perturbation theories and several N-body Green's functions (GW and DMFT). ABINIT can calculate molecules, nanostructures, and solids, regardless of their chemical composition. The suite offers several complete and reliable tables of atomic potentials.


To find out which versions are available, use the command `module spider abinit`. Then run the same command with a version number (e.g., `module spider abinit/8.4.4`) to find out if other modules need to be loaded beforehand. For more information, see [Using Modules](link-to-using-modules-page).


## Atomic Data

We do not have a collection of atomic data for ABINIT. To obtain the files you need, refer to [Atomic data files](link-to-atomic-data-files-page).


Since these files are usually less than 1MB, they can be directly downloaded to a login node using their URL and `wget`. The following example is used to download the hydrogen pseudopotential file:

```bash
[name@server ~]$ wget http://www.pseudo-dojo.org/pseudos/nc-sr-04_pbe_standard/H.psp8.gz
```


## Example Scripts

You will find data files to perform tests and follow the ABINIT tutorials at `$EBROOTABINIT/share/abinit-test/Psps_for_tests/` and `$EBROOTABINIT/share/abinit-test/tutorial`.


### Example Script

More substantial calculations than tests or tutorial exercises should be submitted to the Slurm scheduler. The following script is an example of a job that uses 64 CPU cores in two nodes for 48 hours, requiring 1024MB of memory per core. This example can be adapted to your specific cases.

**File:** `abinit_job.sh`

```bash
#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --nodes=2                # number of nodes
#SBATCH --ntasks=64               # number of MPI processes
#SBATCH --mem-per-cpu=1024M      # memory use per MPI process; default unit is megabytes
#SBATCH --time=2-00:00           # time (DD-HH:MM)
module purge
module load abinit/8.2.2
srun abinit < parameters.txt > output.log
```


**(Retrieved from "https://docs.alliancecan.ca/mediawiki/index.php?title=ABINIT/fr&oldid=117858")**
