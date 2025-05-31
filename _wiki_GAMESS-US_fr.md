# GAMESS (General Atomic and Molecular Electronic Structure System)

GAMESS (General Atomic and Molecular Electronic Structure System) is an *ab initio* quantum chemistry software package.

## Execution

### Submitting a Job

To learn how to submit and monitor a job, see [Running Jobs](link-to-running-jobs-documentation).

The first step is to prepare a GAMESS input file containing the molecular geometry and the calculation to be performed. Consult the [GAMESS documentation](link-to-gamess-documentation), particularly [section 2](link-to-section-2-of-gamess-documentation), which describes the file format and keywords.

In addition to the input file (`name.inp` in our example), prepare a job script specifying the required computational resources.  The input file and script should be in the same directory.

```bash
File: gamess_job.sh
#!/bin/bash
#SBATCH --cpus-per-task=1       # Number of CPUs
#SBATCH --mem-per-cpu=4000M     # memory per CPU in MB
#SBATCH --time=0-00:30          # time (DD-HH:MM)
## Directory for GAMESS supplementary output files ($USERSCR):
#export USERSCR=$SCRATCH
## Directory for GAMESS temporary binary files ($SCR):
## Uncomment the following two lines to use /scratch instead of local disk
#export SCR="$SCRATCH/gamess_${SLURM_JOB_ID}/"
#mkdir -p $SCR
module load gamess-us/20170420-R1
export SLURM_CPUS_PER_TASK
# rungms will use this
rungms name.inp &> name.out
sbatch gamess_job.sh
```

Submit the job to the scheduler with:

```bash
sbatch gamess_job.sh
```

### Scratch Files

By default, temporary binary files (scratch files) are saved on the local disk of the compute node (`$SLURM_TMPDIR`), which should provide the best performance.  Remember that data in `$SLURM_TMPDIR` will be deleted when the job finishes. If local disk space is insufficient, use `/scratch` instead, using the `SCR` environment variable as shown above.

Supplementary output files are copied to the location specified by the `USERSCR` environment variable; by default, this is the user's `$SCRATCH` directory.


| Description                     | Environment Variable | Default location                     |
|---------------------------------|----------------------|--------------------------------------|
| GAMESS temporary binary files   | `SCR`                | `$SLURM_TMPDIR` (node-local storage) |
| GAMESS supplementary output files | `USERSCR`            | `$SCRATCH` (user's SCRATCH directory) |


### Execution on Multiple CPUs

Calculations can be performed on more than one CPU. The `--cpus-per-task` parameter sets the number of CPUs available for the calculation.

Since parallelization is done by sockets, GAMESS can only use CPU cores located on the same compute node. The maximum number of CPU cores for a job therefore depends on the node size in the cluster; for example, 32 CPU cores per node on Graham.

Quantum chemistry calculations are known to not scale as well as classical molecular mechanics, meaning they are not efficient with a large number of CPUs. The precise number of CPUs that can be used efficiently depends on the theoretical level and the number of atoms and basis functions.

To determine a reasonable number of CPUs to use, perform a scalability test, i.e., compare execution times with different numbers of CPUs using the same input file. Ideally, the execution time should halve when twice as many CPUs are used. Obviously, it would be a poor use of resources if, for example, a calculation ran only 30% faster with twice the number of CPUs. Some calculations may even take longer with a higher number of CPUs.


### Memory

Quantum chemistry calculations are often memory-bound, and at higher theoretical levels, larger molecules often require more RAM than is normally available on a single computer. To free up memory, packages like GAMESS use scratch storage to store intermediate results and access the disk later for calculations.

However, even the fastest scratch storage is significantly slower than memory. Therefore, provide sufficient memory as follows:

1. Specify the amount of memory in the job submission script. The value `--mem-per-cpu=4000M` is reasonable as it corresponds to the memory-to-CPU ratio of the base nodes. Requesting more might cause the job to wait to run on a large node.

2. In the `$SYSTEM` group of the input file, use the `MWORDS` and `MEMDDI` options to tell GAMESS how much memory can be used. `MWORDS` is the maximum memory the task can use on each core. The units are 1,000,000 words (not 1024*1024 words), where a word is 64 bits = 8 bytes. `MEMDDI` is the total memory required by the Distributed Data Interface (DDI), in units of 1,000,000 words. The memory required on each CPU core using *p* CPU cores is therefore `MEMDDI/p + MWORDS`.

For more information, see the `$SYSTEM` group section of the [GAMESS documentation](link-to-gamess-documentation).

It is important to keep a safety margin of a few hundred MB between the memory requested from the scheduler and the memory GAMESS can use. If the job results are incomplete and the `slurm-{JOBID}.out` file contains a message like `slurmstepd: error: Exceeded step/job memory limit at some point`, this indicates that Slurm stopped the job because it used more memory than requested. In this case, you can reduce the value of `MWORDS` or `MEMDDI` in the input file, or increase the value of `--mem-per-cpu` in the submission script.


## References

Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=GAMESS-US/fr&oldid=157635](https://docs.alliancecan.ca/mediawiki/index.php?title=GAMESS-US/fr&oldid=157635)"
