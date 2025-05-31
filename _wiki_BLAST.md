# BLAST

BLAST ("Basic Local Alignment Search Tool") finds regions of similarity between biological sequences. The program compares nucleotide or protein sequences to sequence databases and calculates the statistical significance.

## Contents

* [User manual](#user-manual)
* [Databases](#databases)
* [Accelerating the search](#accelerating-the-search)
    * [makeblastdb](#makeblastdb)
    * [Task array](#task-array)
        * [Preprocessing](#preprocessing)
        * [Job submission](#job-submission)
    * [GNU Parallel](#gnu-parallel)
        * [Running with multiple cores on one node](#running-with-multiple-cores-on-one-node)
            * [Job submission](#job-submission-1)
    * [Additional tips](#additional-tips)

## User manual

You can find more information on its arguments in the [user manual](link_to_user_manual_here) or with:

```bash
[name@server ~]$ blastn -help
```

## Databases

Some frequently used sequence databases are installed on the clusters in `/cvmfs/bio.data.computecanada.ca/content/databases/Core/blast_dbs/2022_03_23/`.

Examine that directory and its subdirectories, e.g., with:

```bash
[name@server ~]$ ls /cvmfs/bio.data.computecanada.ca/content/databases/Core/blast_dbs/2022_03_23/
```

## Accelerating the search

For the examples below, the file `ref.fa` will be used as the reference database in FASTA format, and `seq.fa` as the queries.

### makeblastdb

Before running a search, we must build the database. This can be a preprocessing job, where the other jobs are dependent on the completion of the `makeblastdb` job.

Here is an example of a submission script:

**File: makeblastdb.sh**

```bash
#!/bin/bash
#SBATCH --account=def-<user>  # The account to use
#SBATCH --time=00:02:00       # The duration in HH:MM:SS format
#SBATCH --cpus-per-task=1     # The number of cores
#SBATCH --mem=512M            # Total memory for this task
module load gcc/7.3.0
blast+/2.9.0
# Create the nucleotide database based on `ref.fa`.
makeblastdb -in ref.fa -title reference -dbtype nucl -out ref.fa
```

### Task array

BLAST search can greatly benefit from data parallelism by splitting the query file into multiple queries and running these queries against the database.

#### Preprocessing

In order to accelerate the search, the `seq.fa` file must be split into smaller chunks. These should be at least 1MB or greater, but *not smaller* as it may hurt the parallel filesystem.

Using the `faSplit` utility:

```bash
[name@server ~]$ module load kentutils/20180716
[name@server ~]$ faSplit sequence seqs.fa 10 seq
```

will create 10 files named `seqN.fa` where `N` is in the range of `[0..9]` for 10 queries (sequences).

#### Job submission

Once our queries are split, we can create a task for each `seq.fa.N` file using a job array. The task id from the array will map to the file name containing the query to run. This solution allows the scheduler to fit the smaller jobs from the array where there are resources available in the cluster.

**File: blastn_array.sh**

```bash
#!/bin/bash
#SBATCH --account=def-<user>  # The account to use
#SBATCH --time=00:02:00       # The duration in HH:MM:SS format of each task in the array
#SBATCH --cpus-per-task=1     # The number of cores for each task in the array
#SBATCH --mem-per-cpu=512M    # The memory per core for each task in the array
#SBATCH --array=0-9           # The number of tasks: 10
module load gcc/7.3.0
blast+/2.9.0
# Using the index of the current task, given by `$SLURM_ARRAY_TASK_ID`, run the corresponding query and write the result
blastn -db ref.fa -query seq.fa.${SLURM_ARRAY_TASK_ID} > seq.ref.${SLURM_ARRAY_TASK_ID}
```

With the above submission script, we can submit our search and it will run after the database has been created.

```bash
[name@server ~]$ sbatch --dependency=afterok:$(sbatch makeblastdb.sh) blastn_array.sh
```

Once all the tasks from the array are done, the results can be concatenated using:

```bash
[name@server ~]$ cat seq.ref.{0..9} > seq.ref
```

where the 10 files will be concatenated into `seq.ref` file. This could be done from the login node or as a dependent job upon completion of all the tasks from the array.

### GNU Parallel

GNU Parallel is a great tool to pack many small jobs into a single job, and parallelize it. This solution helps alleviate the issue of too many small files in a parallel filesystem by querying fixed-size chunks from `seq.fa` and running on one node and multiple cores.

As an example, if your `seq.fa` file is 3MB, you could read blocks of 1MB and GNU Parallel will create 3 jobs, thus using 3 cores. If we would have requested 10 cores in our task, we would have wasted 7 cores. Therefore, *the block size is important*. We can also let GNU Parallel decide, as done below.  See also [Handling large files](link_to_gnu_parallel_page_here) in the GNU Parallel page.

#### Running with multiple cores on one node

**File: blastn_gnu.sh**

```bash
#!/bin/bash
#SBATCH --account=def-<user>  # The account to use
#SBATCH --time=00:02:00       # The duration in HH:MM:SS format
#SBATCH --cpus-per-task=4     # The number of cores
#SBATCH --mem-per-cpu=512M    # The memory per core
module load gcc/7.3.0
blast+/2.9.0
cmd='blastn -db ref.fa -query -'
# Using the `::::` notation, give the sequences file to GNU parallel
# where
#   --jobs number of core to use, equal $SLURM_CPUS_PER_TASK (the number of cores requested)
#   --keep-order keep same order as given in input
#   --block -1 let GNU Parallel evaluate the block size and adapt
#   --recstart record start, here the sequence identifier `>`
#   --pipepart pipe parts of $cmd together.
#              `--pipepart` is faster than `--pipe` (which is limited to 500MB/s) as `--pipepart` can easily go to 5GB/s according to Ole Tange.
# and redirect results in `seq.ref`.
parallel --jobs $SLURM_CPUS_PER_TASK --keep-order --block -1 --recstart '>' --pipepart $cmd ::: seq.fa > seq.ref
```

Note: The file must not be compressed.

#### Job submission

With the above submission script, we can submit our search and it will run after the database has been created.

```bash
[name@server ~]$ sbatch --dependency=afterok:$(sbatch makeblastdb.sh) blastn_gnu.sh
```

### Additional tips

* If it fits into the node's local storage, copy your FASTA database to the local scratch space (`$SLURM_TMPDIR`).
* Reduce the number of hits returned (`-max_target_seqs`, `-max_hsps` can help), if it is reasonable for your research.
* Limit your hit list to nearly identical hits using `-evalue` filters, if it is reasonable for your research.


**(Retrieved from "https://docs.alliancecan.ca/mediawiki/index.php?title=BLAST&oldid=149335")**
