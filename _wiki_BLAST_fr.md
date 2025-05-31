# BLAST

BLAST (Basic Local Alignment Search Tool) finds similar regions between two or more nucleotide or amino acid sequences and performs an alignment of these homologous regions.

## User Manual

More information on the arguments can be found in the [user manual](link_to_user_manual_here) or by running the command:

```bash
[name@server ~]$ blastn -help
```

## Databases

Some frequently used sequence databases are located on our clusters in `/cvmfs/bio.data.computecanada.ca/content/databases/Core/blast_dbs/2022_03_23/`.  View the contents of this directory and its subdirectories with, for example:

```bash
[name@server ~]$ ls /cvmfs/bio.data.computecanada.ca/content/databases/Core/blast_dbs/2022_03_23/
```

## Speeding Up the Search

In the following examples, the file `ref.fa` is used as the reference database in FASTA format, and the file `seq.fa` for the queries.

### `makeblastdb`

Before running a search, you need to prepare the database. This can be done with a preprocessing task, with other tasks dependent on the result of the `makeblastdb` task.

Here is an example of a submission script:

**File: `makeblastdb.sh`**

```bash
#!/bin/bash
#SBATCH --account=def-<user>  # The account to use
#SBATCH --time=00:02:00       # The duration in HH:MM:SS format
#SBATCH --cpus-per-task=1     # The number of cores
#SBATCH --mem=512M            # Total memory for this task
module load gcc/7.3.0
module load blast+/2.9.0
# Create the nucleotide database based on `ref.fa`.
makeblastdb -in ref.fa -title reference -dbtype nucl -out ref.fa
```

### Job Array

Data parallelism can greatly improve search performance; this involves dividing the query file into several queries that will be performed on the database.

#### Preprocessing

To speed up the search, the `seq.fa` file must be divided into several smaller parts. These parts should be at least 1MB; smaller parts could harm the parallel file system.

With the `faSplit` utility, the command:

```bash
[name@server ~]$ module load kentutils/20180716
[name@server ~]$ faSplit sequence seqs.fa 10 seq
```

creates 10 files named `seqN.fa` where `N` represents `[0..9]` for 10 queries (sequences).

#### Submitting a Job

Once the queries are separated, you can create a task for each `seq.fa.N` file with a job array. The task ID contained in the array will correspond to the name of the file where the queries to be executed are located.

With this solution, the scheduler can use the available cluster resources to execute the smaller tasks.

**File: `blastn_array.sh`**

```bash
#!/bin/bash
#SBATCH --account=def-<user>  # The account to use
#SBATCH --time=00:02:00       # The duration in HH:MM:SS format of each task in the array
#SBATCH --cpus-per-task=1     # The number of cores for each task in the array
#SBATCH --mem-per-cpu=512M    # The memory per core for each task in the array
#SBATCH --array=0-9           # The number of tasks: 10
module load gcc/7.3.0
module load blast+/2.9.0
# Using the index of the current task, given by `$SLURM_ARRAY_TASK_ID`, run the corresponding query and write the result
blastn -db ref.fa -query seq.fa.${SLURM_ARRAY_TASK_ID} > seq.ref.${SLURM_ARRAY_TASK_ID}
```

With the above script, you can submit your BLAST query and it will be executed after the database has been created:

```bash
[name@server ~]$ sbatch --dependency=afterok:$(sbatch makeblastdb.sh) blastn_array.sh
```

When all the array jobs are finished, concatenate the results with:

```bash
[name@server ~]$ cat seq.ref.{0..9} > seq.ref
```

where the 10 files are concatenated into `seq.ref`. This can be done from the login node or as an independent task once all the array jobs are complete.

### GNU Parallel

GNU Parallel is a good tool for grouping multiple small tasks into one and parallelizing it. This solution reduces the problems that occur with multiple small files in a parallel file system with queries on fixed-size blocks in `seq.fa` with one core and multiple nodes.

For example, for a 100MB `seq.fa` file, you could read 10MB blocks and GNU Parallel would create 3 tasks, thus using 3 cores; requesting 10 cores, 7 cores would have been wasted.  The block size is therefore important. You can also let GNU Parallel decide, as in the example below.

See also [Working with large files](link_to_gnu_parallel_page_here) on the GNU Parallel page.

#### Using Multiple Cores in a Node

**File: `blastn_gnu.sh`**

```bash
#!/bin/bash
#SBATCH --account=def-<user>  # The account to use
#SBATCH --time=00:02:00       # The duration in HH:MM:SS format
#SBATCH --cpus-per-task=4     # The number of cores
#SBATCH --mem-per-cpu=512M    # The memory per core
module load gcc/7.3.0
module load blast+/2.9.0
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

#### Submitting a Job

With the above script, you can submit your BLAST query and it will be executed after the database has been created:

```bash
[name@server ~]$ sbatch --dependency=afterok:$(sbatch makeblastdb.sh) blastn_gnu.sh
```

### Additional Tips

* If the node's local storage allows, copy your FASTA database to the local `/scratch` space (`$SLURM_TMPDIR`).
* If your search allows it, reduce the number of responses (`-max_target_seqs`, `-max_hsps`).
* If your search allows it, limit the list of responses with filters (`-evalue`) to keep only near-identical responses.


**(Remember to replace `link_to_user_manual_here` and `link_to_gnu_parallel_page_here` with the actual links.)**
