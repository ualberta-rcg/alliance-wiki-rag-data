# Handling Large Collections of Files

In some fields, particularly in artificial intelligence and machine learning, one often has to deal with hundreds of thousands of files, sometimes comprising several hundred kilobytes. In these cases, one must take into account the object limits imposed by the quotas of file systems. A very large number of files, and particularly small files, causes significant performance problems for file systems and the automated backup of `/home` and `/project`.  We present here the advantages and disadvantages of some solutions for storing these large datasets.

## Contents

1. Locating directories containing a large number of files
2. Locating directories that occupy the most disk space
3. Solutions
    * Local Disk
    * RAM Disk
    * Archiving
        * `dar`
        * HDF5
        * SQLite
        * Parallel Compression
        * Partial Extraction of an Archive File
    * Hidden Files
    * `git`


## Locating Directories Containing a Large Number of Files

For optimization purposes, it is always preferable to first identify the directories where performance gains are possible. You can use the following code to recursively count the files in the subdirectories of the current directory:

```bash
for FOLDER in $(find . -maxdepth 1 -type d | tail -n +2); do
  echo -ne "$FOLDER:\t"
  find $FOLDER -type f | wc -l
done
```

## Locating Directories that Occupy the Most Disk Space

The following command lists the 10 directories that occupy the most space in the current directory:

```bash
[name@server ~]$ du -sh * | sort -hr | head -10
```

## Solutions

### Local Disk

Local disks connected to compute nodes are SATA SSDs or better; in general, their performance is far superior to that of the `/project` and `/scratch` file systems. A local disk is shared by all tasks that are executed simultaneously on one of its compute nodes, which means that the scheduler does not manage disk usage. The actual capacity of local disk space is not the same for all clusters and may vary within the same cluster.

Béluga offers approximately 370 GB of local disk space for CPU nodes; GPU nodes have a 1.6 TB NVMe disk to help with image datasets in artificial intelligence that have millions of small files.

Niagara does not offer local storage on its compute nodes.

In the case of other clusters, you can assume that the available disk space is at least 190 GB.

You can access the local disk from within a task using the environment variable `$SLURM_TMPDIR`. One approach would then be to keep your dataset in a single `tar` archive in the `/project` space and then copy it to the local disk at the beginning of your task, extract it, and use the data during the task. If there have been changes, you can archive the content in a `tar` file and copy it back to the `/project` space.

The following submission script uses an entire node:

**File: `job_script.sh`**

```bash
#!/bin/bash
#SBATCH --time=1-00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
cd $SLURM_TMPDIR
mkdir work
cd work
tar -xf ~/projects/def-foo/johndoe/my_data.tar
# Now do my computations here on the local disk using the contents of the extracted archive...
# The computations are done, so clean up the data set...
cd $SLURM_TMPDIR
tar -cf ~/projects/def-foo/johndoe/results.tar work
```

### RAM Disk

The `/tmp` file system can be used as a RAM disk on compute nodes. It is implemented with `tmpfs`.

* `/tmp` is `tmpfs` on all clusters;
* `/tmp` is emptied at the end of the task;
* like all other memory usage by a task, it counts towards the limit imposed for the `cgroup` associated with the `sbatch` request;
* the capacity of `tmpfs` is set to 100% via the `mount` options, which can affect some scripts, as `MemTotal` then represents the capacity of the physical RAM, which does not correspond to the `sbatch` request.


### Archiving

#### `dar`

Disk archiving utility designed to improve the `tar` tool; see the [dar tutorial](link_to_tutorial_here).

#### HDF5

Binary file format for storing various types of data, including extended objects such as matrices and images. Tools for manipulating these files are available in several languages, such as Python (`h5py`); see [HDF5](link_to_hdf5_info_here).

#### SQLite

SQLite allows the use of relational databases contained in a single file saved on disk, without the intervention of a server. The SQL `SELECT` command is used to access the data, and APIs are available for several programming languages.

With the APIs, you can interact with your SQLite database in C/C++, Python, R, Java, or Perl programs, for example. Modern relational databases have data types for managing the storage of BLOBs (binary large objects) such as the content of image files; rather than storing 5 or 10 million individual PNG or JPEG LOB image files, it would be more practical to group them into a SQLite file.

This solution, however, requires creating a SQLite database; you must therefore know SQL and be able to create a simple relational database. It should be noted that the performance of SQLite can degrade with very large databases (from several gigabytes); you might then prefer a more traditional approach and use MySQL or PostgreSQL with a database server.

The SQLite executable is named `sqlite3`. It is available through the `nixpkgs` module, which is loaded by default on our systems.

#### Parallel Compression

To create an archive with a large number of files, it might be advantageous to create a compressed archive with `pigz` rather than using `gzip`.

```bash
[name@server ~]$ tar -vc --use-compress-program="pigz -p 4" -f dir.tar.gz dir_to_tar
```

Here, the compression uses 4 cores.

#### Partial Extraction of an Archive File

It is not always necessary to extract the entire contents of an archive file. For example, if a simulation or task only requires the files from a particular directory, the directory can be extracted from the archive file and saved to the local disk with:

```bash
[name@server ~]$ tar -zxf /path/to/archive.tar.gz dir/subdir --directory $SLURM_TMPDIR
```

### Hidden Files

### `git`

If you use `git`, the number of files in the hidden subdirectory can increase significantly over time. To speed up performance, use the `git repack` command, which groups several of the files into a few large databases.


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Handling_large_collections_of_files/fr&oldid=159467](https://docs.alliancecan.ca/mediawiki/index.php?title=Handling_large_collections_of_files/fr&oldid=159467)"
