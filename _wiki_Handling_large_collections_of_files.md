# Handling Large Collections of Files

In domains like AI and Machine Learning, managing vast collections (hundreds of thousands or more) of relatively small files (a few hundred kilobytes each) presents challenges due to filesystem quotas and performance issues on shared filesystems.  This document outlines solutions for storing these datasets on a cluster.

## Finding Folders with Lots of Files

To identify areas for cleanup, use this code to recursively count files in subfolders of the current directory:

```bash
for FOLDER in $(find . -maxdepth 1 -type d | tail -n +2); do
  echo -ne "$FOLDER:\t"
  find $FOLDER -type f | wc -l
done
```

## Finding Folders Using the Most Disk Space

This code lists the top 10 directories consuming the most disk space in the current directory:

```bash
du -sh * | sort -hr | head -10
```

## Solutions

### Local Disk

Local disks attached to compute nodes (e.g., SATA SSD or NVMe) generally offer better performance than project or scratch filesystems.  Note that local disk space is shared by all jobs on the node and not allocated by the scheduler.  Available space varies across clusters.  Examples:

* **Béluga:** ~370GB (CPU nodes), 1.6TB NVMe (GPU nodes)
* **Niagara:** No local storage on compute nodes (see [Data management at Niagara](link-to-niagara-page-if-available))
* Other clusters: At least 190GB

Access the local disk within a job using the `$SLURM_TMPDIR` environment variable.  One approach is to archive the dataset as a single `.tar` file in the project space, copy it to the local disk at job start, extract it, use it, re-archive results, and copy back to the project space.

**Example job script (`job_script.sh`):**

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

The `/tmp` filesystem (using `tmpfs`) acts as a RAM disk on compute nodes.  Key features:

* `/tmp` is `tmpfs` on all clusters.
* Cleared at job end.
* Memory usage is subject to cgroup limits from the `sbatch` request.
* `tmpfs` size is set to 100% of available RAM (may affect some scripts).  `df` reports `/tmp` size as physical RAM, not the `sbatch` request.

### Archiving

#### dar

Disk archive utility, a modernized version of `tar`.  See [Dar documentation](link-to-dar-docs-if-available).

#### HDF5

High-performance binary format for various data types (matrices, images).  Tools are available in Python (`h5py`), etc. See [HDF5 documentation](link-to-hdf5-docs-if-available).

#### SQLite

A relational database stored in a single file, accessed via SQL commands and APIs for various programming languages.  Suitable for storing collections of small files (images) as binary blobs.  Performance may degrade for very large files (several GBs); consider using a database server (MySQL, PostgreSQL) in such cases.  The `sqlite3` executable is available via `nixpkgs`.

#### Parallel Compression

Use `pigz` for faster archiving of many files:

```bash
tar -vc --use-compress-program="pigz -p 4" -f dir.tar.gz dir_to_tar
```
(This uses 4 cores.)

#### Partial Extraction from an Archive

Extract only parts of an archive:

```bash
tar -zxf /path/to/archive.tar.gz dir/subdir --directory $SLURM_TMPDIR
```

### Cleaning Up Hidden Files

### Git

For Git repositories, use `git repack` to pack files into larger database files, improving performance.


