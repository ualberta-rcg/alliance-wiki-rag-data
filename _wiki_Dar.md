# Dar: A Modern Replacement for Tar

The `dar` (Disk ARchive) utility is a modern replacement for the classic Unix `tar` tool. First released in 2002, `dar` is open-source, actively maintained, and compiles on any Unix-like system.

Like `tar`, `dar` supports full/differential/incremental backups.  Unlike `tar`, each `dar` archive includes a file index for fast file access and restore – especially useful for large archives! `dar` has built-in, file-by-file compression, increasing resilience against data corruption.  It can optionally skip compression for already-compressed files (e.g., `.mp4`, `.gz`).

`dar` also offers strong encryption, 1-byte resolution archive splitting, extended file attribute support, sparse file support, hard and symbolic link support, data corruption detection and recovery (with minimal data loss), and many other features.  A detailed `tar`-to-`dar` comparison is available on the [dar page](link-to-dar-page-if-available).


## Where to Find Dar

On our clusters, `dar` is available via `/cvmfs`.

With `StdEnv/2020`:

```bash
[user_name@localhost]$ which dar
/cvmfs/soft.computecanada.ca/gentoo/2020/usr/bin/dar
[user_name@localhost]$ dar --version
dar version 2.5.11, Copyright (C) 2002-2052 Denis Corbin
...
```


## Using Dar Manually

### Basic Archiving and Extracting

To archive a subdirectory `test` in the current directory:

```bash
[user_name@localhost]$ dar -w -c all -g test
```

This creates `all.1.dar`.  `all` is the base name, `1` is the slice number.  You can create multiple slices (see below).  Multiple directories and files can be included:

```bash
[user_name@localhost]$ dar -w -c all -g testDir1 -g testDir2 -g file1 -g file2
```

All paths should be relative to the current directory.

To list the archive's contents:

```bash
[user_name@localhost]$ dar -l all
```

To extract `test/filename` to a subdirectory `restore`:

```bash
[user_name@localhost]$ dar -R restore/ -O -w -x all -v -g test/filename
```

`-O` ignores file ownership (useful for non-root users restoring others' files).  `-w` disables warnings if `restore/test` already exists.

To extract the entire `test` directory:

```bash
[user_name@localhost]$ dar -R restore/ -O -w -x all -v -g test
```

Multiple directories and files can be specified using multiple `-g` flags.  `dar` does *not* accept Unix wildcards after `-g`.


#### A Note About the Lustre Filesystem

When archiving files from a Lustre filesystem (typically `/home`, `/project`, or `/scratch` on our general-purpose compute clusters), some extended attributes are saved automatically.  Use `-alist-ea` to see these:

```bash
[name@server ~]$ dar -l all -alist-ea
```

Output will include lines like: `Extended Attribute: [lustre.lov]`.  Extraction to a Lustre-formatted location works normally.  However, extraction to node local storage (`$SLURM_TMPDIR`) may produce errors like: `Error while adding EA lustre.lov : Operation not supported`.

To avoid these errors, use `-u` to exclude specific attributes:

```bash
[name@server ~]$ dar -R restore/ -O -w -x all -v -g test -u 'lustre*'
```

Alternatively, remove the attribute during archive creation:

```bash
[name@server ~]$ dar -w -c all -g test -u 'lustre*'
```

This is only necessary for extraction to non-Lustre locations.


### Incremental Backups

Create differential and incremental backups using `-A` to specify the reference archive.

**Monday (full backup):**

```bash
[user_name@localhost]$ dar -w -c monday -g test
```

**Tuesday (incremental):**

```bash
[user_name@localhost]$ dar -w -A monday -c tuesday -g test
```

**Wednesday (incremental):**

```bash
[user_name@localhost]$ dar -w -A tuesday -c wednesday -g test
```

This results in:

```bash
[user_name@localhost]$ ls *.dar
monday.1.dar     tuesday.1.dar    wednesday.1.dar
```

`wednesday.1.dar` only contains Wednesday's changes.  To restore everything, restore chronologically:

```bash
[user_name@localhost]$ dar -R restore -O -x monday     # Restore full backup
[user_name@localhost]$ dar -R restore -O -w -x tuesday  # Restore first incremental
[user_name@localhost]$ dar -R restore -O -w -x wednesday # Restore second incremental
```


### Limiting the Size of Each Slice

Use `-s` followed by a size and unit (k/M/G/T) to limit slice size.  For a 1340 MB archive:

```bash
[user_name@localhost]$ dar -s 100M -w -c monday -g test
```

This creates 14 slices (`monday.{1..14}.dar`).  Extract using the base name:

```bash
[user_name@localhost]$ dar -O -x monday
```


## External Scripts

Bash functions to facilitate `dar` usage are available [here](link-to-external-scripts-if-available).  Use these as inspiration for your own scripts.
