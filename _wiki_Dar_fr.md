# dar: Disk ARchive Utility

The open-source utility `dar` (Disk ARchive) is designed as a replacement for the Unix `tar` tool. It can be compiled on any Unix-like system and has been actively maintained since its launch in 2002.

Like `tar`, it supports full, differential, and incremental backups. However, it offers faster file access and data restoration than `tar` because each archive contains a file index—a significant advantage for large archives.  `dar` compresses each file separately, providing greater resilience in case of data corruption. It also allows you to avoid compressing already highly compressed files such as `.mp4` or `.gz`.  Its many useful features include robust encryption; archive splitting into slices as small as one byte; handling of extended attributes, sparse files, hard and symbolic links; and detection and recovery of data corruption in both header files with minimal loss. For more information, see the [website](link_to_website_here) and the [comparison with tar](link_to_comparison_here).


## Where to Find the Utility

On our clusters, `dar` is available on `/cvmfs`. With `StdEnv/2020`:

```bash
[user_name@localhost]$ which dar
/cvmfs/soft.computecanada.ca/gentoo/2020/usr/bin/dar
[user_name@localhost]$ dar --version
dar version 2.5.11, Copyright (C) 2002-2052 Denis Corbin
...
```

## Manual Usage

### Basic Archiving and Extraction

Let's assume a subdirectory `test` in the current directory. To create an archive, enter the following command in the current directory:

```bash
[user_name@localhost]$ dar -w -c all -g test
```

This creates the archive file `all.1.dar`, where `all` is the base name and `1` is the slice number. An archive can be split into multiple slices. Multiple directories and files can be included in an archive, for example:

```bash
[user_name@localhost]$ dar -w -c all -g testDir1 -g testDir2 -g file1 -g file2
```

Note that all paths must be relative to the current directory.

To list the contents of an archive, use only the base name:

```bash
[user_name@localhost]$ dar -l all
```

To extract a file into a subdirectory `restore`, use the base name and the file path:

```bash
[user_name@localhost]$ dar -R restore/ -O -w -x all -v -g test/filename
```

The `-O` flag allows you to disregard file ownership. If you restore someone else's files without being an administrator (root), incorrect ownership attribution could cause problems. If you restore your own files, a message will be issued if you are not an administrator, asking you to confirm the operation. To avoid this message, use the `-O` flag. If `restore/test` exists, the `-w` flag disables the warning.

To extract an entire directory, use:

```bash
[user_name@localhost]$ dar -R restore/ -O -w -x all -v -g test
```

Similarly to archive creation, you can pass multiple directories and files with multiple `-g` flags. Note that `dar` does not accept Unix wildcard masks after `-g`.


#### Working with the Lustre File System

Some extended attributes are automatically saved when archived files come from a Lustre file system (usually in `/home`, `/project`, or `/scratch` on one of our general-purpose computing clusters).

To see the extended attributes assigned to each archived file, use the `-alist-ea` flag:

```bash
[name@server ~]$ dar -l all -alist-ea
```

You will see statements like `Extended Attribute: [lustre.lov]`.

With this attribute, file extractions to a Lustre-formatted location will work as usual. However, if you try to extract a file to a local location on a compute node (e.g., in `$SLURM_TMPDIR`), you will get error messages like `Error while adding EA lustre.lov: Operation not supported`.

To avoid these errors, the `-u` flag can exclude a specific attribute type and still extract the affected files:

```bash
[name@server ~]$ dar -R restore/ -O -w -x all -v -g test -u 'lustre*'
```

Another solution is to remove the `lustre.lov` attribute when creating the archive using the same `-u` flag:

```bash
[name@server ~]$ dar -w -c all -g test -u 'lustre*'
```

In conclusion, this is only necessary if you intend to extract files to a location that does not have the Lustre format.


### Incremental Backup

To create a differential and incremental backup, append the base name of the referenced archive to `-A`. Let's take the example of a full backup named `monday` that you create on Monday:

```bash
[user_name@localhost]$ dar -w -c monday -g test
```

On Tuesday, some files are modified, and only these are included in a new incremental backup named `tuesday`, using the `monday` archive as a reference:

```bash
[user_name@localhost]$ dar -w -A monday -c tuesday -g test
```

On Wednesday, other files are modified, and a new backup is created named `wednesday`, with the `tuesday` archive as a reference:

```bash
[user_name@localhost]$ dar -w -A tuesday -c wednesday -g test
```

There are now three files:

```bash
[user_name@localhost]$ ls *.dar
monday.1.dar     tuesday.1.dar    wednesday.1.dar
```

The file `wednesday.1.dar` contains only the files modified on Wednesday, but not the files from Monday or Tuesday. The command:

```bash
[user_name@localhost]$ dar -R restore -O -x wednesday
```

will only restore the files modified on Wednesday. To restore all files, you will need to go through all backups in chronological order:

```bash
[user_name@localhost]$ dar -R restore -O -w -x monday  # restore the full backup
[user_name@localhost]$ dar -R restore -O -w -x tuesday # restore the first incremental backup
[user_name@localhost]$ dar -R restore -O -w -x wednesday # restore the second incremental backup
```

### Limiting Slice Size

To set the maximum size in bytes of each slice, use the `-s` flag followed by a number and a unit of measurement (k, M, G, or T). For example, for a 1340MB archive, the command:

```bash
[user_name@localhost]$ dar -s 100M -w -c monday -g test
```

creates 14 slices named `monday.{1..14}.dar`. To extract from all these slices, use the base name:

```bash
[user_name@localhost]$ dar -O -x monday
```

## External Scripts

A member of our team has created bash functions to make using `dar` easier. We encourage you to use them as inspiration for preparing your own scripts. For details, see [here](link_to_scripts_here).
