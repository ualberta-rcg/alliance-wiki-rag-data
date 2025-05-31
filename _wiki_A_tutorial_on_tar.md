# A Tutorial on 'tar'

## Using `tar` to archive files and directories

The primary archiving utility on Linux and Unix-like systems is the `tar` command. It bundles several files or directories into a single file called an *archive file*, *tar file*, or humorously a *tarball*. By convention, an archive file has `.tar` as the file name extension.  When you archive a directory with `tar`, it includes all files and subdirectories within it, recursively.  So the command `tar --create --file project1.tar project1` will pack all the content of directory `project1` into the file `project1.tar`. The original directory remains unchanged.

You can extract files from an archive file using the same command with a different option: `tar --extract --file project1.tar`. If there is no directory with the original name, it will be created. If a directory of that name exists and contains files of the same names as in the archive file, they will be overwritten.  You can specify a destination directory using another option.


## Compressing and decompressing

The `tar` archiving utility can compress an archive file while creating it. Several compression methods exist.  We recommend either `xz` or `gzip`, used as follows:

```bash
[user_name@localhost]$ tar --create --xz --file project1.tar.xz project1
[user_name@localhost]$ tar --extract --xz --file project1.tar.xz
[user_name@localhost]$ tar --create --gzip --file project1.tar.gz project1
[user_name@localhost]$ tar --extract --gzip --file project1.tar.gz
```

Typically, `--xz` produces a smaller compressed file (better compression ratio) but takes longer and uses more RAM. `--gzip` doesn't compress as much but might be preferable if you have insufficient memory or experience excessive runtime during `tar --create`.

You can run `tar --create` without compression and then use `xz` or `gzip` separately, though this is rarely necessary. Similarly, you can use `xz -d` or `gzip -d` to decompress before `tar --extract`, but this is also seldom needed.

After creating a `tar` file, you can use `gzip` or `bzip2` to compress it further:

```bash
[user_name@localhost]$ gzip project1.tar
[user_name@localhost]$ bzip2 project1.tar
```

These commands produce `project1.tar.gz` and `project1.tar.bz2`.


## Common options

Here are common `tar` options.  Each has a single-letter and a whole-word form:

* `-c` or `--create`: Create a new archive.
* `-f` or `--file=`: Specifies the archive file name.
* `-x` or `--extract`: Extract files from the archive.
* `-t` or `--list`: List the archive's content.
* `-J` or `--xz`: Compress/decompress with `xz`.
* `-z` or `--gzip`: Compress/decompress with `gzip`.

Single-letter options can be combined: `tar -cJf project1.tar.zx project1` is equivalent to `tar --create --xz --file=project1.tar.xz project1`.

Many more options exist, depending on your `tar` version. Use `man tar` or `tar --help` for a complete list. Note that older systems might not support `--xz` compression.


## Examples

We'll use a directory containing `bin/`, `documents/`, `jobs/`, `new.log.dat`, `programs/`, `report/`, `results/`, `tests/`, and `work/`.  Your directory structure may differ, but the principles remain the same.


### Archiving files and directories

#### Archiving a directory

To archive the `results` directory as `results.tar`:

```bash
[user_name@localhost]$ tar -cvf results.tar results
```

This uses `-c` (create), `-v` (verbose), and `-f` (file).  The `-v` option shows the files being added.

#### Archiving files or directories starting with a particular letter

To archive directories starting with "r" into `archive.tar`:

```bash
[user_name@localhost]$ tar -cvf archive.tar r*
```

This archives `report` and `results`.  Similar commands work for files or directories matching other patterns (e.g., `*r*`, `*.dat`).


#### Adding (appending) files to the end of an archive

Use `-r` to add files to an existing archive:

```bash
[user_name@localhost]$ tar -rf results.tar new.log.dat
```

This adds `new.log.dat` to `results.tar`.  You can't add files to compressed archives (`.gz` or `.bz2`).


#### Combining two archive files into one

Use `-A` to append one archive to another:

```bash
[user_name@localhost]$ tar -A -f results.tar report.tar
```

This adds the contents of `report.tar` to `results.tar`.


#### Excluding particular files

To create `results.tar` without `.dat` files:

```bash
[user_name@localhost]$ tar -cvf results.tar --exclude=*.dat results
```


#### Preserving symbolic links

Use `-h` to preserve symbolic links:

```bash
[user_name@localhost]$ tar -cvhf results.tar results
```


### Compressing files and archives

#### Compress a file, files, a tar archive

Archiving and compressing are distinct. Archiving combines files; compressing reduces a file's size using utilities like `gzip` or `bzip2`.

To compress `new.log.dat` and `results.tar` using `gzip`:

```bash
[user_name@localhost]$ gzip new.log.dat
[user_name@localhost]$ gzip results.tar
```

To compress during archive creation, use `-z` (for `gzip`) or `-j` (for `bzip2`):

```bash
[user_name@localhost]$ tar -cvzf results.tar.gz results
[user_name@localhost]$ tar -cvjf results.tar.bz2 results
```


#### Adding files to a compressed archive (tar.gz/tar.bz2)

You can't directly add files to compressed archives. Decompress first, add the files, then recompress.


### Unpacking compressed files and archives

#### Extracting the whole archive

To extract `results.tar` to a new directory `new_results`:

```bash
[user_name@localhost]$ tar -xvf results.tar -C new_results/
```

#### Decompressing gz and bz2 files

Use `gunzip` for `.gz` files and `bunzip2` for `.bz2` files.


#### Extracting a compressed archive file into another directory

To extract `results.tar.gz` to `new_results`:

```bash
[user_name@localhost]$ tar -xvzf results.tar.gz -C new_results/
```

Or in two steps:

```bash
[user_name@localhost]$ gunzip results.tar.gz
[user_name@localhost]$ tar -xvf results.tar -C new_results/
```


#### Extracting one file from an archive or a compressed archive

To extract `results/Res-01/log.15Feb16.4` from `results.tar` to `new_results`:

```bash
[user_name@localhost]$ tar -C ./new_results/ --extract --file=results.tar results/Res-01/log.15Feb16.4
```


#### Extract multiple files using wildcards

```bash
[user_name@localhost]$ tar -C ./new_results/ -xvf results.tar --wildcards "results/*.dat"
```


#### Contents of archive files

#### Listing the contents

To list the contents of `results.tar`:

```bash
[user_name@localhost]$ tar -tf results.tar
```

Use `-v` for verbose output (metadata):

```bash
[user_name@localhost]$ tar -tvf results.tar
```


#### Searching for a file in an archive file without unpacking it

To search for `log.15Feb16.4`:

```bash
[user_name@localhost]$ tar -tf results.tar | grep -a log.15Feb16.4
```


#### Listing the contents of a compressed file (*.gz or .bz2)

Use `-z` for `.gz` and `-j` for `.bz2`:

```bash
[user_name@localhost]$ tar -tvzf results.tar.gz
[user_name@localhost]$ tar -tvjf results.tar.bz2
```


## Other useful utilities

### Size of a file, directory or archive

Use `du -sh`:

```bash
[user_name@localhost]$ du -sh results.tar
```


### Splitting files

Use `split` to split large files:

```bash
[user_name@localhost]$ split -b 100MB results.tar small-res
```

To recombine:

```bash
[user_name@localhost]$ cat small-res* > your_archive_name.tar
```


## Reminder of common commands

* `pwd`: Print working directory.
* `ls`: List directory contents.
* `du -sh`: Show disk usage.


Remember that `gzip`, `bzip2`, and `tar` require free space.  On clusters, use `quota` or `quota -s` to check your quota.
