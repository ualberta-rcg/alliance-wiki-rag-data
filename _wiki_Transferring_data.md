# Transferring Data

Other languages: English, français

Use **data transfer nodes**, also called **data mover nodes**, instead of login nodes when transferring data to and from our clusters.  If a data transfer node is available, its URL will be listed near the top of the main page for each cluster: Béluga, Narval, Cedar, Graham, and Niagara. Globus automatically uses data transfer nodes.

## To and From Your Personal Computer

You'll need software supporting secure file transfers between your computer and our machines.  `scp` and `sftp` work in Linux or Mac OS X command-line environments. On Microsoft Windows, MobaXterm offers a graphical file transfer function and a command-line interface via SSH, while WinSCP is another free option.  Setting up an SSH key connection with WinSCP is detailed [here](link). PuTTY includes `pscp` and `psftp`, similar to the Linux/Mac command-line programs.

If transfers take over a minute, install and try Globus Personal Connect. Globus transfers can be set up to run in the background.


## Between Resources

Globus is the preferred tool for transferring data between systems.  However, other common tools include:

*   SFTP
*   SCP (Secure Copy Protocol)
*   rsync

**Note:** To transfer files between clusters (e.g., from Cedar to Niagara), use the SSH agent forwarding flag `-A` when logging into the other cluster. For example:

```bash
ssh -A USERNAME@cedar.alliancecan.ca
```

Then perform the copy:

```bash
[USERNAME@cedar5 ~]$ scp file USERNAME@niagara.alliancecan.ca:/scratch/g/group/USERNAME/
```

## From the World Wide Web

Standard tools for downloading data from websites are `wget` and `curl`.  Their similarities and differences are compared in several places, such as this [StackExchange article](article) or [here](here). While focused on Alliance Linux systems, this [tutorial](tutorial) also covers Mac and Windows. Both `wget` and `curl` can resume interrupted downloads using the `-c` and `-C -` command-line options, respectively. For cloud services (Google Cloud Storage, Google Drive, Google Photos), consider `rclone`.  `wget`, `curl`, and `rclone` are available on all Alliance clusters by default (no module loading needed). For command-line options, check each tool's man page or use `--help` or `-h`.


## Synchronizing Files

Synchronizing (or syncing) files (or directories) in two locations ensures both copies are identical. Here are several methods:


### Globus Transfer

Globus usually offers the best performance and reliability.  Normally, a Globus transfer overwrites destination files with source files. To avoid overwriting existing matching files, use the **Transfer & Timer Options** (see screenshot) and select "sync".

You can choose how Globus determines which files to transfer:

*   **Their checksums are different:** Slowest but most accurate. Catches changes resulting in files of the same size but different contents.
*   **File doesn't exist on destination:** Transfers only files created since the last sync. Useful for incremental file creation.
*   **File size is different:** A quick test.  A size change implies content change, triggering re-transfer.
*   **Modification time is newer:** Checks file modification times and transfers only newer source files.  Ensure the "preserve source file modification times" option is checked when initiating the transfer.

For more information on Globus, see [Globus](Globus).


### Rsync

`rsync` is a popular synchronization tool but can be slow with many files or high latency between locations.  `rsync` checks modification times and sizes, transferring only mismatched files. If modification times might not match, use the `-c` option to compute checksums at both source and destination, transferring only if checksums differ.

When transferring to `/project` filesystems, **do not** use `-p` and `-g` flags because `/project` quotas are enforced based on group ownership. Preserving group ownership (`-p` and `-g`) may lead to "Disk quota exceeded" errors. Since `-a` includes `-p` and `-g` by default, use `--no-g --no-p`:

```bash
[name@server ~]$ rsync -avzh --no-g --no-p LOCALNAME someuser@graham.alliancecan.ca:projects/def-professor/someuser/somedir/
```

`LOCALNAME` can be a directory or file with its path. `somedir` will be created if it doesn't exist.  `-z` compresses files (excluding those in the `--skip-compress` list), requiring more CPU resources. `-h` makes transferred file sizes human-readable. For very large files, add `--partial` to allow restarting interrupted transfers:

```bash
[name@server ~]$ rsync -avzh --no-g --no-p --partial --progress LOCALNAME someuser@graham.alliancecan.ca:projects/def-professor/someuser/somedir/
```

`--progress` displays progress for each file. For many smaller files, a single progress bar is preferable:

```bash
[name@server ~]$ rsync -azh --no-g --no-p --info=progress2 LOCALNAME someuser@graham.alliancecan.ca:projects/def-professor/someuser/somedir/
```

The above examples transfer from a local system to a project directory on a remote system.  Transfers from a remote system to a local project directory are similar:

```bash
[name@server ~]$ rsync -avzh --no-g --no-p someuser@graham.alliancecan.ca:REMOTENAME ~/projects/def-professor/someuser/somedir/
```

`REMOTENAME` can be a directory or file with its path. `somedir` will be created if it doesn't exist.  Locally (within a single system), you can transfer from home or scratch to project by omitting the cluster name:

```bash
[name@server ~]$ rsync -avh --no-g --no-p LOCALNAME ~/projects/def-professor/someuser/somedir/
```

`somedir` will be created if it doesn't exist before copying `LOCALNAME`. For comparison, the `cp` command can transfer `LOCALNAME` from home to project:

```bash
[name@server ~]$ cp -rv --preserve="mode,timestamps" LOCALNAME ~/projects/def-professor/someuser/somedir/
```

Unlike `rsync`, if `LOCALNAME` is a directory, it will be renamed to `somedir` if `somedir` doesn't exist.


### Using Checksums to Check if Files Match

If Globus is unavailable and `rsync` is too slow, use a checksum utility (e.g., `sha1sum`) on both systems to compare files:

```bash
[name@server ~]$ find /home/username/ -type f -print0 | xargs -0 sha1sum | tee checksum-result.log
```

This creates `checksum-result.log` containing checksums for files in `/home/username/`.  For many or large files, run this in the background (using `screen` or `tmux`).

After running on both systems, use `diff` to find mismatches:

```bash
[name@server ~]$ diff checksum-result-silo.log checksum-dtn.log
```

`find` might crawl directories in different orders, causing false differences.  Use `sort` before `diff`:

```bash
[name@server ~]$ sort -k2 checksum-result-silo.log -o checksum-result-silo.log
[name@server ~]$ sort -k2 checksum-dtn.log -o checksum-dtn.log
```


## SFTP

SFTP (Secure File Transfer Protocol) uses SSH for encrypted file transfers. To connect to `ADDRESS` as user `USERNAME`:

```bash
[name@server]$ sftp USERNAME@ADDRESS
```

Or, use an SSH key for authentication:

```bash
[name@server]$ sftp -i /home/name/.ssh/id_rsa USERNAME@ADDRESS
```

This gives the `sftp>` prompt. Use the `help` command for available commands.  Graphical programs (WinSCP, MobaXterm, FileZilla, Cyberduck) are available for Windows, Linux, and Mac OS.


## SCP

SCP (Secure Copy Protocol) uses SSH for encrypted transfers. Unlike Globus or rsync, it doesn't support synchronization.

To copy `foo.txt` to Béluga:

```bash
[name@server ~]$ scp foo.txt username@beluga.alliancecan.ca:work/
```

To copy `output.dat` from Cedar to your local computer:

```bash
[name@server ~]$ scp username@cedar.alliancecan.ca:projects/def-jdoe/username/results/output.dat .
```

More SCP examples are [here](here).  Always execute the `scp` command on your *local* computer.

SCP's `-r` option recursively transfers directories and files.  Avoid `scp -r` for `/project` because it disables the setgid bit, potentially causing "Disk quota exceeded" errors.


**Note:** For custom SSH key names (other than `id_dsa`, `id_ecdsa`, `id_ed25519`, `id_rsa`), use the `-i` option:

```bash
[name@server ~]$ scp -i /path/to/key foo.txt username@beluga.alliancecan.ca:work/
```


## Prevention and Troubleshooting

### Unable to Read Data

Before transferring, ensure you can read all directories' contents.  This command lists unreadable items:

```bash
[name@server ~]$ find <directory_name> ! -readable -ls
```

### Unable to Write New Data

Check storage usage (space and file limits). Check filesystem permissions to ensure write access at the destination location.


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Transferring_data&oldid=174653](https://docs.alliancecan.ca/mediawiki/index.php?title=Transferring_data&oldid=174653)"
