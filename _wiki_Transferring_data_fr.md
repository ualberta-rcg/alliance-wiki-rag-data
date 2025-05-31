# Transferring Data

This page is a translated version of the page [Transferring data](https://docs.alliancecan.ca/mediawiki/index.php?title=Transferring_data&oldid=174985) and the translation is 100% complete.

Other languages: [English](https://docs.alliancecan.ca/mediawiki/index.php?title=Transferring_data&oldid=174985), français

To transfer data to or from clusters, please use copy nodes instead of login nodes.  For the URL of a copy node, see the table at the top of the pages for Beluga, Narval, Cedar, Graham, and Niagara. Globus automatically uses copy nodes.


## Between a Personal Computer and Our Equipment

To download or upload files between your computer and our infrastructure, you must use software that provides secure transfer functionality.

In a Linux or Mac OS X command-line environment, use the `scp` and `sftp` commands.

Under Windows, MobaXterm offers file transfer functions and a command-line interface via SSH; another free program for data transfer is WinSCP. To configure an SSH key connection with WinSCP, see [these instructions](LINK_TO_INSTRUCTIONS_NEEDED).

The `pscp` and `psftp` commands from PuTTY work similarly to the commands under Linux and Mac.

If it takes more than a minute to transfer files between your computer and our servers, we suggest you install and try Globus Connect Personal; see the Personal Computers section. Globus transfer can be configured and run in the background without intervention.


## Between Our Systems

Globus is the preferred tool and should be used as much as possible.

Other known transfer tools can be used for transfers between our equipment and between another computer and our equipment:

*   SFTP
*   SCP or Secure Copy Protocol
*   rsync

**Note:** To transfer files between another cluster and Niagara, use the SSH flag `-A` when connecting to the other cluster. For example, to copy files from Cedar to Niagara, the command would be:

```bash
ssh -A USERNAME@cedar.alliancecan.ca
```

Then perform the copy with:

```bash
[USERNAME@cedar5 ~]$ scp file USERNAME@niagara.alliancecan.ca:/scratch/g/group/USERNAME/
```


## From the Web

To transfer data from a website, use `wget`. Another well-known tool is `curl`. The two tools are compared in [this StackExchange article](LINK_TO_STACKEXCHANGE_ARTICLE_NEEDED) or on the [DraculaServers website](LINK_TO_DRACULASERVERS_NEEDED).  Even though our topic here is transfer between Alliance Linux systems, we want to highlight [this tutorial](LINK_TO_TUTORIAL_NEEDED) which also discusses Mac and Windows. Interrupted downloads can be resumed with `wget` and `curl` by restarting them again on the command line with `-c` and `-C -` respectively. To obtain data from cloud services such as Google cloud, Google Drive, and Google Photos, use `rclone` instead. By default, our clusters offer `wget`, `curl`, and `rclone` without having to load a module. For command-line options, see the official documentation or run the tool with the `--help` or `-h` commands.


## Synchronizing Data

Data synchronization aims to match the content of two data sources located in different places. There are several ways to do this; the most common are described here.


### Transfer with Globus

Globus is a powerful and reliable tool.

During a Globus transfer, data from the source usually overwrites the data in the destination; all data from the source is therefore transferred. In some cases, files already exist at the destination; if they are identical to those of the source, it is not necessary to transfer them. Under *Transfer Settings*, the *sync* parameter determines how Globus chooses the files to transfer.

The file selection options are:

*   **checksum is different:** examines checksums to detect a change or content error in files of the same size. This option significantly slows down the transfer but offers the greatest accuracy.
*   **file doesn't exist on destination:** transfers only files created since the last synchronization. This option is useful if your files are created incrementally.
*   **file size is different:** transfers files whose size has been modified, assuming that the content has also been modified. This option allows for a quick test.
*   **modification time is newer:** transfers only files whose source timestamp is later than that of the destination. With this option, check *preserve source file modification times*.

For more information, see the [Globus](LINK_TO_GLOBUS_PAGE_NEEDED) page.


### rsync

The `rsync` utility checks the similarity between two datasets; however, it requires considerable time when there are a large number of files, the sites are far apart, or they are on different networks. `rsync` compares the modification dates and the size of the files and performs the transfer only if one of the parameters does not match.

If the modification dates are likely to differ, the `-c` option analyzes the checksums at the source and destination and transfers only the files whose values do not match.

When you transfer data to the `/project` file systems, do not use the `-p` and `-g` flags. Quotas for `/project` are calculated according to group ownership, and preserving the same ownership could produce the error message *Disk quota exceeded*. Since `-a` includes both `-p` and `-g` by default, you must add the options `--no-g --no-p` as follows:

```bash
[name@server ~]$ rsync -avzh --no-g --no-p LOCALNAME someuser@graham.alliancecan.ca:projects/def-professor/someuser/somedir/
```

where `LOCALNAME` is a directory or file preceded by its path and where `somedir` will be created if it does not already exist. The `-z` option compresses files (whose suffixes are not in the list for the `--skip-compress` option) and requires additional CPU resources, while the `-h` option simplifies the numbers that represent the size of the files. If you are transferring very large files, add the `--partial` option so that interrupted transfers are restarted.

```bash
[name@server ~]$ rsync -avzh --no-g --no-p --partial --progress LOCALNAME someuser@graham.alliancecan.ca:projects/def-professor/someuser/somedir/
```

The `--progress` option displays the progress of the transfer of each file. For the transfer of several small files, it is preferable to display the progress of the transfer of all files.

```bash
[name@server ~]$ rsync -azh --no-g --no-p --info=progress2 LOCALNAME someuser@graham.alliancecan.ca:projects/def-professor/someuser/somedir/
```

The examples above are all transfers from a local system to a remote system. Transfers from a remote system to the `/project` directory of a local system work the same way, for example:

```bash
[name@server ~]$ rsync -avzh --no-g --no-p someuser@graham.alliancecan.ca:REMOTENAME ~/projects/def-professor/someuser/somedir/
```

where `REMOTENAME` is a directory or file preceded by its path and where `somedir` will be created if it does not already exist.

More simply, to transfer locally a directory or file (from `/home` or `/scratch`) to `/project` on the same system, do not specify the cluster name:

```bash
[name@server ~]$ rsync -avh --no-g --no-p LOCALNAME ~/projects/def-professor/someuser/somedir/
```

where `somedir` will be created if it does not already exist, before copying the contents of `LOCALNAME` into it.

In comparison, the copy command can also be used to transfer `LOCALNAME` from `/home` to `/project` as follows:

```bash
[name@server ~]$ cp -rv --preserve="mode,timestamps" LOCALNAME ~/projects/def-professor/someuser/somedir/
```

However, unlike with `rsync`, if `LOCALNAME` is a directory, it will be renamed `somedir` if `somedir` does not already exist.


### Comparing Checksums

If you cannot use Globus to synchronize two systems and if `rsync` is too slow, the two systems can be compared with a checksum utility. The following example uses `sha1sum`.

```bash
[name@server ~]$ find /home/username/ -type f -print0 | xargs -0 sha1sum | tee checksum-result.log
```

This command creates a new file named `checksum-result.log` in the current directory containing all the checksums of the files located in `/home/username/`; the sums are displayed as the process progresses.

When there are a large number of files or in the case of very large files, `rsync` can work in the background in `screen`, `tmux` or any other way allowing it to operate despite an SSH connection break.

Once the operation is complete, the `diff` utility will find the files that do not match.

```bash
[name@server ~]$ diff checksum-result-silo.log checksum-dtn.log
```

It is possible that the `find` command borrows a different order, thus detecting false differences; to counter this, run the `sort` command on both files before running `diff`, as follows:

```bash
[name@server ~]$ sort -k2 checksum-result-silo.log -o checksum-result-silo.log
[name@server ~]$ sort -k2 checksum-dtn.log -o checksum-dtn.log
```


## SFTP

To transfer files, SFTP (for Secure File Transfer Protocol) uses the SSH protocol which encrypts the data transferred.

In the following example, user `USERNAME` transfers files remotely to `ADDRESS`.

```bash
[name@server]$ sftp USERNAME@ADDRESS
```

Authentication with the `-i` option can be done using an SSH key.

```bash
[name@server]$ sftp -i /home/name/.ssh/id_rsa USERNAME@ADDRESS
```

At the `sftp>` prompt, you enter the transfer commands; use the `help` command to get the list of available commands.

Graphical applications are also available: WinSCP and MobaXterm under Windows, FileZilla under Windows, Mac, and Linux, Cyberduck under Mac and Windows.


## SCP

SCP stands for secure copy protocol. Like SFTP, SCP uses the SSH protocol to encrypt the data being transferred. Unlike Globus or rsync, SCP does not manage synchronization. The following SCP use cases are among the most frequent:

```bash
[name@server ~]$ scp foo.txt username@beluga.alliancecan.ca:work/
```

This command transfers the file `foo.txt` located in the current directory of my computer to the `$HOME/work` directory of the Beluga cluster. To transfer the file `output.dat` located in my `/project` space of the Cedar cluster to my local computer, I could use a command like:

```bash
[name@server ~]$ scp username@cedar.alliancecan.ca:projects/def-jdoe/username/results/output.dat .
```

See other examples. Note that you always run the `scp` command from your computer and not from the cluster: the SCP connection must always be initiated from your computer, regardless of the direction in which you transfer the data.

The `-r` option allows for a recursive transfer of a group of directories and files.

It is not recommended to use `scp -r` to transfer data to `/project` because the `setGID` bit is disabled in the directories that are created, which can generate errors similar to *Disk quota exceeded* when subsequently creating files; see *Disk quota exceeded* message.

**Warning:** If you use a custom SSH key name, i.e., something other than the default names `id_dsa`, `id_ecdsa`, `id_ed25519`, and `id_rsa`, you must use the `scp -i` option, followed by the path to your private key as follows:

```bash
[name@server ~]$ scp -i /path/to/key foo.txt username@beluga.alliancecan.ca:work/
```


## Preventive Measures and Troubleshooting

### Reading Problem

Make sure you can read all the contents of the directories before transferring them. Under Linux, the following command lists all the items you do not have read access to.

```bash
[name@server ~]$ find <directory_name> ! -readable -ls
```


### Problem Writing New Data

Check again the storage usage to ensure that enough space and enough files are available.

Check again the file system permissions to ensure that you have write permission to where you are transferring the new files.


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Transferring_data/fr&oldid=174985](https://docs.alliancecan.ca/mediawiki/index.php?title=Transferring_data/fr&oldid=174985)"
