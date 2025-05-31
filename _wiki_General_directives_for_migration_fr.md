# General Directives for Migration

This page discusses issues related to transferring your data between our equipment and that of our regional partners.  If you need advice or additional information, contact technical support.

## Preparing for Migration

Check if the migration of your data must be performed by yourself or by our technical team. For any questions, contact technical support.

Data migration is done using Globus; if you are not yet familiar with this service, learn about its operation and ensure that it is compatible with your system. To guarantee the integrity of your data, test the operation of the tools that will be used on test data; these tools are, for example, `tar`, `gzip`, or `zip`.

Begin the migration process as early as possible. Migration time can be increased due to the amount of data to be migrated and the processing load required by the computers or the network. Transferring hundreds of gigabytes will take several hours, but allow a full day in case of difficulty. Transferring terabytes will require a few days.

### Pruning Your Files

Few of us have adopted the practice of regularly inspecting our data to remove superfluous elements. On the occasion of a major migration operation, it is important to clean up your directories and files. Transfer time is reduced accordingly, and storage space, a highly sought-after commodity, is better used.

If you keep the source code when you compile your applications, delete the intermediate files. One or the other of the commands `make clean`, `make realclean`, or `rm *.o` could be useful, depending on your `makefile`.

If you don't know the use of large files with names like `core.12345`, these are probably core dump files that can be deleted.

### Archiving and Compression

Most data transfer applications move a single large file more efficiently than several small files whose total would be equivalent. If your directories or file trees contain a large number of small files, combine them for archiving using `tar`.

In some cases, it may be advantageous to compress large files; this is the case, for example, for text files, whose size is often considerably reduced by the compression operation. However, there is not always a significant time saving in compressing a file that will be decompressed upon arrival. The following points should be considered: the space saved by compressing the file, the compression time, and the availability of bandwidth. These points are discussed in the Data Compression and transfer discussion section of [this webpage](link_to_webpage_here) produced by the US National Center for Supercomputing Applications.

If you feel that compression is advantageous, use `tar` or `gzip`.

### Eliminating Duplicates

Avoid transferring multiple files containing identical data to a new system. Some files with the same name may contain different data. Be sure to give your files unique names to prevent different data from being overwritten.

## Migration Process

As much as possible, use Globus Online to perform the transfer of your data; it is an efficient and user-friendly tool for performing this task. In case of network interruption, Globus has automatic recovery functions. We suggest you select "preserve source file modification times" in Transfer & Timer Options.  Also select "verify file integrity after transfer".

It is even more important to compress your data and avoid duplicates if you do not have Globus. If you must use `scp`, `sftp`, or `rsync`:

Prepare blocks of a few hundred gigabytes that you will transfer one block at a time. If there is an interruption, you will only have to resume the transfer operation on the affected block and the data previously transferred will not be affected. This is where a list of data to be transferred is useful.

Regularly check the progress of the transfer. One indicator to watch is the size of the files. If there has been no change for some time, it may be necessary to intervene.

If you are unable to resume the transfer operation, contact technical support.

Be patient. Even using Globus, data transfer is a time-consuming operation. It is impossible to determine the exact transfer time, but it should be known that hundreds of gigabytes will take several hours and hundreds of terabytes will take several days.

## After Migration

If you did not use Globus or if you did not select the "verify file integrity" option, make sure that the transferred data is not corrupted. A simple way is to compare the size of the source files to the size of the destination files. For a more thorough examination, use `cksum` and `md5sum` to compare the files. Those whose size or checksum does not match should be transferred again.

## Technical Support

To learn how to use archiving and compression utilities, use the Linux command `man <command>` or `<command> --help`.

Contact technical support.
