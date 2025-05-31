# Storage and File Management

## Introduction

We offer many storage options to meet the needs of a wide range of domains. Depending on your specific needs and usage, you can choose from various solutions ranging from long-term storage to high-speed temporary local storage. In most cases, our file systems are shared resources and should be used responsibly; indeed, dozens or even hundreds of users can be affected by a single individual who behaves irresponsibly. These file systems are designed for storing a limited number of very large files, usually binary files, since very large text files (hundreds of MB and more) are not easily readable by a human being; for this reason, you should avoid storing thousands of small files of a few megabytes, especially in the same directory. A better approach would be to use commands such as `tar` or `zip` to convert a directory of several small files into a very large archive file; see [Archiving and Compressing Files](Archiving and Compressing Files).

It is your responsibility to check how long your data has been stored. The role of most file systems is not to provide a long-term archiving service; therefore, you must move files and directories that are no longer used to another location, whether on your personal computer or another storage resource that you control. Transferring large amounts of data is usually done with Globus.

Note that storage resources are not for your personal data, but for research data.

When your account is created on a cluster, your `home` directory contains references to your `project` and `scratch` spaces via symbolic links, shortcuts to these other file systems from your `home` directory. Note that these symbolic links may only appear a few hours after your first connection. You own your own `home` and `scratch` spaces, while the `project` space is shared by the research group. This group may consist of users who have accounts linked to that of the principal investigator or accounts of members of a resource allocation.

A user can therefore have access to several different `project` spaces associated with one or more principal investigators, and the `project` directories in their `home` directory contain the symbolic links to these different `project` spaces. All accounts have access to one or more `project` spaces. The `projects` directory in your account contains a symbolic link to each `project` space you have access to.

For a user whose account is linked to a single principal investigator's account, the default `project` space is the same default `project` space as that of the principal investigator's account.

For a user whose account is linked to several accounts, the default `project` space is the same as that of the principal investigator who has the largest number of accounts associated with them.

All users can check the available disk space and disk space used by the `project`, `home`, and `scratch` file systems with the `diskusage_report` command-line utility, available on the clusters. To do this, log in to the cluster via SSH; at the prompt, enter `diskusage_report` and then press the Enter key. The utility produces a report similar to this:

```
# diskusage_report
                   Description                Space           # of files
                 Home (username)         280 kB/47 GB              25/500k
              Scratch (username)         4096 B/18 TB              1/1000k
       Project (def-username-ab)       4096 B/9536 GB              2/500k
          Project (def-username)       4096 B/9536 GB              2/500k
```

For a more detailed report, use the Diskusage Explorer tool.


## Types of Storage

Our resources include various file systems for storage; make sure you use the appropriate space for a particular need. Here are the main file systems in our infrastructure, some of their characteristics, and the needs for which they are designed.

**HOME**: It may seem logical to store all your files and do all your work in your `home` directory; however, the quota for this directory is relatively small and performance is limited for reading and writing large amounts of data. This directory is more suitable for source code, small parameter files, and job submission scripts.

**PROJECT**: The quota for the `project` space is much larger and well-suited for data sharing between members of a group since, unlike `home` or `scratch`, it is linked to a professor's account and not to that of a particular user. The data saved here should be relatively static, i.e., it will be modified rarely during a month; frequently modifying this data or frequently renaming or moving files could represent too heavy a load for the tape backup system.

**SCRATCH**: This type of storage is the best choice for intensive read/write operations of large files (> 100MB per file). However, be aware that important data must be copied elsewhere because there is no backup copy on `scratch` and older files are likely to be purged. This space should only be used for temporary files such as checkpoint files, job output data, or other data that can be easily recreated.

Do not use SCRATCH to store all your files. This space is designed for temporary files that you can lose without much consequence.

**SLURM_TMPDIR**: While a job is running, the environment variable `$SLURM_TMPDIR` contains the unique path to a temporary directory of a fast local file system on each of the compute nodes reserved for that job. This directory is deleted with its contents when the job ends, so this variable should only be used for temporary files used during job execution. The advantage of this file system is that performance is better because it is located locally on the compute node. It is particularly suitable for large collections of small files (< 1 MB per file). Jobs share this space on each node, and the available capacity depends on the technical characteristics of each. For more information, see [Local Storage on Compute Nodes](Local Storage on Compute Nodes).


## Per-User /project Space Consumption

For `/home` and `/scratch`, the `diskusage_report` command gives the space and inode usage for each user, while for `/project`, it gives the total quota of the group, including all member files. Since files belonging to a user can be located anywhere in `/project`, it is difficult to obtain the exact amount of files and the amount of files per user or per project when a user has access to several projects. An estimate of the space and inode usage per user in the total `/project` space can however be obtained with the command `lfs quota -u $USER /project`.  In addition, an estimate of the number of files in a directory (and its subdirectories) can be obtained with the `lfs find` command, for example `lfs find <path to the directory> -type f | wc -l`.


## Best Practices

Regularly clean up data in the `project` and `scratch` spaces, as these file systems are used for immense data collections.

Use only text files of less than a few megabytes.

As much as possible, reserve `scratch` storage and local storage for temporary files. For local storage, you can use the temporary directory `$SLURM_TMPDIR` created by the scheduler for this purpose.

If the program needs to search inside a file, it is faster to read the entire file first.

If some unused files need to be kept, archive and compress them and, if possible, copy them elsewhere, for example in the nearline file system.

For more information on managing a large number of files, we recommend reading [this page](this page), especially if you are limited by the quota on the number of files.

You may have problems, regardless of the type of parallel read access to files in a file system like `home`, `scratch`, and `project`; to counter this, use a specialized tool like MPI-IO.

If the storage solutions offered do not meet your needs, contact technical support.


## Quotas and Policies

In order for all users to have sufficient space, quotas and policies are imposed on backups and the automatic purging of certain file systems.

On our clusters, each user has default access to `home` and `scratch` spaces, and each group has a default of 1TB of `project` space.

For a slight increase in `project` and `scratch` spaces, use the [fast access service](fast access service). For a significant increase in project spaces, apply within the framework of the [resource allocation competition](resource allocation competition).

To find out your quota usage for file systems on Cedar and Graham, use the `diskusage_report` command.


**(Multiple tables with identical information follow. Only one is included below for brevity.)**

| File System | Default Quota             | Based on Lustre | Copied for Backup | Purged | Available by Default | Mounted on Compute Nodes |
|-------------|--------------------------|-----------------|--------------------|--------|----------------------|-------------------------|
| `/home`     | 50GB and 500K files/user | yes              | yes                | no     | yes                   | yes                      |
| `/scratch`  | 20TB and 1M files/user   | yes              | no                 | yes    | yes                   | yes                      |
| `/project`  | 1TB and 500K files/group | yes              | yes                | no     | yes                   | yes                      |
| `/nearline` | 1TB and 5000 files/group | yes              | yes                | no     | yes                   | no                       |


*This quota is fixed and cannot be changed.*

*For more information, see the automatic purge policy.*

*The `/project` space can be increased to 40TB per group using the fast access service, provided that the quota for the project space is at least 1TB and the total for the four general-purpose clusters is at most 43TB. The request must be made by the responsible principal investigator for the group by contacting technical support.*

As of April 1, 2024, the fast access service will allow higher quotas for `/project` and `/nearline` spaces; for more information, see the Storage section in the [Fast Access Service webpage](Fast Access Service webpage). If you need more storage than what is offered by the fast access service, you will need to submit an application to the [annual resource allocation competition](annual resource allocation competition).


**(Additional tables with varying information regarding quotas and file system characteristics are present in the original document.  They are omitted here for brevity, but would be included in the final Markdown file.)**

The `home` and `project` spaces are backed up every night; copies are kept for 30 days and deleted files are kept for an additional 60 days. Note that this is different from the age limit for purging `scratch` space files. To recover a previous version of a file or directory, contact technical support, mentioning the full path to the file(s) and the date of the version.


## For More Information

* [Diskusage Explorer](Diskusage Explorer)
* [Project Directory](Project Directory)
* [Data Sharing](Data Sharing)
* [Data Transfer](Data Transfer)
* [Lustre](Lustre)
* [Archiving and Compressing Files](Archiving and Compressing Files)
* [Working with a Large Number of Files](Working with a Large Number of Files)
* [Parallel Input/Output: Tutorial](Parallel Input/Output: Tutorial)

