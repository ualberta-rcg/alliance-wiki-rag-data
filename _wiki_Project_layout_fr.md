# Project Layout

The `/project` spaces on the Beluga, Cedar, Graham, and Narval filesystems are organized by groups. Access to `/project` is usually through symbolic links from your `/home` directory.  These symbolic links are in the format `$HOME/projects/group_name`.

Within a group's space, the principal investigator owns the directory, and group members have read and write permissions. However, new files have read-only permissions by default. To allow write access for members, the best approach is to create a specific directory:

```bash
[name@server ~]$ mkdir $HOME/projects/def-profname/group_writable
[name@server ~]$ setfacl -d -m g::rwx $HOME/projects/def-profname/group_writable
```

For information on data sharing, file ownership, and Access Control Lists (ACLs), see [Data Sharing](Data Sharing link needed).

By default, a `/project` space has a quota of 1TB and 500,000 files; this can be increased to 10TB upon request to [Technical Support](Technical Support link needed). If your group has higher quotas due to the [Resource Allocation Competition](Resource Allocation Competition link needed), you will know your allocated quota for the year. Note that allocated storage space is cluster-specific and generally cannot be used on another cluster.

To check used and available space:

```bash
[name@server ~]$ diskusage_report
```

To ensure that files copied or moved into `/project` belong to the same group and are counted against the quota, it's helpful to set the `setgid` bit on the appropriate directory. This ensures that all new files and subdirectories inherit the same group as their parent; subdirectories also inherit `setgid`.  However, group ownership is not changed for pre-existing files and subdirectories (use `chgrp` for that), and files moved into the directory retain their group ownership. To set `setgid` on a directory:

```bash
[name@server ~]$ chmod g+s <directory name>
```

To set `setgid` on existing subdirectories:

```bash
[name@server ~]$ find <directory name> -type d -print0 | xargs -0 chmod g+s
```

For more information on `setgid`, see [this page](setgid link needed).

The `newgrp` command changes your default group during an interactive session:

```bash
[name@server ~]$ newgrp rrg-profname-ab
```

Then copy data to the appropriate `/project` directory.  However, the default group is only changed for that session; you'll need to use `newgrp` again to change the default group on your next login.

If you receive `Disk quota exceeded` error messages (see [Disk quota exceeded message](Disk quota exceeded link needed)), it might be because files are associated with the wrong group, particularly your personal group (the group with the same name as your username), which has a quota of only 2MB. To find and fix group ownership issues for these files, use:

```bash
find <directory name> -group $USER -print0 | xargs -0 chgrp -h <group>
```

where `<group>` is something like `def-profname`, a group with a reasonable quota of a terabyte or more.


## Example

In this example, Sue is the principal investigator and Bob is a member of her group. Initially, Sue's and Bob's directories appear identically structured:

`/home/sue/scratch` (symbolic link)
`/home/sue/projects` (directory)
`/home/bob/scratch` (symbolic link)
`/home/bob/projects` (directory)

However, the `scratch` symbolic link points to different directories: `/scratch/sue` for Sue and `/scratch/bob` for Bob.

Assuming Bob has only one role defined in CCDB, Bob's `project` directory would have the same content as Sue's `project` directories, and `projects` for Bob would be identical to `projects` for Sue. Also, if Sue and Bob have no other roles and are not associated with any other projects, each of their `projects` directories would contain only one subdirectory, `def-sue`.

Each of `/home/sue/project`, `/home/bob/project`, `/home/sue/projects/def-sue`, and `/home/bob/projects/def-sue` would point to the same directory, `/project/<some number>`. This directory is the best place for Sue and Bob to share data; they can create directories there and have read and write access.  So Sue can create a directory `foo`:

```bash
$ cd ~/projects/def-sue
$ mkdir foo
```

and Bob can copy files into `~/projects/def-sue/foo`, accessible to both.

Now, assuming Sue obtained resources with storage space through the resource allocation competition (as is often the case), there would be another entry in their respective `projects` directories, similar to `~/projects/rrg-sue-ab`. This directory would be used to store and share data for a project within the competition.


To share a file with a user not sponsored by the principal investigator, for example, Heather, the simplest way is to configure permissions so she can read the directory or file, usually through an Access Control List (ACL); for details, see the [Data Sharing](Data Sharing link needed) page. Note that filesystem permissions can be modified for all directories or files, not just those in your `/project` space. You can share a directory from your `/scratch` space or a specific subdirectory of your `/project` space.

It's good practice to limit file sharing to the `/project` and `/scratch` spaces.

Remember that Heather will likely need access to multiple levels of the filesystem structure; you must grant her read and write permissions for each directory between `~/projects/def-sue` and the directory containing the files to be shared. We assumed Heather has an account on the cluster, but it's also possible to share data with researchers who don't have an account with the Alliance by creating a [common drop-off point](common drop-off point link needed) in Globus.

Of course, if Heather becomes a regular collaborator with Sue, Sue could sponsor her and grant her the same access as Bob.


## Summary

*   **scratch:** Used for private and temporary files.
*   **home:** Usually used for a small number of relatively private files (e.g., job scripts).
*   **project:** Group project space, usually used for shared data as it's persistent, backed up, and relatively large (up to 10TB and more if allocated through the resource allocation competition).


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Project_layout/fr&oldid=135106](https://docs.alliancecan.ca/mediawiki/index.php?title=Project_layout/fr&oldid=135106)"
