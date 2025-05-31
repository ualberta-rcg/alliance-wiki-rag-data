# Sharing Data

**IMPORTANT NOTE:** Don't ever issue a bulk `chmod -R 777` on your home folder, or any of your folders. This is a HUGE security risk and is completely unacceptable on shared facilities such as our clusters.  It's also never necessary.

Having to share some but not all of your data with a colleague or another research group is common. Our systems provide various mechanisms to facilitate this:

*   **Same Research Group:** Use the project space that each research group shares. If your research requires creating a group on a national cluster, contact technical support; users cannot create their own groups.
*   **No Cluster Account:** Use Globus and a shared endpoint.
*   **Colleague with Cluster Account (Different Group):** Use filesystem permissions (the main topic of this page).

When sharing a file, the individual must have access to the entire directory chain from `/scratch` or `/project` to the file.  Think of a document in a safe in your apartment; giving someone the safe combination is useless if they can't access your apartment building, apartment, and bedroom.  In the filesystem, this means having execute permission for each directory between the root and the file's directory.


## Filesystem Permissions

Like most modern filesystems, those on our clusters support permissions to read, write, and execute files and directories. When you try to read, modify, delete a file, or access a directory (e.g., with `cd`), the Linux kernel verifies your rights. If not, you'll see "Permission denied".

For each filesystem object, there are three user categories:

1.  The object's owner (usually the creator).
2.  Members of the object's group (usually the same as the owner's default group).
3.  Everyone else.

Each category can have read, write, or execute rights.  This makes nine permissions per object.

You can see current permissions with:

```bash
[name@server ~]$ ls -l name_of_object
```

For example, `-rw-r--r--` means the owner can read and write, but not execute; the group and others can only read.  The owner and group are also printed.

To change permissions, use `chmod` with the user category, a plus/minus sign, and the permission (r, w, x):

*   `u`: owner
*   `g`: group
*   `o`: others
*   `a`: all

```bash
[name@server ~]$ chmod g+r file.txt   # Grant group read permission
[name@server ~]$ chmod o-x script.py  # Withdraw execute permission from others
[name@server ~]$ chmod a+r file.txt   # Grant everyone read permission
```

**Octal Notation:**  Unix filesystem permissions can also be represented using octal notation (less intuitive). Three bits represent permissions for each user category, interpreted as a number (0-7) using:  `(read_bit)*4 + (write_bit)*2 + (execute_bit)*1`.

For example, `-rw-r--r--` is `644` (owner: 4+2+0=6; group/others: 4+0+0=4).

To exercise rights, you need read and execute permission ("5" or "7" in octal) on the containing directory.  Use `chmod` with octal notation:

```bash
[name@server ~]$ chmod 770 name_of_file  # Owner: rwx; group: rwx; others: ---
```

You can only modify permissions of files/directories you own. Use `chgrp` to change the group.


### The Sticky Bit

In shared directories (like a project space), the sticky bit prevents users from deleting or renaming others' files.  Without it, users with write and execute permission can delete any file, even if they aren't the owner.

Set the sticky bit:

```bash
[name@server ~]$ chmod +t <directory name>
[name@server ~]$ chmod 1774 <directory name> # Octal notation (sticky bit + rwxrwxr--)
```

`ls -l` shows the sticky bit as "t" or "T":

```
$ ls -ld directory
drwxrws--T 2 someuser def-someuser 4096 Sep 25 11:25 directory
```

Unset the sticky bit:

```bash
[name@server ~]$ chmod -t <directory name>
[name@server ~]$ chmod 0774 <directory name> # Octal notation
```

In project spaces, the directory owner is the PI.


### Set Group ID Bit

The `setGID` bit makes new files/directories inherit the parent directory's group ownership. This is crucial for project filesystems where quotas are enforced by group.

If enabled, new files/directories will have the same group as the parent.  Enable it:

```bash
[someuser@server]$ chmod g+s dirTest
```

`ls -l` shows `s` in the group permissions.  An uppercase `S` means execute permissions are removed, but `setGID` remains.


### Set User ID Bit

The `setUID` bit is disabled on our clusters for security reasons.


## Default Filesystem Permissions

Default permissions are defined by the `umask`.  Display the current value:

```bash
[name@server ~]$ umask -S
```

`umask` only applies to *new* files.  Changing it doesn't affect existing files.  Set your own `umask` (in a session or `.bashrc`):

```bash
[name@server ~]$ umask <value>
```

| `umask` value | `umask` meaning       | Human-readable explanation                                      |
|---------------|-----------------------|-----------------------------------------------------------------|
| 077           | u=rwx,g=,o=           | Files are readable, writable, and executable by the owner only. |
| 027           | u=rwx,g=rx,o=           | Owner: rwx; Group: rx; Others: ---                             |
| 007           | u=rwx,g=rwx,o=           | Owner/Group: rwx; Others: ---                                  |
| 022           | u=rwx,g=rx,o=rx         | Owner: rwx; Group/Others: rx                                    |
| 002           | u=rwx,g=rwx,o=rx         | Owner/Group: rwx; Others: rx                                    |


A user needs execute permission on all directories in the file's path.  A file might have `o=rx`, but a user can't read/execute it if the parent directory lacks `o=x`. The user must also be a member of the file's group to use group permissions. ACLs also affect access.


### Change of the Default `umask` on Cedar, Béluga, and Niagara

In summer 2019, the default `umask` was changed on Cedar, Béluga, and Niagara to match Graham's.  This change did *not* expose files inappropriately.


### Changing the Permissions of Existing Files

To change existing files to match the new default permissions:

```bash
[name@server ~]$ chmod g-w,o-rx <file>
[name@server ~]$ chmod -R g-w,o-rx <directory> # For a whole directory
```


## Access Control Lists (ACLs)

ACLs provide finer-grained permissions than standard file permissions.  Use `getfacl` to view and `setfacl` to modify ACLs.


### Sharing Access with an Individual

To allow `smithj` read and execute access to `my_script.py`:

```bash
$ setfacl -m u:smithj:rx my_script.py
```

To allow `smithj` read and write access to a subdirectory (including new files):

```bash
$ setfacl -d -m u:smithj:rwX /home/<user>/projects/def-<PI>/shared_data
$ setfacl -R -m u:smithj:rwX /home/<user>/projects/def-<PI>/shared_data
```

*   `-d`: sets default access rules for the directory (new data).
*   `-R`: sets rules recursively (existing data).
*   `X`: execute permission only if the item is already executable.

The directory must be owned by you. Parent directories must allow execute permission for `smithj` (using `setfacl` or `chmod o+x`). When sharing project directories, use the path starting with `/project`, not `/home/<user>/projects`. Use `realpath` to find the physical path.


### Removing ACL

```bash
setfacl -bR /home/<user>/projects/def-<PI>/shared_data
```


### Data Sharing Groups

For complex scenarios, create a data sharing group.


### Creating a Data Sharing Group

1.  Email technical support to request creation (specify the name and that you should be the owner).
2.  Access ccdb.computecanada.ca/services/ and add members.


### Using a Data Sharing Group

Parent directories must have execute permissions for the group.  Add the group to the ACL:

```bash
$ setfacl -d -m g:wg-datasharing:rwx /home/<user>/projects/def-<PI>/shared_data
$ setfacl -R -m g:wg-datasharing:rwx /home/<user>/projects/def-<PI>/shared_data
```


## Troubleshooting

### Testing Read Access Recursively

```bash
[name@server ~]$ find <directory_name> ! -readable -ls
```

This lists all unreadable items.
