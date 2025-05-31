# Sharing Data

**WARNING:** Never use the command `chmod -R 777` in your directories, especially not in your `/home` directory. This would be a HUGE risk to the security of your data and is unacceptable on shared systems such as our compute clusters.  Furthermore, this command is never really necessary.

It is frequent that you need to share your data with a colleague or another research group, and our clusters offer all the means to do so.

To share data with a member of a research group you are part of, the best approach is to use the `/project` space available to group members. If you need to create a group that will use one of the national clusters, contact technical support, as users cannot create their own groups.

To share data with someone who does not have an account on the cluster you will be using, you can create a common drop-off point in Globus.

To share data with another user who has an account on the same cluster but is not part of the same group, the easiest way is to use the permissions of the file system in question, which is the main topic here.

The person with whom you want to share your data must be able to access all directories from the `/scratch` or `/project` spaces, up to the directory containing the file. To have access, for example, to a document placed in a safe located in a room of your apartment, it is not enough to provide me with the combination to open the safe; I must also be able to enter the building, then your apartment, then the room where the safe is located. In the context of a file system, this means granting the other user execute permission to all directories between the root directory (e.g., `/scratch` or `/project`) and the directory containing the file in question.


## File System Permissions

Like most modern file systems, those of our clusters have features for reading, writing, and executing files and directories. When a user tries with the `cd` command to read, modify, or delete a file or to gain access to a directory, the Linux kernel first checks the permissions. If the action is impossible, a message announces that permission is not granted.

There are three categories of users for file or directory objects in a file system:

*   The owner of the object, usually the user who created it
*   The members of the object's group, usually the same as the default group members of the owner
*   All others

Each of these categories of users can be associated with read, write, and execute permissions for the object. With three categories of users and three types of permissions, there is therefore a possibility of nine permissions that can be associated with each object.

To find out the permissions associated with an object, use:

```bash
[name@server ~]$ ls -l name_of_object
```

The output indicates the permissions associated with the owner, the group, and others. For example, `-rw-r--r--` allows the owner only read and write (`read` and `write`); reading is permitted to group members and all other users. The result also shows the name of the object owner and the group.

To modify the permissions associated with a file or directory, use the `chmod` command followed by the user category, then the plus (+) or minus (-) sign to either allocate or remove the permission, and finally, the nature of the permission, either read (`r`) for `read`, write (`w`) for `write`, or execute (`x`) for `execute`. The user categories are `u` (`user`) for the owner, `g` for the group, and `o` (`others`) for all other users of the cluster. Thus, the command:

```bash
[name@server ~]$ chmod g+r file.txt
```

grants read permission to all members of the group to which the file `file.txt` belongs, while the command:

```bash
[name@server ~]$ chmod o-x script.py
```

removes the permission to execute the file `script.py` to everyone except the owner and the group. The user category `a` is used to signify all (`all`); thus:

```bash
[name@server ~]$ chmod a+r file.txt
```

indicates that all users of the cluster can read the file `file.txt`.

For Unix permissions, many use octal notation, even though the latter is less intuitive. The permissions for a user category are represented by three bits that are interpreted as a number from 0 to 7 with the formula (`read_bit`)*4 + (`write_bit`)*2 + (`execute_bit`)*1. In our example, the octal representation would be 4+2+0 = 6 for the owner and 4+0+0 = 4 for the group and others, i.e., the value 644.

Note that to have your permissions related to a file, you must have access to the directory containing that file; you must therefore have read and execute permissions (5 and 7 in octal notation) for that directory.

To modify permissions, use the `chmod` command with the octal notation mentioned above; for example:

```bash
[name@server ~]$ chmod 770 name_of_file
```

grants all users in your group write, read, and execute permissions. Of course, you can only modify the permissions associated with a file or directory that you own. To change the group, use the `chgrp` command.


### Sticky Bit

As is often the case when a professor works with several students and collaborators, the `/project` space is in a directory shared by several users who have read, write, or execute permissions: it is therefore necessary to ensure that files and directories cannot be deleted by a user other than their owner. The Unix file system has the sticky bit feature that prevents a file from being deleted or renamed by a user other than the owner of the file or directory. Without this sticky bit, users who have read and write permissions for a directory can rename or delete all files in the directory, even if they are not the owners.

To set the `rwxrwxr--` permissions and the sticky bit on a directory, use the `chmod` command as follows:

```bash
[name@server ~]$ chmod +t <directory name>
```

or in octal notation with mode 1000, thus:

```bash
[name@server ~]$ chmod 1774 <directory name>
```

In `ls -l`, the sticky bit is represented by the letter t (or T), at the end of the permissions field, as follows:

```bash
$ ls -ld directory
drwxrws--T 2 someuser def-someuser 4096 Sep 25 11:25 directory
```

It is disabled by the command:

```bash
[name@server ~]$ chmod -t <directory name>
```

or in octal:

```bash
[name@server ~]$ chmod 0774 <directory name>
```

For the project space, the owner of the directory is the principal investigator who sponsors the students and collaborators.


### Group ID Bit

When files and directories are created in a parent directory, it is very useful in some cases to be able to automatically associate the owner or group of these new files and directories with the parent directory or the group to which they are linked. This is very important to the operation of the system files of the `/project` spaces of Cedar and Graham for example, since storage quotas are counted by group.

If the `setGID` bit is enabled for a directory, new files and subdirectories created under it inherit the owner of the group to which the directory is associated. Let's look at an example.

First check which groups `someuser` belongs to with the command:

```bash
[someuser@server]$ groups
someuser def-someuser
```

`someuser` belongs to two groups: `someuser` and `def-someuser`. In the active directory, there is a directory that belongs to the group `def-someuser`.

```bash
[someuser@server]$ ls -l
drwxrwx---  2 someuser   def-someuser       4096 Oct 13 19:39 testDir
```

If we create a file in this directory, we see that it belongs to `someuser`, the default group of `someuser`.

```bash
[someuser@server]$ touch dirTest/test01.txt
[someuser@server]$ ls -l dirTest/
-rw-rw-r-- 1 someuser   someuser    0 Oct 13 19:38 test01.txt
```

We probably don't want to be in `/project`, but we want a newly created file to have the same group as the parent directory. Enable the `setGID` permission of the parent directory as follows:

```bash
[someuser@server]$ chmod g+s dirTest
[someuser@server]$ ls -l
drwxrws---  2 someuser   def-someuser       4096 Oct 13 19:39 dirTest
```

Notice that the `x` permission of the group permissions is now `s`; new files created in `dirTest` will be associated with the same group as the parent directory.

```bash
[someuser@server]$ touch dirTest/test02.txt
[someuser@server]$ ls -l dirTest
-rw-rw-r-- 1 someuser   someuser      0 Oct 13 19:38 test01.txt
-rw-rw-r-- 1 someuser   def-someuser  0 Oct 13 19:39 test02.txt
```

If we create a directory under a directory where `setGID` is enabled, this new directory will be associated with the same group as the parent directory and `setGID` will also be enabled.

```bash
[someuser@server]$ mkdir dirTest/dirChild
[someuser@server]$ ls -l dirTest/
-rw-rw-r-- 1 someuser   someuser      0 Oct 13 19:38 test01.txt
-rw-rw-r-- 1 someuser   def-someuser  0 Oct 13 19:39 test02.txt
drwxrwsr-x 1 someuser   def-someuser  0 Oct 13 19:39 dirChild
```

It may be important to distinguish between `S` (uppercase) and `s`. The uppercase `S` indicates that execute permissions have been removed from the directory, but that `setGID` is still enabled. It is easy to confuse the two forms, which can create unexpected permission problems, such as the inability for other group members to access files in your directory.

```bash
[someuser@server]$ chmod g-x dirTest/
[someuser@server]$ ls -l
drwxrS---  3 someuser   def-someuser       4096 Oct 13 19:39 dirTest
```


### User ID Bit

The `setUID` bit does not work on our clusters. It is disabled for security reasons.


## Default File System Permissions

Default permissions are defined by the `umask` attribute. A default value is defined for each Linux system. To display this value in your session, run:

```bash
[name@server ~]$ umask -S
```

For example, the result on Graham would be:

```bash
[user@gra-login1]$ umask -S
u=rwx,g=rx,o=
```

This means that by default, the new files you create can be read, modified, and executed by yourself; they can be read and executed by members of the file's group; other users do not have access.

The `umask` attribute only applies to new files; changing `umask` does not change access permissions to existing files.

You may want to define less restrictive permissions (e.g., to allow other users to read and execute files) or more restrictive permissions (e.g., to prevent your group from reading or executing files). You can define your `umask` attribute in a session or in your `.bashrc` file with the command:

```bash
[name@server ~]$ umask <value>
```

where `<value>` is an octal value. The following table shows useful `umask` options.

| Value | Permissions          | Effect                                                                     |
|-------|----------------------|-----------------------------------------------------------------------------|
| 077   | u=rwx,g=,o=          | Files can be read, modified, and executed by the owner only.             |
| 027   | u=rwx,g=rx,o=         | Files can be read and executed by the owner and the group, but can only be modified by the owner. |
| 007   | u=rwx,g=rwx,o=        | Files can be read, modified, and executed by the owner and the group.     |
| 022   | u=rwx,g=rx,o=rx       | Files can be read and executed by everyone, but can only be modified by the owner. |
| 002   | u=rwx,g=rwx,o=rx      | Files can be read and executed by everyone, but can only be modified by the owner and the group. |


Other conditions determine access to files.

The user who wants to access a file must have execute permission for all directories in the path of that file. For example, a file might have the permissions `o=rx`, but a regular user will not be able to read or execute it if the parent directory does not also have the permission `o=x`.

The user who wants to access a file with group permissions must be a member of the file's group.

The permissions of a file or directory can be modified after their creation with `chmod`.

Access to files is also determined by Access Control Lists (ACLs).


### Modifying the Default `umask` Attribute on Cedar, Beluga, and Niagara

In the summer of 2019, we noticed that the `umask` attribute was not the same on all our servers. As of October 16, 2019, `umask` has been modified and is now the same as on Graham.

| Cluster | Previous Value | Value since 2019-10-16 |
|---------|-----------------|------------------------|
| Beluga  | 002             | 027                     |
| Cedar   | 002             | 027                     |
| Niagara | 022             | 027                     |

This means that permissions have become more restrictive for newly created files. If this is not suitable, you can modify your `umask` in your `.bashrc`. In general, however, we recommend keeping the default permissions.

Your files were not more at risk before this modification. From the beginning, access permissions are restrictive for your `/home`, `/project`, and `/scratch` directories; they cannot be accessed by other users unless you have granted them execute permission.


### Changing the Permissions of Existing Files

To make the permissions the same as the new default permissions, you can use `chmod` as follows:

```bash
[name@server ~]$ chmod g-w,o-rx <file>
```

For the entire directory, use:

```bash
[name@server ~]$ chmod -R g-w,o-rx <directory>
```


## Access Control Lists

### Sharing Data with Another User

Unix-like operating systems have been working with these permissions for many years, but the possibilities are limited. Since there are only three categories of users (owner, group, others), how can you allow reading to a particular user who does not belong to a group? Should you then allow everyone to read the file? Fortunately, the answer is no, since in such cases, our national systems offer access control lists (ACLs) per user. The two commands to do this are:

*   `getfacl` to find out the permissions defined in the list,
*   `setfacl` to modify these permissions.


#### Sharing a Single File

For example, to grant user `smithj` read and execute permission for the file `my_script.py`, the command would be:

```bash
$ setfacl -m u:smithj:rx my_script.py
```


#### Sharing a Subdirectory

To grant read and write access to a single user in a subdirectory, including new files that will be created there, use the following commands:

```bash
$ setfacl -d -m u:smithj:rwX /home/<user>/projects/def-<PI>/shared_data
$ setfacl -R -m u:smithj:rwX /home/<user>/projects/def-<PI>/shared_data
```

**Note:** The uppercase `X` attribute gives execute permission only when the directory or file already has execute permission. To be visible, a directory must have execute permission.

The first command determines the access rules for the directory `/home/<user>/projects/def-<PI>/shared_data`; all files and directories that will be created there will inherit the same ACL rule. It is necessary for *new* data. The second command determines the ACL rules for the directory `/home/<user>/projects/def-<PI>/shared_data` and all current content. It only applies to *existing* data.

For this method to work, it is necessary that:

*   You are the owner of the directory, `/home/smithj/projects/def-smithj/shared_data` in our example;
*   The parent directories (and parent's parents, etc.) of the one you want to share grant execute permission to the user with whom you want to share it. In our example, you could use `setfacl -m u:smithj:X ...` or grant permission to all users with `chmod o+x ...`. It is not necessary to grant public read permission. In particular, you will need to grant execute permission for the directory (`/projects/def-<PI>`) either to all users or to each user (one at a time) with whom you want to share your data.
*   To share a directory from the `/project` file system, give your collaborators a path that starts with `/project` and not `/home/<user>/projects`. The latter path contains symbolic links (simlinks or shortcuts) to the physical directories of `/project` and the directory to be found cannot be reached by others who would not have access to your `/home` directory. The `realpath` command allows you to obtain the physical path to which the simlink points. For example, `realpath /home/smithj/projects/def-smithj/shared_data` could return `/project/9041430/shared_data`. The physical path to a `/project` directory is not the same for all our clusters; if your `/project` directory needs to be shared on more than one cluster, check the physical path on each one with `realpath`.


#### Removing Access Control Lists

To recursively remove all attributes in a directory, use:

```bash
setfacl -bR /home/<user>/projects/def-<PI>/shared_data
```


### Sharing Data with a Group

In cases of more complex data sharing (with multiple users on multiple clusters), it is possible to create a *sharing group*. This is a special group composed of the users with whom the data must be shared. The group obtains its access permissions via Access Control Lists (ACLs).

You will need a group in special cases of data sharing.


#### Creating a Data Sharing Group

The following procedure describes the creation of the `wg-datasharing` group.

1.  Write to technical support to request the creation of the group; indicate the name of the group and say that you are the owner.
2.  When you receive confirmation of the group's creation, go to `ccdb.computecanada.ca/services/`.
3.  Click on the name of the group in question to display the details of that group.
4.  Add a member (e.g., Victor Van Doom with his CCI identifier vdv-888).


#### Using a Data Sharing Group

As with sharing with a single user, the parent directories of the data you want to share must have execute permission, either for everyone or for the group with which you want to share them. This means that in the `/project` directory, the principal investigator must consent as follows (unless you have permission to do this yourself):

```bash
$ chmod o+X /project/def-<PI>/
```

or

```bash
$ setfacl -m g:wg_datasharing:X /project/def-<PI>/
```

Finally, you can add your group to the Access Control List (ACL) for the directory you want to share. The commands are the same as those for sharing with a user:

```bash
$ setfacl -d -m g:wg-datasharing:rwx /home/<user>/projects/def-<PI>/shared_data
$ setfacl -R -m g:wg-datasharing:rwx /home/<user>/projects/def-<PI>/shared_data
```


## Troubleshooting

### Checking Your Read Rights

To find out the files and subdirectories to which you do *not* have read rights, use the command:

```bash
[name@server ~]$ find <directory_name> ! -readable -ls
```
