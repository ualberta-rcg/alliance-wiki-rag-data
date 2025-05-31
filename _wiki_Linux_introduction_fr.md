# Introduction to Linux

This article is aimed at Windows or Mac users with little or no experience in a UNIX environment. It should give you the basics needed to access and quickly use compute servers.

Connecting to servers uses the SSH protocol in text mode. You do not have a graphical interface, but a console. Note that Windows executables do not work on our servers without the use of an emulator.

SHARCNET offers a self-paced training tutorial; click on [Introduction to the Shell](link_to_tutorial_here).


## Getting Help

Generally, commands are documented in the reference manuals available on the servers. You can access them from the terminal with:

```bash
[name@server ~]$ man command
```

`man` uses `less` (see section "Viewing and Editing Files"); you must press `q` to exit the program.

By convention, executables themselves contain help on how they should be used.  Generally, you get the information with the command line arguments `-h`, `--help`, or in some cases `-help`, for example:

```bash
[name@server ~]$ ls --help
```

## Navigating the System

Upon connection, you will be directed to the `$HOME` directory (UNIX term for "folder" or "directory") of your user account.

When creating an account, `$HOME` contains nothing other than configuration files that are hidden, i.e., those prefixed with a dot (`.`).

In a Linux system, it is strongly discouraged to create files or directories whose names contain spaces or special characters; these special characters include accents.


### Listing the Contents of a Directory

To list the files of a directory in a terminal, use the `ls` (list) command:

```bash
[name@server ~]$ ls
```

To include hidden files:

```bash
[name@server ~]$ ls -a
```

To sort the results by date (from newest to oldest) rather than alphabetically:

```bash
[name@server ~]$ ls -t
```

To get detailed information about files (access permissions, owner, group, size, and last modification date):

```bash
[name@server ~]$ ls -l
```

The `-h` option gives the file sizes in a human-readable format.  Options can be combined, for example:

```bash
[name@server ~]$ ls -alth
```


### Navigating the File System

To move around the file system, use the `cd` (change directory) command.  Thus, to move into `my_directory`, enter:

```bash
[name@server ~]$ cd my_directory
```

To go back to the parent directory, enter:

```bash
[name@server ~]$ cd ..
```

Finally, to return to the root of your user account (`$HOME`):

```bash
[name@server ~]$ cd
```


### Creating and Deleting Directories

To create a directory, use the `mkdir` (make directory) command:

```bash
[name@server ~]$ mkdir my_directory
```

To delete a directory, use the `rmdir` (remove directory) command:

```bash
[name@server ~]$ rmdir my_directory
```

Deleting a directory with this method will only work if it is empty.


### Deleting Files

Files are deleted with the `rm` (remove) command:

```bash
[name@server ~]$ rm my_file
```

You can recursively delete a directory:

```bash
[name@server ~]$ rm -r my_directory
```

The (potentially dangerous!) `-f` option can be useful to override delete confirmation requests and continue the operation after an error.


### Copying and Renaming Files or Directories

To copy a file, use the `cp` (copy) command:

```bash
[name@server ~]$ cp source_file destination_file
```

To recursively copy a directory:

```bash
[name@server ~]$ cp -R source_directory destination_directory
```

To rename a file or directory, use the `mv` (move) command:

```bash
[name@server ~]$ mv source_file destination_file
```

This command can also move a directory.  Then replace `source_file` with `source_directory` and `destination_file` with `destination_directory`.


## File Permissions

A UNIX system has three levels of permissions: read (`r`), write (`w`), and execute (`x`). For a file, the file must be readable to be read, writable to be modified, and executable to be executed (if it is an executable or a script). For a directory, read permission is required to list its contents, write permission to modify its contents (add or delete a file), and execute permission to modify the directory.

Permissions apply to three types of users: the owner (`u`), the group (`g`), and all other people (`o`). To know the permissions associated with the files and subdirectories of the current directory, use the command:

```bash
[name@server ~]$ ls -la
```

The first 10 characters of each line indicate the permissions. The first character indicates the file type:

* `-`: a regular file
* `d`: a directory
* `l`: a symbolic link

Then, from left to right, we find the read, write, and execute permissions of the owner, group, and other users. Here are some examples:

* `drwxrwxrwx`: a directory accessible for reading and writing by everyone
* `drwxr-xr-x`: a directory that can be listed by everyone, but where only the owner can add or delete files
* `-rwxr-xr-x`: a file executable by everyone, but which can only be modified by its owner
* `-rw-r--r--`: a file readable by everyone, but which can only be modified by its owner
* `-rw-rw----`: a file that can be read and modified by its owner or its group
* `-rw-------`: a file that can only be read or modified by its owner
* `drwx--x--x`: a directory that can only be listed or modified by its owner, but through which everyone can pass to reach a deeper subdirectory
* `drwx-wx-wx`: a directory in which everyone can write, but whose contents only the owner can list

It is important to note that to be able to read or write to a directory, it is necessary to have execute (`x`) access in all parent directories, up to the root `/` of the file system. Thus, if your personal directory has permissions `drwx------` and contains a subdirectory with permissions `drwxr-xr-x`, other users will not be able to read the contents of this subdirectory because they do not have execute access to the parent directory.

The `ls -la` command then gives a number, followed by the name of the file owner, the name of the file group, the size of the file, the date of its last modification, and its name.

The `chmod` command allows you to modify the permissions associated with a file. The simple way to use it is to specify which permissions you want to add or remove to which type of user. Thus, you specify the list of users (`u` for the owner, `g` for the group, `o` for other users, `a` for all three options), followed by a `+` to add a permission or a `-` to remove a permission, and followed by the list of permissions to modify (`r` for read, `w` for write, `x` for execute).  Permissions not specified are not affected. Here are some examples:

Prevent group members and other users from reading or modifying the `secret.txt` file:

```bash
[name@server ~]$ chmod go-rwx secret.txt
```

Allow everyone to read the `public.txt` file:

```bash
[name@server ~]$ chmod a+r public.txt
```

Make the `script.sh` file executable:

```bash
[name@server ~]$ chmod a+x script.sh
```

Allow group members to read and write to the `share` directory:

```bash
[name@server ~]$ chmod g+rwx share
```

Prevent other users from reading the contents of their personal directory:

```bash
[name@server ~]$ chmod go-rw ~
```


## Viewing and Editing Files


### Viewing a File

To view a read-only file, use the `less` command:

```bash
[name@server ~]$ less file_to_view
```

Use the keyboard arrows or the mouse wheel to move around the document. You can search for a term in the document by entering `/term_to_search`. Exit by pressing `q`.


### Comparing Two Files

The `diff` command shows the differences between two files:

```bash
[name@server ~]$ diff file1 file2
```

The `-y` option displays the files side-by-side.


### Searching in a File

The `grep` command searches for a given expression in a file:

```bash
[name@server ~]$ grep 'tata' file1
```

Or multiple files:

```bash
[name@server ~]$ grep 'tata' fich*
```

Note that, under Linux, the `*` character can replace zero, one, or a series of characters. The `?` character replaces (exactly) one character.

The search text can also contain variables. For example, to search for the text "No. 10" or "No. 11", etc. with any number between 10 and 29, you can use the command:

```bash
[name@server ~]$ grep 'No. [1-2][0-9]' file
```

The search text must be in the form of a regular expression. To learn more about regular expressions, see [Regular Expression Documentation](link_to_regex_doc_here).

