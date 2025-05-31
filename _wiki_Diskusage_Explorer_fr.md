# Diskusage Explorer

This page is a translated version of the page [Diskusage Explorer](https://docs.alliancecan.ca/mediawiki/index.php?title=Diskusage_Explorer&oldid=149108) and the translation is 100% complete.

Other languages:

* [English](https://docs.alliancecan.ca/mediawiki/index.php?title=Diskusage_Explorer&oldid=149108)
* français


## Directory Contents

**Important:** For now, this tool is only available on Béluga and Narval.

The Diskusage Explorer tool allows you to obtain details on disk space usage in your `/home`, `/scratch`, and `/project` directories. This information is updated daily and is sorted in a SQLite format for quick access.  In our example, we will see the disk space consumption of the `def-professor` directory in `/project`.


### ncurses Interface

Select a `/project` space you have access to and want to analyze; in our example, we are analyzing `def-professor`.

```bash
[name@server ~]$ diskusage_explorer /project/def-professor
```

This command loads a browser that shows the resources consumed by all files in a directory tree.

**Navigating with the duc ncurses tool**

Enter `c` to alternate between disk space consumed and the number of files, `q` or `<esc>` to exit, and `h` for help.

To only consult a subdirectory of this `/project` space without having to navigate through the entire tree, use:

```bash
[name@server ~]$ diskusage_explorer /project/def-professor/subdirectory/
```

The command `man duc` displays a manual page.


### Graphical Interface

If the login node is particularly busy or if you have a large number of files in your `/project` space, the display may be slow and irregular. For best results, see how to use `diskusage_explorer` on your own computer.

We recommend using the standard ncurses text mode on our login nodes, but `diskusage_explorer` also includes a nice graphical interface.

First, make sure your SSH connection allows for the correct display of interface applications. You can then use a graphical interface with the command:

```bash
[name@server ~]$ duc gui -d /project/.duc_databases/def-professor.sqlite /project/def-professor
```

You can navigate with the mouse and also use `c` to switch between file size and the number of files.

**Navigating with the duc graphical interface tool**


### Navigating Faster on Your Computer

First install the `diskusage_explorer` software on your local computer, then, still on your local computer, download the SQLite file from your cluster and launch `duc`.

```bash
rsync -v --progress username@beluga.calculcanada.ca:/project/.duc_databases/def-professor.sqlite  .
duc gui -d ./def-professor.sqlite  /project/def-professor
```

This will allow you to navigate more pleasantly.


## Space Used and Number of Files per User on Cedar

On Cedar, each member of a group can obtain data on space usage and the number of files per user with the `diskusage_report` command and the `--per_user` and `--all_users` options. With the first option, the command displays only the group members who have the most files and/or who occupy the most space. By adding the second option, we obtain the usage for all members. This command helps identify users who could better manage their data.

In the next example, the user `user01` runs the command and obtains the following result:

```bash
[user01@cedar1 ~]$ diskusage_report --per_user --all_users
```

```
Description	Space	# de fichiers
/home (user user01)	109k/50G	12/500k
/scratch (user user01)	4000/20T	1/1000k
/project (group user01)	0/2048k	0/1025
/project (group def-professor)	9434G/10T	497k/500k

Breakdown for project def-professor (Last update: 2023-05-02 01:03:10)
User	File count	Size	Location
-------------------------------------------------------------------------
user01	28313	4.00 GiB	On disk
user02	11926	3.74 GiB	On disk
user03	14507	6121.03 GiB	On disk
user04	4010	377.86 GiB	On disk
user05	125929	262.75 GiB	On disk
user06	201099	60.51 GiB	On disk
user07	84806	1721.33 GiB	On disk
user08	26516	947.23 GiB	On disk
Total	497106	9510.43 GiB	On disk

Breakdown for nearline def-professor (Last update: 2023-05-02 01:01:30)
User	File count	Size	Location
-------------------------------------------------------------------------
user03	5	1197.90 GiB	On disk and tape
Total	5	1197.90 GiB	On disk and tape
```

This group is composed of 8 users and the result clearly shows that 4 of them have a large number of files containing little data.

```
User	File count	Size	Location
-------------------------------------------------------------------------
user01	28313	4.00 GiB	On disk
user02	11926	3.74 GiB	On disk
user05	125929	262.75 GiB	On disk
user06	201099	60.51 GiB	On disk
```

Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Diskusage_Explorer/fr&oldid=149109](https://docs.alliancecan.ca/mediawiki/index.php?title=Diskusage_Explorer/fr&oldid=149109)"
