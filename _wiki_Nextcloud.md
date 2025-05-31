# Alliance Nextcloud Service

Other languages: English, français

We provide Nextcloud, a Dropbox-like cloud storage service, for all Alliance users. You can use your Alliance username and password to log in to the [Nextcloud server](https://nextcloud.computecanada.ca). A complete [Nextcloud user manual](https://docs.nextcloud.com/server/latest/) is available from the [official Nextcloud documentation](https://docs.nextcloud.com/server/latest/). A manual is also available as a PDF document under your account once you connect. All data transfers between local devices and Alliance's Nextcloud are encrypted.

The Nextcloud service is aimed at users with relatively small datasets (up to 100 GB). For anything larger, we recommend using the [Globus](https://www.globus.org/) service.

If you are not familiar with the concept of Nextcloud, you may try the [demo on the Nextcloud website](https://nextcloud.com/).

We recommend taking this opportunity to take a look at your data and do some cleanup: remove data you no longer need, check with whom you share your data, etc.


## Alliance Nextcloud Service Description

* **Server URL:** https://nextcloud.computecanada.ca
* **Server Location:** Simon Fraser University, Burnaby, BC
* **Fixed Quota:** 100 GB per user
* **Backup Policy:** Daily backup without offsite copy
* **Access Methods:** Web interface, Nextcloud Desktop Sync Client, Nextcloud mobile apps, and any WebDAV client
* **Documentation:** [PDF](link-to-pdf) and [online](https://docs.nextcloud.com/server/latest/)


## Using the Nextcloud Web Interface

To use the web interface, log in to Alliance Nextcloud from a web browser using your Alliance username and password. You can upload and download files between Nextcloud and your mobile device or computer, edit files, and share files with other Alliance users. For more information, see the [Nextcloud user manual](https://docs.nextcloud.com/server/latest/).


## Using Nextcloud Desktop Synchronization Client and Mobile Apps

You can [download the Nextcloud Desktop Sync Client or Nextcloud mobile apps](https://nextcloud.com/install/) to synchronize data from your computer or your mobile device respectively. Once installed, the software will "sync" everything between your Nextcloud folder and your local folder. It may take some time to synchronize all data. You can make changes to files locally and they will be updated in Nextcloud automatically.


## Using WebDAV Clients

In general, you can use any WebDAV clients to "mount" a Nextcloud folder to your computer using the following WebDAV URL:

`https://nextcloud.computecanada.ca/remote.php/webdav/`

Once mounted, you can drag and drop files between the WebDAV drive and your local computer.

* **Mac OSX:** Select Go -> Connect to the Server, enter the WebDAV URL for the Server Address, and click Connect. You will be asked for your username and password to log in. After authentication, you will see a WebDAV drive on your Mac.
* **Windows:** Use the "Map Network Drive ..." option, select a drive letter, then use WebDAV URL `https://nextcloud.computecanada.ca/remote.php/webdav/` in the Folder field.
* You may also consider using Cyberduck or other clients instead. Cyberduck is available for OSX and Windows.
* **Linux:** There are many WebDAV applications available for Linux. Consult the [Nextcloud user manual](https://docs.nextcloud.com/server/latest/) for recommendations.


### Detail: WebDAV vs Synchronization Client

The WebDAV clients mount your Nextcloud storage on your computer. Files are not copied; for example, when you edit a file, you edit the original file on the Alliance Nextcloud system at Simon Fraser University.

When you connect with a Synchronization client, the first thing the client does is synchronize your files stored in the Alliance Nextcloud system with a copy of those files on your own computer. All files that are different get downloaded to your own client. When files are changed, they are re-copied to all the synchronized systems to ensure that the files are the same everywhere. The synchronization copies can take a lot of time when you (and/or your collaborators) change files frequently. The advantage is that you can work on the files offline, i.e., when you do not have network connectivity. They will be synchronized when network connectivity is re-established.


## Using UNIX Command Line Tools

You can use any available WebDAV command line clients, like `curl` and `cadaver`, to copy files between your Unix computer and Nextcloud. Command line tools are useful when you want to copy data between a remote server you log in to and Nextcloud. `curl` is usually installed on Mac OSX and Linux systems and can be used to upload and download files using a URL.


### Upload a file using `curl`

```bash
[name@server ~]$ curl -k -u <username> -T <filename> https://nextcloud.computecanada.ca/remote.php/webdav/
```

### Download a file using `curl`

```bash
[name@server ~]$ curl -k -u <username> https://nextcloud.computecanada.ca/remote.php/webdav/<filename> -o <filename>
```

### Upload and download files using `rclone`

Unlike `curl`, `rclone` lets you create a configuration once for each remote device and use it repeatedly without having to enter the service details and your password every time. The password will be stored encrypted in `~/.config/rclone/rclone.conf` on the computer or server where the `rclone` command is used.

First, [install rclone on your computer if it has a Unix-like environment](https://rclone.org/downloads/).

If used from our clusters, please note that it is not necessary to install `rclone` as it is already available:

```bash
$ [name@server ~] $ which rclone
$ /cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/bin/rclone
```

Next, configure a remote storage device profile with:

```bash
$ rclone config
```

You now have the option to edit an existing remote device, create a new remote device, delete a remote device, and so on. Let's say we want to create a new remote service profile called `nextcloud`:

1. choose "n" for "New remote"
2. Enter name for new remote --> nextcloud
3. Type of storage to configure --> 52 / WebDAV
4. URL of http host to connect to --> `https://nextcloud.computecanada.ca/remote.php/dav/files/<your CCDB username>`
5. Name of the WebDAV site/service/software you are using --> 2 / Nextcloud
6. User name --> `<your CCDB username>`
7. choose "y" for "Option pass"
8. Password --> `<your CCDB password>`
9. Leave "Option bearer_token" empty
10. choose "no" for "Edit advanced config"
11. choose "yes" for "Keep this 'nextcloud' remote"
12. choose "q" to quit config

You should now be able to see your new remote service profile in the list of configured ones with:

```bash
$ rclone listremotes
```

You can probe available disk space with:

```bash
$ rclone about nextcloud:
```

To upload a file, run:

```bash
$ rclone copy /path/to/local/file nextcloud:remote/path
```

To download a file, run:

```bash
$ rclone copy nextcloud:remote/path/file .
```


## Sharing Files Using Nextcloud

When you select a file or directory to share, type the user’s first name, last name, or username and the list of matched users registered in CCDB will be displayed in “Firstname Lastname (username)” format. Please review the name carefully as some are very similar; in doubt, enter the username which is unique. You can also share files with a group using their CCDB group name (default, RPP, RRG, or other shared groups).

To share a file with people who don’t have an Alliance account, use the **Share link** option and provide their email address. Nextcloud will send an email notification with a link to access the file.


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Nextcloud&oldid=155538](https://docs.alliancecan.ca/mediawiki/index.php?title=Nextcloud&oldid=155538)"
