# Nextcloud

This page is a translated version of the page Nextcloud and the translation is 100% complete.

Other languages: English, français

The Nextcloud service allows data storage and sharing in a similar way to Dropbox. To connect to the server, use the username and password of your account with the Alliance. You can consult the [User Manual](link-to-user-manual) and the [Nextcloud online documentation](link-to-online-documentation). Once connected, another PDF document is also available via your account. Data between local computers and our Nextcloud service is all encrypted.

The Nextcloud service is designed for relatively modest datasets (up to 100GB). For large datasets, we recommend [Globus](link-to-globus).

To familiarize yourself with the service, see the [demo on the Nextcloud website](link-to-nextcloud-demo).

Take this opportunity to organize your files, eliminate data you no longer need, or check who your data is shared with.


## Contents

1. Description
2. Nextcloud Web Interface
3. Nextcloud Desktop Synchronization Client and Mobile Applications
4. WebDAV Clients
    * Comparison between WebDAV and Synchronization Client
5. UNIX Command Line Tools
    * Uploading a file with curl
    * Downloading a file with curl
    * Uploading or downloading a file with rclone
6. Sharing files


## Description

* **Server URL:** https://nextcloud.computecanada.ca
* **Location:** Simon Fraser University, Burnaby, BC
* **Fixed Quota:** 100 GB per user
* **Data Backup:** Once a day; no external backup copy
* **Access Methods:** Web interface, Nextcloud Desktop Synchronization client, Nextcloud mobile applications, any WebDAV client
* **Documentation:** [User manual (PDF)](link-to-pdf-manual) and [online documentation](link-to-online-documentation).


## Nextcloud Web Interface

To use the web interface, connect to Nextcloud via a browser with the username and password of your Alliance account. You will be able to download and upload files between Nextcloud and your mobile device or computer, or modify and share files with other users. For more information, see the [User Manual](link-to-user-manual).


## Nextcloud Desktop Synchronization Client and Mobile Applications

You can [download the Nextcloud Desktop Sync Client or the Nextcloud mobile applications](link-to-download) to synchronize data from your computer or mobile device, respectively. Once installed on your workstation, the client synchronizes the contents of your Nextcloud directory with the contents of the directory on your local machine. Note, however, that this operation may take some time. You can modify files locally and they will be automatically updated in Nextcloud.


## WebDAV Clients

Generally, all WebDAV clients will allow you to mount a Nextcloud directory on your computer via https://nextcloud.computecanada.ca/remote.php/webdav/.

You can then drag and drop files between the WebDAV drive and your local computer.

* **Mac OSX:** Select Go -> Connect to Server, enter https://nextcloud.computecanada.ca/remote.php/webdav/ in the Server Address field and click Connect. You must then enter your username and password to connect. After authentication, a WebDAV drive will be present on your Mac.
* **Windows:** With the Map Network Drive option, select a drive and enter https://nextcloud.computecanada.ca/remote.php/webdav/ in the Folder field.  You can also use any other client, for example Cyberduck, which is available for OSX and Windows.
* **Linux:** Several WebDAV applications are available; see the recommendations in the [User Manual](link-to-user-manual).


### Comparison between WebDAV and Synchronization Client

WebDAV clients mount your Nextcloud storage on your computer. Files are not copied, i.e., when you modify a file, what is actually modified is the original file stored in the Nextcloud system located at Simon Fraser University.

When you connect with the Nextcloud synchronization client, the client starts by synchronizing your files on Nextcloud with a copy of the files on your computer. Files that are different are downloaded to your own client. Modified files are copied back to all synchronized systems so that they are identical everywhere. Copying can take a long time if you and/or your collaborators frequently modify files. Here, the advantage is that you can work without being connected to the server and the next time you connect, the files will be synchronized.


## UNIX Command Line Tools

You can use any WebDAV command-line clients available to you, such as `curl` and `cadaver`, to copy files between your workstation and Nextcloud. Command-line tools are useful for copying data between Nextcloud and a server you connect to.  `curl` is usually installed on Mac OSX and Linux systems; it can be used to download and upload files with a URL.


### Uploading a file with curl

```bash
[name@server ~]$ curl -k -u <username> -T <filename> https://nextcloud.computecanada.ca/remote.php/webdav/
```

### Downloading a file with curl

```bash
[name@server ~]$ curl -k -u <username> https://nextcloud.computecanada.ca/remote.php/webdav/<filename> -o <filename>
```

### Uploading or downloading a file with rclone

Unlike `curl`, `rclone` allows you to create a configuration once for each storage service and use this configuration repeatedly without having to enter the host details and your password each time. The password is encrypted and stored on the computer or server where the command `~/.config/rclone/rclone.conf` is used.

First, [install rclone on your computer](link-to-rclone-install) if the environment is similar to Unix. If you are using one of our clusters, rclone is available and does not need to be installed.

```bash
$ [name@server ~] $ which rclone
$ /cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/bin/rclone
```

Then configure a remote service with the command:

```bash
$ rclone config
```

You have the option to modify an existing service and create or delete a remote service. In our example, we create a service profile named `nextcloud`.

```
choose "n"  for "New remote"
Enter name for new remote --> nextcloud
Type of storage to configure --> 52 / WebDAV
URL of http host to connect to --> https://nextcloud.computecanada.ca/remote.php/dav/files/<your CCDB username>
Name of the WebDAV site/service/software you are using --> 2 / Nextcloud
User name --> <your CCDB username>
choose "y" for "Option pass"
Password --> <your CCDB password>
Leave "Option bearer_token" empty
choose "no" for "Edit advanced config"
choose "yes" for "Keep this 'nextcloud' remote"
choose "q" to quit config
```

Your new remote service profile should now be in the list of configured profiles; to check, run:

```bash
$ rclone listremotes
```

To find out the available disk space, use:

```bash
$ rclone about nextcloud:
```

To upload a file, use:

```bash
$ rclone copy /path/to/local/file nextcloud:remote/path
```

To download a file, use:

```bash
$ rclone copy nextcloud:remote/path/file .
```


## Sharing files

When you select a file or directory that you want to share with another user registered in CCDB, enter the first name, last name, or username of that person and the list of corresponding users will be displayed. Be careful to enter this information correctly as several names are similar; if in doubt, enter the username which is unique to each person.

You can also enter the name of a CCDB group (by default, research platforms and portals, research groups and other groups where sharing is configured) to share with its members.

The Share link option also allows sharing with people who do not have an account with the Alliance; Nextcloud sends them a notification with the link to access the file.


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Nextcloud/fr&oldid=155550](https://docs.alliancecan.ca/mediawiki/index.php?title=Nextcloud/fr&oldid=155550)"
