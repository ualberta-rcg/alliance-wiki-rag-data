# Globus File Transfer Service

**Other languages:** English, français

Globus is a service for fast, reliable, and secure file transfer. Designed for researchers, Globus offers an easy-to-use interface with background monitoring features that automate file transfers between resources (Alliance cluster, supercomputing facility, campus cluster, lab server, desktop, or laptop).

Globus uses GridFTP but simplifies GridFTP and other data movement complexities.  It improves transfer performance over GridFTP, rsync, scp, and sftp by automatically tuning settings, restarting interrupted transfers, and checking file integrity.

Globus is accessible via the [main Globus website](https://www.globus.org/) or the [Alliance Globus portal](https://portal.globus.org/).


## Using Globus

Since May 21, 2024, Globus uses our current organization name. If you haven't reopened a Globus session since then, close all active sessions (Globus Web GUI, command line interface, or applications using the Globus API) and follow these instructions:

1. Go to the [Alliance Globus portal](https://portal.globus.org/).  Your "existing organizational login" is your CCDB account. Select "Digital Research Alliance of Canada" (not "Digital Research Alliance of Canada - Staff") in the dropdown box, then click "Continue".
2. On the next page, enter your CCDB username (not email address) and password. This will take you to the Globus web portal.

![Choose Digital Research Alliance of Canada for Globus Organization dropdown](image1.png)
![Digital Research Alliance of Canada Globus authentication page](image2.png)


### To Start a Transfer

Globus transfers occur between collections (previously known as endpoints). Most Alliance clusters have pre-configured collections. To transfer files to/from your computer, create a collection. This requires initial setup, but subsequent transfers only require running Globus Connect Personal on your machine (see [Personal Computers](#personal-computers)).

If the Globus Portal's File Manager page isn't displayed, select it from the left sidebar.

![Globus File Manager](image3.png)

Select the second "Panels" button (top right) to view two collections simultaneously. Find collections by searching (top right) for a collection name.

![Selecting a Globus collection](image4.png)

For example, to transfer to/from the Béluga cluster, type "beluga", wait two seconds, and select `computecanada#beluga-dtn`.

All resources are prefixed with `computecanada#` (e.g., `computecanada#beluga-dtn`, `computecanada#cedar-globus`, `computecanada#graham-globus`, `alliancecan#niagara`, `alliancecan#hpss`).  ('dtn' stands for data transfer node).

Authentication may be required depending on the site.  For example, accessing a Graham-hosted collection requires your Alliance username and password.  Authentication typically lasts one week for CC collections; personal collections don't expire.

Select a second collection, searching and authenticating if needed.  Once activated, navigate directories and files. Highlight files/directories to transfer, then click the blue arrow buttons to initiate the transfer.  The transfer will receive a unique ID and start immediately. You'll receive an email upon completion. Monitor transfers via the "Activity" button (left sidebar).

![Initiating a transfer](image5.png)

**See also:** [How To Log In and Transfer Files with Globus](https://www.globus.org/how-to-guides/)


### Options

Globus offers transfer options between the "Start" buttons:

* Sync: Transfer only new or changed files.
* Delete: Delete destination files not present in the source.
* Preserve: Preserve source file modification times.
* Verify: Verify file integrity after transfer (default).
* Encrypt: Encrypt the transfer (significantly reduces performance).  Use only for sensitive data.


### Personal Computers

Globus Connect Personal is a desktop client for Windows, macOS, and Linux.  The Globus Connect Personal page provides installation instructions for various operating systems, including Linux command-line installation.  For Linux command-line installation, refer to the [Globus FAQ](https://docs.globus.org/faq/) for path configuration and permissions.


#### To Install Globus Connect Personal

![Finding the installation button](image6.png)

1. Go to the [Alliance Globus portal](https://portal.globus.org/) and log in.
2. In the File Manager screen, click the "Collections" icon (left sidebar).
3. Click "Get Globus Connect Personal" (top right).
4. Click the download link for your OS (click "Show me other supported operating systems" if needed).
5. Install Globus Connect Personal.

Your endpoint name will be `[your username]#[name you give setup]` (e.g., `smith#WorkPC`).


#### To Run Globus Connect Personal

This setup is done once. For file transfers, ensure Globus Connect Personal is running and the endpoint isn't paused.

![Globus Connect Personal application](image7.png)

Closing Globus Connect Personal during a transfer will stop it. Reopen the program to restart.


#### Transfer Between Two Personal Endpoints

Transfer between personal endpoints isn't enabled by default. Contact `globus@tech.alliancecan.ca` for Globus Plus account setup.

**For more information:** See the [Globus.org how-to pages](https://www.globus.org/how-to-guides/), specifically:

* [Globus Connect Personal for Mac OS X](link-to-macos-instructions)
* [Globus Connect Personal for Windows](link-to-windows-instructions)
* [Globus Connect Personal for Linux](link-to-linux-instructions)


## Globus Sharing

Globus sharing facilitates collaboration. It allows access to files on your Alliance cluster account, even without the other user having an account on that system. Files can be shared with any Globus account holder worldwide. See [How To Share Data Using Globus](link-to-sharing-guide).


### Creating a Shared Collection

Sharing requires system-level sharing enabled.

* **Globus sharing is disabled on Niagara.**
* **Project requires permission to share:** To enable sharing on `/project` for other clusters, the PI must contact `globus@tech.alliancecan.ca` with confirmation, the path, and read-only or read/write permissions. Data must be moved/copied to this path; symbolic links won't work.  Otherwise, you'll receive an error: "The backend responded with an error: You do not have permission to create a shared endpoint on the selected path. The administrator of this endpoint has disabled creation of shared endpoints on the selected path."
* Globus sharing is enabled for `/home`. Sharing on `/project` is disabled by default to prevent accidental sharing. For testing, create a share in your `/home` directory.
* We suggest using a path clearly indicating shared files (e.g., `/project/my-project-id/Sharing`).  Once enabled, create shared endpoints for subdirectories (e.g., `/project/my-project-id/Sharing/Subdir-01`, `/project/my-project-id/Sharing/Subdir-02`). Create separate shares for different users. Contact `globus@tech.alliancecan.ca` for `/project` share creation.


1. Log into the [Alliance Globus portal](https://portal.globus.org/).
2. In the endpoint field, enter the endpoint identifier (e.g., `computecanada#beluga-dtn`) and activate if needed.
3. Open the "Share" option.

![Open the Share option](image8.png)

4. Select a folder and click the "Share" button.
5. Click "Add a Guest Collection".

![Click on Add Guest Collection](image9.png)

6. Name the share and adjust the sharing path using the "Browse" button.

![Managing a shared collection](image10.png)


### Managing Access

![Managing shared collection permissions](image11.png)

The initial access list includes only your account. Click "Add Permissions -- Share With" to add users or groups.

![Send an invitation to a share](image12.png)

The path is relative to the share.  For sharing the entire collection, use `/`.  For subdirectory sharing (e.g., "Subdir-01"), use `/Subdir-01/` or browse to it.

Share via email, username, or group:

* **User:** Search by email address, name, or Globus username. Email is suitable if you don't know the Globus username.  It allows sharing with users without Globus accounts (they'll need to create one). Using a username is best if the user already has a Globus account.
* **Group:** Share with multiple users simultaneously. Search by group name or UUID (UUID is recommended to avoid ambiguity).

Enable write permissions using the checkbox. Read access cannot be removed. Click "Add Permission".  You can add/remove write permissions via the checkbox next to the user/group name.  Remove users/groups by clicking the 'x'.


### Removing a Shared Collection

![Removing a shared collection](image13.png)

1. Click "Collections" (left sidebar), then "Shareable by You".
2. Click the shared collection title.
3. Click "Delete Collection" and confirm.


### Sharing Security

Sharing involves risks. Consider the following:

* Obtain permission before sharing others' data.
* Verify recipients. Globus usernames aren't linked to Alliance usernames. Use email addresses unless you have the exact account name.
* Trust group owners if sharing with groups you don't control.
* Back up important files before granting write access, as users can delete or overwrite files.
* Restrict sharing to subdirectories, not your home directory.


## Globus Groups

Globus groups simplify permission management for multiple users.


### Creating a Group

1. Click "Groups" (left sidebar).
2. Click "Create New Group" (top right).
3. Enter the group name and description.
4. Select group visibility (private or public).
5. Click "Create Group".

![Creating a Globus group](image14.png)


### Inviting Users

Select "Invite users", enter email addresses or search for usernames, and click "Add".  Invited users will receive an email.


### Modifying Membership

Click a user to modify their role (Admin, Manager, Member) and status. Click "Save Changes".


## Command Line Interface (CLI)

### Installing

The Globus CLI is a Python module installable using pip.

1. Create a virtual environment: `virtualenv $HOME/.globus-cli-virtualenv`
2. Activate it: `source $HOME/.globus-cli-virtualenv/bin/activate`
3. Install Globus CLI: `pip install globus-cli`
4. Deactivate: `deactivate`
5. Add the virtual environment to your path (optional, for easier use):
   ```bash
   export PATH=$PATH:$HOME/.globus-cli-virtualenv/bin
   echo 'export PATH=$PATH:$HOME/.globus-cli-virtualenv/bin' >> $HOME/.bashrc
   ```


### Using

See the [Globus Command Line Interface (CLI) documentation](link-to-cli-docs).


### Scripting

Use the [Globus SDK for Python](link-to-python-sdk).


## Virtual Machines (Cloud VMs such as Arbutus, Cedar, Graham)

Globus endpoints don't exist for cloud VMs due to the lack of singular storage.  If needed and other transfer methods aren't suitable, use Globus Connect Personal or Globus Connect Server.


### Globus Connect Personal

Easier to install and manage, but designed for laptops/desktops.

* [Install Globus Connect Personal on Windows](link-to-windows-gcp-install)
* [Install Globus Connect Personal on Linux](link-to-linux-gcp-install)


### Globus Connect Server

Designed for headless installations and offers additional features (e.g., adding multiple servers to an endpoint). Requires opening ports (see [https://docs.globus.org/globus-connect-server/v5/#open-tcp-ports_section](https://docs.globus.org/globus-connect-server/v5/#open-tcp-ports_section)).


## Object Storage on Arbutus

Accessing object storage requires a cloud project with allocated object storage.  These steps are needed once:

1. Generate the access ID and secret key using the OpenStack command-line client:
   ```bash
   source <project name>-openrc.sh
   openstack ec2 credentials create
   ```
2. Log into the Globus portal ([https://www.globus.org/](https://www.globus.org/)).
3. In the File Manager, enter or select "Arbutus S3 buckets".

![Globus Arbutus S3 bucket endpoint](image15.png)

4. Click "Continue" to grant data access.
5. Click "Allow".
6. Click "Continue". Enter the access code and secret key from step 1 into the AWS IAM Access Key ID and AWS IAM Secret Key fields.

![Globus Arbutus S3 bucket Keys](image16.png)

7. Click "Continue".


## Support and More Information

For support or more information, email `globus@tech.alliancecan.ca` with:

* Name
* Compute Canada Role Identifier (CCRI)
* Institution
* Inquiry/issue (specify source and destination sites)


**(Retrieved from "https://docs.alliancecan.ca/mediawiki/index.php?title=Globus/en&oldid=173698")**

**Note:**  Replace `image1.png`, `image2.png`, etc., with actual image file names or URLs.  Also replace the bracketed links with the correct URLs.
