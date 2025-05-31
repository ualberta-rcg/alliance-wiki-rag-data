# Globus

Globus is a service that allows for fast, reliable, and secure file transfers. Designed specifically for research needs, the Globus graphical interface includes background monitoring features that automate the management of file transfers between two storage locations, whether it's our clusters or another site, a cluster located on a campus, a lab server, a desktop computer, or a laptop.

Globus uses the GridFTP transfer protocol, but allows you to avoid the complex and laborious tasks associated with it, as well as other aspects related to data movement. The service improves the performance of GridFTP, rsync, scp, and sftp protocols by automatically adjusting transfer parameters, automatically restarting when a transfer is interrupted, and verifying file integrity.

You can access the service via the Globus website or our Globus portal at [https://globus.alliancecan.ca/](https://globus.alliancecan.ca/).


## Contents

* [Utilisation](#utilisation)
    * [Lancer un transfert](#lancer-un-transfert)
    * [Options](#options)
    * [Ordinateurs personnels](#ordinateurs-personnels)
        * [Pour installer Globus Connect Personal](#pour-installer-globus-connect-personal)
        * [Pour lancer Globus Connect Personal](#pour-lancer-globus-connect-personal)
        * [Transfert entre deux points de chute personnels](#transfert-entre-deux-points-de-chute-personnels)
* [Partage de fichiers avec Globus](#partage-de-fichiers-avec-globus)
    * [Création d’une collection partagée](#création-dune-collection-partagée)
    * [Gestion des accès](#gestion-des-accès)
    * [Suppression d’une collection partagée](#suppression-dune-collection-partagée)
    * [Sécurité](#sécurité)
* [Groupes Globus](#groupes-globus)
    * [Création d’un groupe](#création-dun-groupe)
    * [Invitations](#invitations)
    * [Permissions](#permissions)
* [Interface ligne de commande (CLI)](#interface-ligne-de-commande-cli)
    * [Installation](#installation)
    * [Utilisation](#utilisation-1)
    * [Scripts](#scripts)
* [Machines virtuelles dans un nuage](#machines-virtuelles-dans-un-nuage)
    * [Globus Connect Personal](#globus-connect-personal-1)
    * [Globus Connect Server](#globus-connect-server)
* [Stockage objet sur Arbutus](#stockage-objet-sur-arbutus)
* [Soutien technique et renseignements additionnels](#soutien-technique-et-renseignements-additionnels)


## Utilisation

Since May 21, 2024, access to Globus is done with the new name of our organization. If you haven't logged into Globus after this date, close all active sessions in the Globus web interface, the command-line interface, and the Globus API. Follow the instructions to open your sessions again by selecting the new organization name.

Go to the [Alliance Globus portal](https://globus.alliancecan.ca/). Select  `Alliance de recherche numérique du Canada` or `Digital Research Alliance of Canada` (and not `Digital Research Alliance of Canada - Staff`) from the dropdown list and click `Continue`. Enter your CCDB account credentials. This will take you to the Globus web portal.

![Authentification Globus pour l'Alliance](image_placeholder)

![Page d'accueil du portail Globus de l'Alliance](image_placeholder)


### Lancer un transfert

Data transfers are done between collections (endpoints in previous versions). Collections are already defined for most of our systems. To transfer files to or from your computer, you must create a collection. Once this somewhat demanding step is accomplished, all that remains is to ensure that the Globus Connect Personal application is running on your computer to perform a transfer. See the [Ordinateurs personnels](#ordinateurs-personnels) section below.

If the `File Manager` page of the Globus portal is not displayed (see image), select it from the left bar.

![File Manager](image_placeholder)

Three `Panels` buttons are located at the top right of the page; to see two collections together, click the second button.

To find a collection, click `Search` and enter the name of the collection.

![Sélectionner une collection](image_placeholder)

To select a collection, you can start typing its name. For example, to transfer data to or from Béluga, enter `beluga`, wait two seconds, and select `computecanada#beluga-dtn` from the displayed list.

The names of all our resources have the prefix `computecanada#`, for example, `computecanada#beluga-dtn`, `computecanada#cedar-globus`, `computecanada#graham-globus`, `alliance#niagara`, or `alliance#hpss`. The abbreviation `dtn` means `data transfer node`.

Depending on the location of the collection, you may need to authenticate. For example, if you activate a collection located on Graham, you will need to enter your username and password. Authentication for one of our collections usually remains valid for a week, while that of personal collections has no expiration date.

Search to select a second collection and authenticate if required.

When a collection is active, a list of directories and files is displayed; you can double-click on the directories and use the button to navigate the structure. To select a directory or file you want to transfer, click on its name; for multiple selections, use Ctrl + click. Then click one of the large blue buttons with white arrows to start the transfer. This creates a task with a unique identifier and the transfer starts immediately; you will receive an email when the transfer is complete. To track progress and see details about the transfer, click the `Activity` button in the left bar.

![Initier un transfert](image_placeholder)

See also [How To Log In and Transfer Files with Globus](link_placeholder) on the Globus.org website.


### Options

Several other options are located in the `Transfer & Sync Options` area between the two central `Start` buttons. Here you can ask Globus to:

* synchronize, to transfer new files or modified files,
* delete files at the destination that do not exist at the source,
* preserve file modification time information,
* verify data integrity after a transfer (option selected by default),
* encrypt the transfer.

Note that the encryption function greatly reduces transfer performance and should only be used for sensitive data.


### Ordinateurs personnels

Globus provides a client for use with a Windows, macOS X, or Linux computer; see [Globus Connect Personal](#globus-connect-personal).

The Globus Connect Personal page contains links on how to configure for different operating systems, including how to proceed on the command line under Linux. If you are using Globus Connect Personal on the command line under Linux, see [this FAQ](link_placeholder) for shared paths and their permissions.


#### Pour installer Globus Connect Personal

![Trouver le bouton pour l'installation](image_placeholder)

Log in to the Alliance Globus portal, if you haven't already.

In the `File Manager` window, click the `Collections` icon in the left bar.

Click the `+ Get Globus Connect Personal` button at the top right of the window.

Click the download link for your operating system. For other operating systems, click `Show me other supported operating systems`.

Install Globus Connect Personal.

You should now have access via Globus to the endpoint. The full name is `[your username]#[name you give setup]` for example, `smith#WorkPC`.


#### Pour lancer Globus Connect Personal

The above procedure only needs to be followed once to configure the endpoint. Afterwards, to perform file transfers, make sure that Globus Connect Personal is running, i.e., launch the application and check that the endpoint is not paused.

![Globus Connect Personal pour un point de chute personnel](image_placeholder)

**Note:** The transfer will stop if the Globus Connect Personal application is closed on your endpoint during a transfer to or from that endpoint. To resume the transfer, launch the application again.


#### Transfert entre deux points de chute personnels

Even if it is possible to create endpoints for several personal computers, the transfer between two personal endpoints is not done by default. For this type of transfer, contact `globus@tech.alliancecan.ca` to create a Globus Plus account.

For more information, consult the [Globus help pages](link_placeholder), in particular:

* [Globus Connect Personal for Mac OS X](link_placeholder)
* [Globus Connect Personal for Windows](link_placeholder)
* [Globus Connect Personal for Linux](link_placeholder)


## Partage de fichiers avec Globus

Globus sharing facilitates collaboration between colleagues. The sharing function allows access to files stored on one of our systems, even if the other user does not have an account on that system. Files can be shared by anyone with a Globus account, regardless of their location. See [How To Share Data Using Globus](link_placeholder).


### Création d’une collection partagée

To share a file or folder on an endpoint, the system hosting the files must allow such sharing.

Sharing is disabled on Niagara.

**Sharing permission for the `/procjet` directory**

For clusters other than Niagara, the principal investigator must write to `globus@tech.alliancecan.ca` and indicate:

* that Globus sharing should be enabled;
* the path;
* the permission (read-only or read-write).

The data to be shared must be copied or moved to this path. Creating a symbolic link will not allow access to the data.

Otherwise, you will get the error `The backend responded with an error: You do not have permission to create a shared endpoint on the selected path. The administrator of this endpoint has disabled creation of shared endpoints on the selected path.`

Sharing is enabled for the `/home` directory. By default, sharing on `/project` is disabled to prevent users from accidentally sharing other users' files. Test the sharing function in your `/home` directory.

We suggest using a path whose name clearly indicates that files could be shared there, for example `/project/my-project-id/Sharing`

Once sharing is enabled for the path, you can create a new shared Globus endpoint for any subdirectory under that path. For example, you could create the subdirectories `/project/my-project-id/Sharing/Subdir-01` and `/project/my-project-id/Sharing/Subdir-02`

Create a different `Share` for each and share them with different users.

To have a `Share` in `/project`, write to `globus@tech.alliancecan.ca`.

With your Globus credentials, log in to the [Alliance Globus portal](https://globus.alliancecan.ca/). A transfer window will be displayed. In the `endpoint` field, enter the identifier of the endpoint you want to share (e.g., `computecanada#beluga-dtn`, `computecanada#cedar-dtn`, `computecanada#graham-globus`, `alliance#niagara`, etc.) and activate the endpoint if requested.

![Option Share](image_placeholder)

Select a directory you want to share and click the `Share` button to the right of the directory list.

![Bouton Add a Guest Collection](image_placeholder)

Click the `Add a Guest Collection` button in the upper right corner.

![Collection partagée](image_placeholder)

Enter a name that will be easily recognizable. You can also indicate where the sharing will be done with the `Browse` button.


### Gestion des accès

![Gestion des permissions pour les collections partagées](image_placeholder)

After creating a shared collection, you will see the current list of authorized accesses, which will only contain your account. Since sharing is not very useful without a second person, click the `Add Permissions -- Share With` button to add the people or groups with whom you want to share your data.

![Envoi d'une invitation de partage](image_placeholder)

In the following form, the `Path` field is used to define the sharing; since in most cases you will want to share the entire collection, this field will contain `/`. However, to share the subdirectory `Subdir-01` with specific people, enter `/Subdir-01/` or use the `Browse` button to select it.

You will then be asked to indicate whether you want to share using an email address, a username, or a group.

If you choose the username, a window will allow you to search by full name or Globus username.

The email address is a good choice if you don't know the username used by the person in question on Globus. It will also allow you to share data with people who do not have a Globus account, although they will need to create one to access the shared files.

This solution is ideal for those who already have a Globus account, as they will not have to do anything to participate in the sharing. Enter the person's name or Globus username (if you know it), choose the corresponding name from the list, and then click `Use Selected`.

The `group` choice allows you to share the file simultaneously with several people. It is possible to search by group name or its Universally Unique Identifier UUID. Since a group name can be ambiguous, make sure that the sharing is done with the desired group. This problem will be avoided by using the group's UUID, indicated on the `Groups` page (see the Groups section).

To grant read permission, click the `write` box for the group or user. Note that it is not possible to remove read access. When the form is complete, click the `Add Permission` button. It is also possible to add or remove write access by clicking in the `WRITE` box.

To remove a user or group from the sharing list, simply click the `x` at the end of the corresponding line.


### Suppression d’une collection partagée

![Removing a shared collection](image_placeholder)

When you no longer need it, you can delete the shared collection. To do this:

1. Click `Collections` on the left of the screen, then click the `Shareable by You` tab and then on the title of the collection to delete.
2. Click the `Delete Endpoint` button on the right of the screen.
3. Confirm by clicking the red button.

The collection is now deleted. This does not delete your files or those that others may have uploaded.


### Sécurité

Sharing files involves some risk. By allowing sharing, you allow others to view files that you were alone in controlling until now. Although not exhaustive, the list below lists some elements to consider before sharing.

* If you are not the owner, make sure you have the right to share the files.
* Make sure you only share files with the right people. Check if the person you are adding to the list is the one you think; some names may look alike. Remember that Globus usernames have no connection to those of the Alliance. We recommend the email address sharing method, unless you know the exact account name.
* If sharing is done with a group over which you have no control, make sure that the person who manages the group is trustworthy, as unauthorized people could be added to view your data.
* If you grant the right to modify the data, keep a backup copy of important files elsewhere than on the shared endpoint, as users of the shared endpoint may delete or modify the files, or do anything you could personally do.
* We strongly recommend that sharing be limited to a secondary directory and not apply to the top-level directory.


## Groupes Globus

Globus groups are an easy way to manage permissions for sharing with multiple users. When you create a group, you can use it from the sharing interface to control user access.


### Création d’un groupe

Click the `Groups` button in the left bar. Click the `Create New Group` button in the upper right corner. This displays the `Create New Group` window.

![Création d'un groupe Globus](image_placeholder)

Enter the group name in the `Group Name` field.

Enter the group description in the `Group Description` field.

Indicate whether the group will be visible only to its members (private group) or if all Globus users will be able to see it.

Click `Create Group` to add the group.


### Invitations

After creating the group, you can add users to it by selecting `Invite Users` and then adding their email address (preferred method) or by searching for their username. After choosing the users who are invited to join the group, click the `Add` button so that they receive a message inviting them to join. Once they have accepted the invitation, their name will appear in the group.


### Permissions

Click on a username to change its role or status. Roles grant the permissions `Admin` (all permissions), `Manager` (modify roles), and `Member` (no management permissions). Click `Save`.


## Interface ligne de commande (CLI)


### Installation

The Globus command-line interface is a Python module that installs with pip. Here is the installation procedure on one of our clusters:

1. Create a virtual environment to install the interface (see [Créer et utiliser un environnement virtuel](link_placeholder)).
   ```bash
   $ virtualenv $HOME/.globus-cli-virtualenv
   ```
2. Activate the virtual environment.
   ```bash
   $ source $HOME/.globus-cli-virtualenv/bin/activate
   ```
3. Install the interface (see [Installer des modules](link_placeholder)).
   ```bash
   $ pip install globus-cli
   ```
4. Deactivate the virtual environment.
   ```bash
   $ deactivate
   ```
To avoid having to load the virtual environment each time you use Globus, modify the path.
   ```bash
   $ export PATH=$PATH:$HOME/.globus-cli-virtualenv/bin
   $ echo 'export PATH=$PATH:$HOME/.globus-cli-virtualenv/bin' >> $HOME/.bashrc
   ```


### Utilisation

Consult the Globus [Command Line Interface (CLI)](link_placeholder) page.


### Scripts

For information on the Python API, see [Globus SDK for Python](link_placeholder).


## Machines virtuelles dans un nuage

Globus endpoints exist for clusters (Béluga, Cedar, Graham, Niagara, etc.) but not for cloud virtual machines. We cannot create a specific endpoint because there is no storage space reserved for each virtual machine.

If you need an endpoint for your virtual machine and you don't have another transfer mechanism, you can use Globus Connect Personal or Globus Connect Server.


### Globus Connect Personal

Globus Connect Personal is easier to install, manage, and pass through the firewall, but is designed to be installed on personal computers.

* [Installation for Windows](link_placeholder)
* [Installation for Linux](link_placeholder)


### Globus Connect Server

Globus Connect Server is designed for command-line environments (without a graphical interface) and has some features that you probably won't use, such as the ability to add multiple servers to an endpoint. Some ports must be opened to allow transfers (see [https://docs.globus.org/globus-connect-server/v5/#open-tcp-ports_section](https://docs.globus.org/globus-connect-server/v5/#open-tcp-ports_section)).


## Stockage objet sur Arbutus

To use object storage on Arbutus, your cloud project must have a storage allocation. The following procedure is done only once.

You must first generate the access ID and secret key with an OpenStack command-line client.

1. Import your credentials with `source <project name>-openrc.sh`.
2. Create the access key and secret key with `openstack ec2 credentials create`.
3. Log in to the Globus portal with [https://www.globus.org/](https://www.globus.org/).
4. In the `File Manager` window, enter or select `Arbutus S3 buckets`.

![Collection Arbutus S3 buckets](image_placeholder)

5. Click `Continue` to consent to data access.
6. Click `Allow`.
7. Click `Continue`. In the `AWS IAM Access Key ID` field, enter the access code generated by `openstack ec2 credentials create`; in the `AWS IAM Secret Key` field, enter the secret key.

![Arbutus S3, code d'accès et clé secrète](image_placeholder)

8. Click `Continue` to complete the configuration.


## Soutien technique et renseignements additionnels

To learn more about how we use Globus or if you need technical support for this service, write to `globus@tech.alliancecan.ca` including the following information:

* Name
* CCRI (Compute Canada Role Identifier)
* Institution
* Request or problem; don't forget to mention the source and destination sites for your transfer

Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Globus/fr&oldid=174073](https://docs.alliancecan.ca/mediawiki/index.php?title=Globus/fr&oldid=174073)"

**Note:**  Placeholder text (`image_placeholder`, `link_placeholder`) has been used where image and link URLs were missing from the original HTML.  These should be replaced with the actual URLs.
