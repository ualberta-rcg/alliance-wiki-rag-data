# Cloud Quick Start

This page is a translated version of the page [Cloud Quick Start](https://docs.alliancecan.ca/mediawiki/index.php?title=Cloud_Quick_Start&oldid=149098) and the translation is 100% complete.

Other languages: [English](https://docs.alliancecan.ca/mediawiki/index.php?title=Cloud_Quick_Start&oldid=149098), [français](https://docs.alliancecan.ca/mediawiki/index.php?title=Cloud_Quick_Start/fr&oldid=149099)

## Before You Begin

### Possessing a Cloud Project

You must have a cloud project to access the cloud environment. If you do not have a cloud project, see [Getting a Project in the Cloud Environment](<insert_link_here>). Once a cloud project is associated with your account, you will receive a confirmation email containing details on how to access your project; make sure you know where to find this information.

### Using a Compatible Browser

Access to cloud projects works smoothly with Firefox and Chrome browsers. Other browsers may also work well, but some are not supported by our web interface and display the message "Danger: There was an error submitting the form. Please try again." This is particularly the case with Safari on Mac; an update might solve the problem, but we recommend using Firefox or Chrome. If you still have problems, contact [technical support](<insert_link_here>).

## Creating Your First Instance

Your cloud project will allow you to create instances (also called virtual machines or VMs) that you can access from your computer via our web interface.

### Connect to the Cloud Interface to Access Your Project

The link to this interface is in the confirmation email that was sent to you. Click on the link to open your project in your browser. If your browser is not compatible, open a compatible browser and paste the URL into the address bar. If you know the name of the cloud where your project is located but don't have its URL, see the list in [Cloud Resources](<insert_link_here>). Log in with your credentials (username and password) and not with your email address.

### Consult the OpenStack Dashboard

OpenStack is the platform that provides web access to the clouds. Once connected, the OpenStack dashboard displays your project's resources. For information on the OpenStack dashboard and navigation, see the [official OpenStack documentation](<insert_link_here>).

Below are the instructions for starting Linux and Windows instances.  The operating system is that of the instance and not the computer you are using to connect. Your prior planning should indicate the operating system you will be using; if in doubt, contact [technical support](<insert_link_here>).


### Linux

### Windows

## SSH Key Pairs

When creating an instance, password authentication is disabled for security reasons. OpenStack creates your instance with a public secure shell (SSH) key installed, and to connect, you must use this SSH key pair. If you have already used SSH keys, the public key can come from a key pair you have already created on another cloud; if so, see below [Import a Key Pair](#import-a-public-key). If you have never used an SSH key pair or do not want to use an existing pair, you must create a key pair. If you are working on Windows, see [Generate SSH Keys on Windows](#generate-ssh-keys-on-windows), otherwise, see [Use SSH Keys on Linux](#use-ssh-keys-on-linux). For more information on creating and managing keys, see [SSH Keys](<insert_link_here>).

### Import a Public Key

1. In the OpenStack menu on the left, select `Compute->Key Pairs`.
2. Click the `Import Public Key` button.
3. Enter a name for the key pair.
4. Paste your public key (currently, only RSA type SSH keys are valid).
5. Make sure the public key you paste does not contain any end-of-line characters or spaces.
6. Click the `Import Public Key` button.

It is not recommended to create key pairs in OpenStack because they are not created with a passphrase, which causes security problems.


### Launch an Instance

To create an instance, select `Compute->Instances` in the left-hand menu, then click the `Launch Instance` button. The instance creation form is displayed. You can use the specifications described in your pre-planning step or reproduce the example below.

The `Launch Instance` window presents several options:

#### Details

* **Instance Name:** Enter the instance name, without any special characters or spaces; see the [naming rules](<insert_link_here>).
* **Description:** This field is optional.
* **Availability Zone:** Leave `Any Availability Zone`.
* **Number:** Enter the number of instances to create. If you don't need multiple instances, leave the value 1.

#### Source

* **Select boot source:** For your first instance, select `Image`; see information on other options in [Boot from a Volume](<insert_link_here>).
* **Create a new volume:** Click `Yes`; the instance data will be saved in the cloud volume (persistent storage). For more information on using and managing volumes, see [Working with Volumes](<insert_link_here>).
* **Volume Size (GB):** Enter the planned size; otherwise, 30 GB is a reasonable size for the operating system and a modest amount of data. For more information on using and managing volumes, see [Working with Volumes](<insert_link_here>).
* **Delete volume on instance termination:** Click `No` to prevent the volume from being accidentally deleted. Click `Yes` if you want the volume to always be deleted with the instance.

#### Allocated and Available

* **Available:** The list under `Available` shows the images your instance can boot from. For Linux beginners, we recommend the latest Ubuntu image, but you can select one of the other Linux operating systems. To select an image, click the arrow at the end of its line and the image will be moved under `Allocated`. It is important to remember the image you selected, for example Ubuntu, Fedora, etc.

#### Flavor

* **Allocated and Available:** The flavor identifies the hardware used by your instance and therefore the memory and processing capacity. The list under `Available` shows the flavors for the boot source image. Click the > icon at the beginning of the line to see if this flavor conforms to the allocation for your project. If this resource is not sufficient, an alert will be displayed. Select another flavor and click the arrow at the end of the line to move it to the `Allocated` list. For more information, see [Instance Flavors](<insert_link_here>).

#### Networks

* Change the values only if necessary. On Arbutus, select the default network which usually starts with `def-project-name`.

#### Network Ports

* Do not change the values for now.

#### Security Groups

* The default security group should appear in the `Allocated` list. If this is not the case, move it from the `Available` list by clicking the arrow at the end of the line. For more information, see [Security Groups](<insert_link_here>).

#### Key Pairs

* Under `Available`, select the SSH key pair you created earlier and move it to the `Allocated` list by clicking the arrow at the end of the line. If you do not have a key pair, you can create or import it by clicking the buttons at the top of the window (see [SSH Key Pairs](#ssh-key-pairs) above). For information on managing and using key pairs, see [SSH Keys](<insert_link_here>).

#### Configuration

* Do not change the values for now; for information on customizing scripts, see [Using cloud-init](<insert_link_here>).

#### Server Groups

* Do not change the values for now.

#### Scheduler Hints

* Do not change the values for now.

#### Metadata

* Do not change the values for now.

Once you have checked the options and defined your instance, click the `Launch Instance` button to create your instance. The list of your instances will be displayed. The `Task` column shows the status of the current task which will probably be `Building`. Once the instance is built, the status will become `Active`, which may take a few minutes.


### Network Configuration

#### Assign a Public IP Address

Display the instances page with `Compute->Instances`. A drop-down menu is located at the end of your instance's line.

1. Click the ▼ icon at the end of the line for your instance and select `Associate Floating IP`.
2. Then, in the `Allocate Floating IP` window, click the `Allocate IP` button. If you are doing this association for the first time, click the + icon in the `Manage Floating IP Associations` window.
3. If later you need to allocate another public IP address for this instance, you can select one from the drop-down list in the `IP Address` field.
4. Click the `Associate` button.

You should now have two IP addresses in the column, one in the 192.168.X.Y format and the other, your public key. The list of your public addresses and associated projects is also located under `Network->Floating IPs`. You will need your public IP address to connect to your instance.


#### Configure the Firewall

Display the `Security Groups` page with `Network->Security Groups`.

1. On the line for the default group, click the `Manage Rules` button on the right.
2. On the rule management page, click the `+Add Rule` button.
3. In the `Rule` drop-down menu, select `SSH`.
4. Leave `CIDR` in the `Remote` field.
5. Replace the contents of the `CIDR` field with `your-ip/32`, which is the IP address of the physical computer you want to use to connect to your instance. To find out your current IP address, enter `ipv4.icanhazip.com` in your browser. To access your instance from another IP address, you can add other rules for each address. To specify a series of IP addresses, use [this tool](<insert_link_here>) to calculate your CIDR rule.
6. Click the `Add` button and the new rule will be displayed in the security group list.

**Important Points:**

* Do not delete the default security rules; the operation of your instance would be compromised (see [Security Groups](<insert_link_here>)).
* Do not modify the security rules; to do so, you must delete them and add them once modified. If you make a mistake when creating a rule for the security group, delete the rule by clicking the button to the left of the row in the security group window and add a new modified rule.
* If you change the location from which you are working (and therefore your IP address), you must add the rule described here for the new address. Note that when you change your physical workplace, for example to work from home rather than from work, you also change networks.
* If you do not have a static IP address for the network you are using, remember that it may change. If you can no longer connect to your instance after a while, check if your IP address has changed by entering `ipv4.icanhazip.com` in your browser and check if it matches what is in your security rule. If your IP address changes often but the numbers on the far left remain the same, it might be more reasonable to add a range of IP addresses rather than having to frequently modify the security rules. To determine a CIDR range, [use this tool](<insert_link_here>) or consult the [CIDR notation](<insert_link_here>).
* It may be helpful to provide a description for your security rules, such as `office` or `home`. This will allow you to know if a rule is no longer needed when you want to add a new rule to connect from home, for example.


### Connecting to Your Instance via SSH

In the first step of this guide, you saved a private key on your computer; it is important to know where to find this key because you need it to connect to your instance. You must also remember the type of image you selected (Ubuntu, Fedora, etc.) and the public IP address associated with your instance.

#### Connecting from Linux or Mac

Open a terminal and enter the command:

```bash
[name@server ~]$ ssh -i /path/where/your/private/key/is/my_key.key <user name>@<public IP of your server>
```

Where `<user name>` is the name of the user connecting and `<public IP of your VM>` is the public IP you associated with your instance in the previous step. The default username depends on the image.

| Distribution | Username |
|---|---|
| Debian | debian |
| Ubuntu | ubuntu |
| CentOS | centos |
| Fedora | fedora |
| AlmaLinux | almalinux |
| Rocky | rocky |

These default users all have sudo privileges. Direct connection to the root account via SSH is disabled.


#### Connecting from Windows

The SSH connection must be made through an interface application. We recommend MobaXTerm (see instructions below); you can also connect via PuTTY (see [Connecting to a Server with PuTTY](<insert_link_here>)).

1. Download MobaXTerm.
2. To connect:
    * Launch the MobaXterm application.
    * Click `Sessions` then `New session`.
    * Select an SSH session.
    * In the `Remote host` field, enter the public IP address of your instance.
    * Make sure the `Specify username` box is checked and enter the image type for your instance in lowercase.
    * Click the `Advanced SSH settings` tab and click the `Use private key` box.
    * Click the page icon to the right of the `Use private key` field. In the window that appears, select the key pair (.pem file) that you saved on your computer at the beginning of this guide.
    * Click OK. MobaXterm saves the information you entered to connect at other times and opens an SSH connection for your instance. An SFTP connection is also opened to allow you to drag and drop files in both directions, via the left panel.


## For More Information

* [Introduction to Linux](<insert_link_here>), on how to work on the command line under Linux
* [Virtual Instance Security](<insert_link_here>)
* [Configuring a Data Server or Web Server](<insert_link_here>)
* [Managing Cloud Resources with OpenStack](<insert_link_here>)
* [Cloud Computing Technical Glossary](<insert_link_here>)
* [Automate Instance Creation](<insert_link_here>)
* [Save an Instance](<insert_link_here>)
* [Technical Support](<insert_link_here>)


## Requesting Access to a Windows Image

To create a Windows instance on one of our clouds, you must first request access to a Windows image by contacting [technical support](<insert_link_here>).

Access to a Windows Server 2012 image and a username will be provided; this access is valid for a 180-day evaluation period. It may be possible to associate a Windows license with an instance created with the evaluation image, but we do not provide these licenses.


## SSH Key Pair

### Create a Key Pair

Windows instances encrypt administrator account passwords with a public key. The corresponding private key is used for decryption.

It is recommended to create a new key pair with OpenStack rather than importing an existing key pair. To do this:

1. In the left menu, click `Access and Security`.
2. Click the `Key Pairs` tab.
3. Click `+Create Key Pair`; this displays the creation window.
4. Enter the name of the key pair.
5. Click the `Create Key Pair` button.
6. Save the .pem file to your disk.

If you want to use an existing key pair, first consult the [notes below](#notes-about-key-pairs).


### Launch an Instance

To create an instance, click the `Instances` option in the left menu, then the `Launch Instance` button. The instance creation form is displayed.

#### Details Tab

* **Availability Zone:** Only the `nova` zone is available; keep this name.
* **Instance Name:** Enter the name of your instance respecting the [naming conventions](<insert_link_here>).
* **Flavor:** The flavor determines the hardware characteristics of the instance; select `p2-3gb`. The Windows image is rather demanding and requires a large capacity bootable drive. Type c flavors have root disks of only 20GB while type p flavors offer more capacity. The RAM of the smallest type p flavor is 1.5GB, which from experience is not enough to operate Windows well. The performance of the instance will be better if you use a slightly larger flavor such as `p2-3gb`.
* **Number of instances:** Number of instances to create.
* **Instance boot source:** Source used to launch the instance; select `Boot from image (creates a new volume)`.
* **Image name:** Name of the Windows image allocated to you.
* **Device size:** Size of the root disk; enter 30GB or more. At the end, the operating system occupies about 20GB, but more space is required for the preparatory steps.
* **Delete on Termination:** If this box is checked, the volume created with the instance is deleted when the instance is terminated. In general, it is not recommended to check the box since the volume can be deleted manually and the instance can be terminated without deleting the volume.
* **Project Limits:** In the progress bars, the green color shows the proportion of resources used by the instance that will be launched. The red color indicates that the flavor uses more resources than those allocated to the project. The blue shows the resources used by the project.

#### Access and Security Tab

* **Key Pairs:** Select your SSH key pair. If there is only one key pair, it is displayed by default. If you do not have a key pair, refer to the [SSH Key Pair](#ssh-key-pair) section above.
* **Security Groups:** Make sure the `default` box is checked.

#### Network Boot Tab

* Do not modify the contents of this field. Information relating to networks will be presented after the launch of the instance.

#### Post-Creation Tab

* Do not modify the contents of this field.

#### Advanced Options Tab

* Do not modify the `Automatic` option in the `Disk Partitioning` field.

After checking the contents of all fields, click `Start` to launch the instance. The list of instances is displayed and the `Task` column shows the current task of the instance; initially, the `Task` column will probably show `Block Device Mapping`. Once the instance is created and the boot started, the `Power State` column shows `Active`. To create the volume, copy the image and start the boot, it will take at least 10 minutes.


### Localization and Licensing

The first boot of the instance will not be completed until the localization, language, and keyboard settings are selected and you have accepted the license terms via the OpenStack dashboard console.

To display the console:

1. In the left menu, click the `Instances` option.
2. Click the name of the Windows instance.
3. Click the `Console` tab and wait for the console to be displayed.

If nothing is displayed on the console, the screen may be in sleep mode; click on the screen or press a key on the keyboard to reactivate the screen. Since the cursor is often slow to react, use the keyboard keys instead.

* The tab key to select the fields.
* The up and down arrows to select the options.
* Enter the first letters of the country or region to position the drop-down menu near the selection.

To finish, press the tab key until the `next` field is selected and press Enter. You will be asked to accept the license terms. Press the tab key until the `I accept` field is selected. Press Enter. The instance will restart and the console will display a login screen with the date and time (UTC).


### Network

Under the `Instances` tab is the list of instances with the corresponding IP addresses. Each instance has at least one private IP address, but some instances may also have a second public IP address.

#### Private IP Address

When you create an OpenStack project, a local network is created for you. This network is used for communication between instances as well as for communication between instances and outside the project. A private IP address does not allow access to the instance from outside. For each instance created within a project, the network associates a private address that is unique to it; this address is in the 192.168.X.Y format.

#### Public IP Address

Public IP addresses allow external tools and services to contact the instance, for example to perform management tasks or to provide web content. Domain names can also point to a public IP address.

To assign a public IP address to an instance, click the ▼ icon to expand the menu in the `Actions` column, then select `Associate Floating IP`. If you are doing this for the first time, your project has not yet received an external IP address. You must press the + button; this displays the `Manage Floating IP Associations` window. There is only one group of public addresses and the appropriate group will be selected by default; click the `Associate` button. The `Allocate Floating IP` window is displayed and shows the IP address and port of its NAT; click the `Allocate IP` button.


#### Firewall and Rules Allowing the Remote Desktop Protocol (RDP)

To connect to your instance with a remote client, you must first allow the RDP protocol.

1. In the left menu, select `Access and Security`. Under the `Security Groups` tab, select the `default` group and click the `Manage Rules` button.
2. In the rule management window, click the `+Add Rule` button.
3. There is a predefined rule for RDP; select this rule in the drop-down menu of the `Rule` field; in the `Remote` field, leave `CIDR`.
4. In the `CIDR` field, replace `0.0.0.0/0` with your IP address. If you don't know your current IP address, you can get it by entering `ipv4.icanhazip.com` in your browser. Leaving `0.0.0.0/0` allows possible access to your instance by anyone and makes it vulnerable to brute-force attacks. To allow access for other IP addresses, add rules for these addresses or specify a group of addresses with [this tool](<insert_link_here>).
5. If you leave `0.0.0.0/0` in the `CIDR` field, the resource administrator can block all access to your instance until the security rules are adequate.
6. Finally, click the `Add` button.


### Remote Desktop Connection

To connect to a Windows instance, we will use a remote connection client. To do this, we must provide a floating IP address, a username, and a password.

#### Retrieve the Password

To retrieve the password:

1. In the left menu, click `Instances`.
2. In the drop-down menu for the instance, select `Retrieve Password`.

The password has been encrypted with the public key you selected when creating the instance. To decrypt it:

1. Display the file where your private key is located.
2. If you followed the instructions for SSH key pairs, a private key corresponding to the public key should be saved on your computer; the name has the .pem suffix.
3. Select the private key.
4. Click `Decrypt Password`.

Do not close this window as we will use the password in the next step. The password can be retrieved again by repeating this process.


#### From a Windows Client

Many versions of Windows offer remote desktop connection by default; if you cannot find this feature, you can install it from [this Microsoft site](<insert_link_here>) (installation is free).

Launch the Remote Desktop Connection and connect to your Windows instance.

1. In the `Computer` field, enter the public IP address.
2. Enter your `Username`.
3. Click the `Connect` button at the bottom of the window.
4. At the prompt, enter the password retrieved in the previous step.
5. Click `OK`.

You will probably receive a message indicating that the identity of the remote computer cannot be verified and asking if you want to continue anyway; this is normal, so answer `Yes`. Your Windows instance will be displayed in the remote desktop connection client window.

[To be completed]

TODO: The specific certificate error is "The certificate is not from a trusted certifying authority". Is seeing this alert really normal? Do we want to register the windows image certificate with a signing authority? Could we use letsencrypt or should we just ignore this issue?


#### From a Linux Client

Under Linux, you must have a remote connection client. Several clients are available; however, we recommend Remmina which seems to work well when tested with Ubuntu. Instructions for Remmina and other Linux systems including Ubuntu, Debian and Fedora can be found [on this webpage](<insert_link_here>).

Once the connection is established with your Windows instance:

1. Click `Create a new remote desktop file` (file with the green plus (+) symbol).
2. A window similar to the one shown on the right should appear.
3. In the `Server` field, enter the public IP address of your Windows instance.
4. In the `User name` field, enter your username.
5. In the `Password` field, enter the password obtained in the previous step.
6. Click `Connect`.


#### From a Mac Client

[To be completed]


### Licensing

[To be completed]

TODO: need to provide information which would be helpful for users to know what path to take to get a license. Should cover things like:

* Where to go to get a license
* What kind of license do I need/what licenses will work on the cloud
* How to apply my license to my existing cloud VM
* How to apply it to a new VM (if that is different than above bullet item)


### Notes About Key Pairs

There are different formats for key files and you have the option of protecting or not protecting your private keys with passphrases. To be able to decrypt the password for your Windows instance, your private key must be in OpenSSH format and not be protected with a passphrase. If your key pair was created by OpenStack and you downloaded the .pem key file, the private key will already be in the required format. If you created your key pair with the `ssh-keygen` command and you did not set a passphrase, the format will also most likely be correct. For more information on key pairs, see the [SSH Keys](<insert_link_here>) page.

Here is an example of a private key suitable for OpenSSH format, without a passphrase:

```
-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEAvMP5ziiOw9b5XMZUphATDZdnbFPCT0TKZwOI9qRNBJmfeLfe
...
DrzXjRpzmTb4D1+wTG1u7ucpY04Q3KHmX11YJxXcykq4l5jRZTKj
-----END RSA PRIVATE KEY-----
```

In the center, `...` replaces several lines of characters similar to the one before and the one after. The two examples of private keys below will not work for Windows instances with OpenStack.

**OpenSSH format with passphrase:**

```
-----BEGIN RSA PRIVATE KEY-----
Proc-Type: 4,ENCRYPTED
DEK-Info: DES-EDE3-CBC,CA51DBE454ACC89A

0oXD+6j5aiWIwrNMiGYDqoD0OqlURfKeQhy//FwHuyuithOSI8uwjSUqV9BM9vi1
...
8XaBb/ALqh8zLQOXEUuTstlMWXnhzBWLvu7tob0QN7pI16g3CXuOag==
-----END RSA PRIVATE KEY-----
```

**ssh.com format without passphrase:**

```
BEGIN SSH2 ENCRYPTED PRIVATE KEY ----
Comment: "rsa-key-20171130"
P2/56wAAA+wAAAA3aWYtbW9kbntzaWdue3JzYS1wa2NzMS1zaGExfSxlbmNyeXB0e3JzYS
...
QJX/qgGp0=
---- END SSH2 ENCRYPTED PRIVATE KEY ----
```


## For More Information

* [Virtual Instance Security](<insert_link_here>)
* [Creating an Instance on Linux](<insert_link_here>)
* [Managing Cloud Resources with OpenStack](<insert_link_here>)
* [Cloud Computing Technical Glossary](<insert_link_here>)
* [Automate Instances](<insert_link_here>)
* [Save Instance](<insert_link_here>)
* [Technical Support](<insert_link_here>)

Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Cloud_Quick_Start/fr&oldid=149099](https://docs.alliancecan.ca/mediawiki/index.php?title=Cloud_Quick_Start/fr&oldid=149099)"
