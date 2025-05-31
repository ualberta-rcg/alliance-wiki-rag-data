# Setting up a GUI Desktop on a VM

This page is a translated version of the page [Setting up GUI Desktop on a VM](https://docs.alliancecan.ca/mediawiki/index.php?title=Setting_up_GUI_Desktop_on_a_VM&oldid=130428) and the translation is 100% complete.

Other languages:

* [English](https://docs.alliancecan.ca/mediawiki/index.php?title=Setting_up_GUI_Desktop_on_a_VM&oldid=130428)
* français


Some software that you can install on your virtual machine (VM or instance) is only, or preferably, accessible via its graphical user interface (GUI). You can use a graphical interface with X11 forwarding, but you might get better performance using VNC to connect to a graphical session located on your instance.

We describe here the steps to set up a desktop interface with VNC on an instance using the Ubuntu operating system.


## Install a Desktop Environment

On your instance, install a desktop environment with a graphical interface. Several packages are available for Ubuntu:

* `ubuntu-unity-desktop`
* `ubuntu-mate-desktop`
* `lubuntu-desktop`
* `xubuntu-desktop`
* `xfce4`
* `ubuntu-desktop`
* `kde-plasma-desktop`
* `ubuntu-desktop-minimal`
* `cinnamon`
* `icewm`

This article shows some of these desktops. The following commands install a MATE desktop.

```bash
name@server ~ $ sudo apt update
name@server ~ $ sudo apt upgrade -y
name@server ~ $ sudo apt install ubuntu-mate-desktop
```

During the installation of the `ubuntu-mate-desktop` package, you must select the default session manager; the best choice would be `lightdm`. This installation can often take 15 to 30 minutes.


## Install the TigerVNC Server

This software installed on your instance allows you to use the desktop interface you installed in step 1.

```bash
name@server ~ $ sudo apt install -y tigervnc-common tigervnc-standalone-server
```

This command installs the TigerVNC server and the necessary software. For more information on VNC servers, see our [VNC wiki page](LINK_TO_VNC_WIKI_PAGE_IF_AVAILABLE).


## Start the VNC Server

```bash
name@server ~ $ vncserver
```

On the first start of the VNC server, you must enter a password that you will use to connect to the VNC desktop.  A view-only password is not required. To change your password, use the `vncpasswd` command.


## Test the Connection

Test the connection by opening port 5901 (to learn how to open a port to your OpenStack instance, see [Security Groups](LINK_TO_SECURITY_GROUPS_PAGE_IF_AVAILABLE)) and connect with a VNC client, for example, TigerVNC. This option is not secure because the data entering and leaving the instance will not be encrypted. However, this step allows you to test the client-server connection before connecting securely via an SSH tunnel; you can skip this step if you know how to configure an SSH tunnel correctly.


## Connect via an SSH Tunnel

You can consult this example which uses a compute node on our clusters.

For connecting under Linux or Mac:

1. Open your terminal.
2. In your local terminal, enter:  `ssh -i filepathtoyoursshkey/sshprivatekeyfile.key -L 5901:localhost:5901 ubuntu@ipaddressofyourVM`
3. Launch your VNC client.
4. In the VNC server field, enter `localhost:5901`.
5. The graphical desktop for your remote session should open.


Close port 5901; this port is no longer used after the connection with the VNC server is established via an SSH tunnel and it is recommended to [remove this rule in your security groups](LINK_TO_SECURITY_GROUPS_PAGE_IF_AVAILABLE).


## Stop the VNC Server

When you no longer need the desktop, stop the VNC server with:

```bash
name@server ~ $ vncserver -kill :1
```

Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Setting_up_GUI_Desktop_on_a_VM/fr&oldid=130428](https://docs.alliancecan.ca/mediawiki/index.php?title=Setting_up_GUI_Desktop_on_a_VM/fr&oldid=130428)"
