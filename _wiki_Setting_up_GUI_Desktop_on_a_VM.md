# Setting up GUI Desktop on a VM

Other languages: English, français

Some software installed on your virtual machine (VM, or instance) is best accessed through its graphical user interface (GUI). While SSH + X11 forwarding works, VNC offers potentially better performance for remote desktop connections.  These instructions are for a VM running Ubuntu.

## Install a GUI Desktop on your VM

Many desktop packages are available for Ubuntu.  Examples include:

*   `ubuntu-unity-desktop`
*   `ubuntu-mate-desktop`
*   `lubuntu-desktop`
*   `xubuntu-desktop`
*   `xfce4`
*   `ubuntu-desktop`
*   `kde-plasma-desktop`
*   `ubuntu-desktop-minimal`
*   `cinnamon`
*   `icewm`

[This article](link_to_article_needed) shows examples of these desktops.  Below are commands to install the MATE desktop:

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install ubuntu-mate-desktop
```

During installation, you'll choose a default display manager; `lightdm` is a good option.  Installation may take 15-30 minutes.

## Install TigerVNC Server

TigerVNC runs on your VM, enabling remote GUI access using client software.

```bash
sudo apt install -y tigervnc-common tigervnc-standalone-server
```

This installs the TigerVNC server and supporting software.  See our [VNC documentation](link_to_vnc_docs_needed) for details on VNC servers and clients.

## Start the VNC Server

```bash
vncserver
```

You'll be prompted to set a password.  A view-only password isn't required. Use `vncpasswd` later to change your password.

## Test Your Connection

Open port 5901 (see [security groups](link_to_security_groups_needed) for opening ports on OpenStack VMs) and connect using a VNC viewer like TigerVNC.  This connection is insecure (unencrypted).  It's for testing before securely connecting with an SSH tunnel (next step). You can skip this step if you're comfortable setting up an SSH tunnel.


## Connect Using an SSH Tunnel ([SSH_tunnelling](link_to_ssh_tunnelling_needed))

An example of creating an SSH tunnel to a VNC server on a cluster compute node is [available here](link_to_ssh_tunnel_example_needed).

Here's how to connect using an SSH tunnel on Linux or macOS:

1.  Open your terminal.
2.  Type this command, replacing placeholders:

```bash
ssh -i filepathtoyoursshkey/sshprivatekeyfile.key -L 5901:localhost:5901 ubuntu@ipaddressofyourVM
```

3.  Start your VNC viewer.  Enter `localhost:5901` in the VNC server field.
4.  Your remote GUI desktop should open.

## Close Port 5901

After connecting via SSH tunnel, close port 5901 in your security groups ([security groups](link_to_security_groups_needed)).

## Stop the VNC Server

To stop the VNC server:

```bash
vncserver -kill :1
```

**(Retrieved from "https://docs.alliancecan.ca/mediawiki/index.php?title=Setting_up_GUI_Desktop_on_a_VM&oldid=130426")**
