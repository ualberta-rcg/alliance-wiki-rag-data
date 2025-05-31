# Managing Your Linux VM

Linux is widely used for virtual machines. Commonly used distributions include AlmaLinux, CentOS, Debian, Fedora, and Ubuntu.  This document provides assistance for common tasks. It's also possible to use the Windows operating system.

## Managing Users on Linux

There are several methods to allow multiple people to access a virtual machine. Our recommendation is to create new user accounts and associate them with SSH keys.

### Creating a User Account and Keys

To create a user account on Ubuntu, use the command:

```bash
[name@server ~]$ sudo adduser --disabled-password USERNAME
```

For the new user to be able to connect, they will need a key pair; depending on the operating system, see [Generating SSH Keys on Windows](link-to-windows-ssh-keygen-doc) or [Creating a Key Pair](link-to-linux-mac-ssh-keygen-doc) on Linux and Mac. Then add the public key to `/home/USERNAME/.ssh/authorized_keys` for the virtual machine and verify that the permissions and owner are correct, as indicated in steps 2 and 3 of [Connecting with a Key Pair](link-to-ssh-key-connection-doc).


### Admin Privileges

To grant admin (root) privileges to a user, use the command:

```bash
[name@server ~]$ sudo visudo -f /etc/sudoers.d/90-cloud-init-users
```

This starts an editor where you can add a line like:

```
USERNAME ALL=(ALL) NOPASSWD:ALL
```

For more information on the `visudo` command and how to edit the file, consult the [DigitalOcean tutorial](link-to-digitalocean-tutorial).


### System and Security Issues

Refer to:

* [Data Recovery from a Compromised Virtual Machine](link-to-compromised-vm-recovery-doc)
* [Virtual Machine Recovery via the Console](link-to-console-vm-recovery-doc)


**(Note:  Please replace bracketed links like `[link-to-windows-ssh-keygen-doc]` with the actual links to the relevant documentation.)**
