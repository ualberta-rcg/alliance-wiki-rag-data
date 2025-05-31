# Managing Your Linux VM

The majority of researchers use the Linux Operating System on their VMs. Common Linux distributions used are AlmaLinux, CentOS, Debian, Fedora, and Ubuntu. This page will help you with some common tasks to manage your Linux VM. VMs can also run the Microsoft Windows operating system. Some Windows management tasks are described [here](link_to_windows_page).


## Linux VM User Management

There are a number of ways to allow more than one person to log into a VM. We recommend creating new user accounts and adding public SSH Keys to these accounts.

### Creating a User Account and Keys

A new user account can be created on Ubuntu with the command:

```bash
sudo adduser --disabled-password USERNAME
```

To be able to connect, the new user will need a key pair.  See [generating SSH keys in Windows](link_to_windows_keygen) or [creating a key pair in Linux or Mac](link_to_linux_mac_keygen) depending on their operating system. Then, their public key must be added to `/home/USERNAME/.ssh/authorized_keys` on the VM, ensuring permissions and ownership are correct as described in steps 2 and 3 of [Connecting using a key pair](link_to_keypair_connection).


### Granting Admin Privileges

In Ubuntu, administrative or root user privileges can be given to a new user with the command:

```bash
sudo visudo -f /etc/sudoers.d/90-cloud-init-users
```

This opens an editor where a line like `USERNAME ALL=(ALL) NOPASSWD:ALL` can be added. For more detailed information about the `visudo` command and how to edit this file, see this [DigitalOcean tutorial](link_to_digitalocean_tutorial).


### Dealing with System and Security Issues

See our guides for how to:

* [Recover data from a compromised VM](link_to_data_recovery)
* [Recover your VM from the dashboard](link_to_dashboard_recovery)


**(Note:  Please replace bracketed placeholders like `[link_to_windows_page]` with the actual links.)**
