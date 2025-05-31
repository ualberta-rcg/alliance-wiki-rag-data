# Configuring WSL as a ControlMaster Relay Server

This document describes how to use WSL's ControlMaster to connect to clusters with multiple native Windows applications for a certain duration, without having to authenticate for each session.  This is a draft; we are working on its final version. If you have suggestions, please contact technical support.

## Contents

* [Install Linux on Windows with WSL](#install-linux-on-windows-with-wsl)
* [Install other software](#install-other-software)
* [Configuration overview](#configuration-overview)
* [Connect to the Ubuntu VM and create the `custom_ssh` directory](#connect-to-the-ubuntu-vm-and-create-the-custom_ssh-directory)
* [Modify `.ssh/config` on Ubuntu](#modify-sshconfig-on-ubuntu)
* [Modify authorized keys](#modify-authorized-keys)
* [Start the sshd server on Ubuntu](#start-the-sshd-server-on-ubuntu)
* [Modify `.ssh/config` on `smart` with `RemoteCommand`](#modify-sshconfig-on-smart-with-remotecommand)
* [Connect to Cedar](#connect-to-cedar)
* [Alternative configuration option](#alternative-configuration-option)
* [Configuration with MobaXterm](#configuration-with-mobaxterm)


## Install Linux on Windows with WSL

See [Working with Windows Subsystem for Linux (WSL)](link-to-wsl-docs-here)

In the configuration files:

* The distribution is Ubuntu.
* The hostname for the WSL instance is `ubuntu`; `/etc/hostname` contains `ubuntu` and `/etc/hosts` contains `127.0.0.1 localhost ubuntu`.
* The Windows system name is `smart`, and the connection is made by the user named `jaime`.
* The username for the Ubuntu VM is also `jaime`.
* The username for the Alliance is `pinto`, and we want to connect to Cedar.


## Install other software

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install openssh-server -y
```

You can connect to Ubuntu from Windows with `ssh localhost`.


## Configuration overview

```
[ssh client] ----> [ssh relay server] ----> [ssh target server]
your Windows     modified authorized_keys     using cedar for
  machine          in your Ubuntu VM           this exercise
 smart             ubuntu                 Cedar
```


## Connect to the Ubuntu VM and create the `custom_ssh` directory

```bash
jaime@ubuntu:~$ cat custom_ssh/sshd_config
Port 2222
HostKey /home/jaime/custom_ssh/ssh_host_ed25519_key
HostKey /home/jaime/custom_ssh/ssh_host_rsa_key
AuthorizedKeysFile /home/jaime/custom_ssh/authorized_keys
ChallengeResponseAuthentication no
UsePAM no
Subsystem sftp /usr/lib/openssh/sftp-server
PidFile /home/jaime/custom_ssh/sshd.pid
```

To copy the `ssh_host` keys from `/etc/ssh`, use:

```bash
sudo cp /etc/ssh/ssh_host_ed25519_key /home/jaime/custom_ssh/
```


## Modify `.ssh/config` on Ubuntu

```bash
jaime@ubuntu:~$ cat ~/.ssh/config
Host cedar
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlMaster auto
    ControlPersist 10m
    HostName cedar.alliancecan.ca
    User pinto
```


## Modify authorized keys

```bash
jaime@ubuntu:~/custom_ssh$ cat /home/jaime/custom_ssh/authorized_keys
ssh-ed25519 AAAZDINzaC1lZDI1NTE5AAC1lZDIvqzlffkzcjRAaMQoTBrPe5FxlSAjRAaMQyVzN+A+
```

Use the SSH public key you downloaded from CCDB.


## Start the sshd server on Ubuntu

```bash
jaime@ubuntu:~/custom_ssh$ /usr/sbin/sshd -f ${HOME}/custom_ssh/sshd_config
```

Make sure the server is started with your profile and not with the root profile (`root`). You will need to start the sshd server each time you restart your computer or WSL is closed and launched again.


## Modify `.ssh/config` on `smart` with `RemoteCommand`

```bash
jaime@smart ~/.ssh cat config
Host ubuntu
        Hostname localhost
        RemoteCommand ssh cedar
```


## Connect to Cedar

```bash
jaime@smart ~
$ ssh -t ubuntu -p 2222
Enter passphrase for key '/home/jaime/.ssh/id_ed25519':
Last login: Fri Mar 22 10:50:12 2024 from 99.239.174.157
================================================================================
Welcome to Cedar! / Bienvenue sur Cedar!
...
...
...
[pinto@cedar1 ~]$
```


## Alternative configuration option

You can also customize the authorized keys for Ubuntu and the Windows `~/.ssh/config` file so that some graphical applications work without having to specify `RemoteCommand` (e.g., WinSCP). In this case, `RemoteCommand` is specified for the public key.

```bash
jaime@ubuntu:~/custom_ssh$ cat /home/jaime/custom_ssh/authorized_keys
command="ssh cedar" ssh-ed25519 AAAZDINzaC1lZDI1NTE5AAC1lZDIvqzlffkzcjRAaMQoTBrPe5FxlSAjRAaMQyVzN+A+

jaime@smart ~/.ssh cat config
Host ubuntu
        Hostname localhost
        #RemoteCommand ssh cedar
```

Subsequently, you can still use `ssh ubuntu -p 2222` from a Windows shell.


## Configuration with MobaXterm

(Further details on configuring with MobaXterm would go here)


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Configuring_WSL_as_a_ControlMaster_relay_server/fr&oldid=151927](https://docs.alliancecan.ca/mediawiki/index.php?title=Configuring_WSL_as_a_ControlMaster_relay_server/fr&oldid=151927)"
