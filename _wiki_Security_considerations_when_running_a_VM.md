# Security Considerations When Running a VM

On the cloud, you are responsible for the security of your virtual machines. This document is not a complete guide, but will set out some things you need to consider when creating a VM on the cloud.

## Basic Security Talk Recording

There is a recording of an ~1.5 hr talk on some basic security considerations when working with VMs in the cloud available on YouTube called [Safety First!](link-to-youtube-video).

Below is a list of links to different sections of the recording for easier video navigation:

* [Talk overview](link-to-youtube-timestamp)
* [Cloud service levels](link-to-youtube-timestamp)
* [General security principles](link-to-youtube-timestamp)
* [Key topics](link-to-youtube-timestamp)
* [Creating a first VM (with some comments about security)](link-to-youtube-timestamp)
* [OpenStack security groups](link-to-youtube-timestamp)
* [SSH Security](link-to-youtube-timestamp)
* [Logs](link-to-youtube-timestamp)
* [Creating backups of VMs](link-to-youtube-timestamp)


## Keep the Operating System Secured

* Apply security updates on a regular basis (see [Updating your VM](#updating-your-vm)).
* Avoid using packages from unknown sources.
* Use a recent image; for example, don't use Ubuntu 14.04 when Ubuntu 18.04 is available.
* Use SSH key authentication instead of passwords. Cloud instances use SSH key authentication by default, and enabling password-based authentication is significantly less secure.
* Install `fail2ban` to block brute-force attacks.


## Network Security

* Limit who can access your service. Avoid using `0.0.0.0` in the CIDR field of the security group form - in particular, don't create rules for "0.0.0.0" in the default security group, which applies automatically to all project instances.
* Be aware of the range you are opening with the netmask you are configuring.
* Do not bundle ranges of ports to allow access.
* Think carefully about your security rules. Consider the following:

**These services aren't meant to be publicly accessible:**

* ssh (22) - this service allows interactive login to your instance and MUST NOT be made publicly accessible
* RDP (3389) - this service allows interactive login to your instance and MUST NOT be made publicly accessible
* mysql (3306)
* VNC (5900-5906) - this service allows interactive login to your instance and MUST NOT be made publicly accessible
* postgresql (5432)
* nosql
* tomcat
* ... many, many others

**Some services are meant to be accessible from the internet:**

* Apache (80, 443)
* Nginx (80, 443)
* ... others

* Configure your web server to use HTTPS instead of HTTP. In many cases, HTTP should only be used to redirect traffic to HTTPS.
* Do NOT run a mail server.
* Do NOT run a BitTorrent server.


## Updating your VM

In order to keep a VM's operating system secure, it must be regularly updated - ideally weekly, or as often as new packages become available. To upgrade a Linux VM choose the commands below for your particular distribution. Note you will need to reconnect to your VM after rebooting.

### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get dist-upgrade
sudo reboot
```

### CentOS

```bash
sudo yum update
sudo reboot
```

### Fedora

```bash
sudo dnf update
sudo reboot
```


## Further Reading

An Amazon article on securing instances: [https://aws.amazon.com/articles/1233/](https://aws.amazon.com/articles/1233/)


**(Please replace the bracketed placeholders with actual links.)**
