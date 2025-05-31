# Security Considerations When Running a VM

This page is a translated version of the page [Security considerations when running a VM](link-to-english-page) and the translation is 100% complete.

Other languages:

* [English](link-to-english-page)
* français


## Basic Information

The video [Safety First!](video-link) (approximately 90 minutes) covers basic information; it is available in English only.

You can go directly to the following topics:

* Talk overview
* Cloud service levels
* General security principles
* Key topics
* Creating a first VM (with some comments about security)
* OpenStack security groups
* SSH security
* Logs
* Creating backups of VMs


## Operating System Security

Regularly perform security updates (see [Updating a Virtual Instance](#updating-a-virtual-instance) below).

Avoid using packages from untrusted sources.

Use the most recent image; for example, avoid using Ubuntu 14.04 if Ubuntu 18.04 is available.

Use default SSH key authentication; it is much more secure than passwords.

Install `fail2ban` to prevent brute-force attacks.


## Network Security

Limit access to your service. Avoid using `0.0.0.0` in the CIDR field of the security group form and, in particular, do not create rules for `0.0.0.0` for the default security group, which would allow access to all instances of the project.

Pay attention to the IP addresses made available by the `netmask` configuration.

Do not group access ports.

Pay attention to security rules, especially for:

**Services that should NOT be publicly accessible:**

* ssh (22); this service allows interactive connection with your instance and MUST NOT be publicly accessible.
* RDP (3389); this service allows interactive connection with your instance and MUST NOT be publicly accessible.
* mysql (3306)
* VNC (5900-5906); this service allows interactive connection with your instance and MUST NOT be publicly accessible.
* postgresql (5432)
* nosql
* RDP (3389)
* tomcat
* and many others

**Services that should be publicly accessible:**

* Apache (80, 443)
* Nginx (80, 443)
* and others

Configure the web server for HTTPS rather than HTTP. In many cases, HTTP should only be used to redirect to HTTPS.

Do not install a mail server.

Do not install a BitTorrent server.


## Updating a Virtual Instance

Regularly update the operating system of your instances, ideally weekly or whenever new packages are available. Use the following commands, depending on the Linux distribution. You will need to restart your instance and log in again.

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


## References

* [Tips for Securing Your EC2 Instance](amazon-link) (Amazon article).


**(End of translated content)**
