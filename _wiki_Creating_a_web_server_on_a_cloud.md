# Creating a Web Server on a Cloud

This page describes the simplest case of creating a web server on our clouds using Ubuntu Linux and the Apache web server.

## Security Considerations

Any time you make a computer accessible to the public, security must be considered.  "Accessible to the public" could mean allowing SSH connections, displaying HTML via HTTP, or using 3rd party software to provide a service (e.g., WordPress). Services such as SSH or HTTP are provided by programs called "daemons", which stay running all the time on the computer and respond to outside requests on specific ports. With OpenStack, you can manage and restrict access to these ports, including granting access only to a specific IP address or to ranges of IP addresses; see [Security groups](link-to-security-groups-page). Restricting access to your VM will improve its security. However, restricting access does not necessarily remove all security vulnerabilities. If we do not use some sort of encryption when sending data (e.g., passwords), an eavesdropper can read that information.

Transport Layer Security (TLS) is the common way to encrypt this data, and any website which uses logins (e.g., WordPress, MediaWiki) should use it; see [Configuring Apache to use SSL](link-to-ssl-config-page). It is also possible that data transmitted from your web server to a client could be modified on the way by a third party if you are not encrypting it. While this might not directly cause issues for your web server, it can for your clients. In most cases, it is recommended to use encryption on your web server. You are responsible for the security of your virtual machines and should take this seriously.


## Installing Apache

Create a persistent virtual machine (see [Booting from a volume](link-to-booting-from-volume-page)) running Ubuntu Linux by following the [Cloud Quick Start](link-to-cloud-quickstart-page) instructions.

Open port 80 to allow HTTP requests into your VM by following [these instructions](link-to-port-opening-instructions) but selecting HTTP from the drop-down box instead of SSH.

While logged into your VM:

Update your apt-get repositories with the command:

```bash
sudo apt-get update
```

Upgrade Ubuntu to the latest version with the command:

```bash
sudo apt-get upgrade
```

Upgrading to the latest version of Ubuntu ensures your VM has the latest security patches.

Install the Apache web server with the command:

```bash
sudo apt-get install apache2
```

![Apache2 test page](apache-test-page.png) *(Click for larger image)*

Go to the newly created temporary Apache web page by entering the floating IP address of your VM into your browser's address bar. This is the same IP address you use to connect to your VM with SSH. You should see something similar to the Apache2 test page shown above.

Start modifying the content of the files in `/var/www/html` to create your website, specifically the `index.html` file, which is the entry point for your newly created website.


### Change the Web Server's Root Directory

It is often much easier to manage a website if the files are owned by the user who is connecting to the VM. In the case of the Ubuntu image we're using in this example, this is user `ubuntu`. Follow these steps to direct Apache to serve files from `/home/ubuntu/public_html`, for example, instead of from `/var/www/html`.

Use the command (or some other editor) to change the line `<Directory /var/www/>` to `<Directory /home/ubuntu/public_html>`:

```bash
sudo vim /etc/apache2/apache2.conf
```

Use the command to edit the line `DocumentRoot /var/www/html` to become `DocumentRoot /home/ubuntu/public_html`:

```bash
sudo vim /etc/apache2/sites-available/000-default.conf
```

Create the directory in the Ubuntu user's home directory with:

```bash
mkdir public_html
```

Copy the default page into the directory with:

```bash
cp /var/www/html/index.html /home/ubuntu/public_html
```

Then restart the Apache server for these changes to take effect with:

```bash
sudo service apache2 restart
```

You should now be able to edit the file `/home/ubuntu/public_html/index.html` without using `sudo`. Any changes you make should be visible if you refresh the page you loaded into your browser in the previous section.


## Limiting Bandwidth

If your web server is in high demand, it is possible that it may use considerable bandwidth resources. A good way to limit overall bandwidth usage of your Apache web server is to use the [Apache bandwidth module](link-to-apache-bandwidth-module-docs).


### Installing

```bash
sudo apt install libapache2-mod-bw
sudo a2enmod bw
```


### Configuring

An example configuration to limit total bandwidth from all clients to 100Mbps is:

```apache
BandWidthModule On
ForceBandWidthModule On

# Exceptions to bandwidth of 100Mbps should go here above limit
# below in order to override it

# limit all connections to 100Mbps
# 100Mbps *1/8(B/b)*1e6=12,500,000 bytes/s
BandWidth all 12500000
```

This should be placed between the `<VirtualHost></VirtualHost>` tags for your site. The default Apache site configuration is in the file `/etc/apache2/sites-enabled/000-default.conf`.


## Where to Go From Here

* [Configuring Apache to use SSL](link-to-ssl-config-page)
* [Apache2 documentation](link-to-apache2-docs)
* [w3schools HTML tutorial](link-to-w3schools-html-tutorial)


**(Remember to replace bracketed placeholders like `[link-to-security-groups-page]` with actual links.)**
