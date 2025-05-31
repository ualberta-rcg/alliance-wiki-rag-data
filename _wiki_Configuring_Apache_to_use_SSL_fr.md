# Configuring Apache to Use SSL

This page describes how to configure Apache to use SSL, covering both signed and self-signed certificates.

## Understanding SSL/TLS

The term SSL refers to both the Transport Layer Security (TLS) protocol and its predecessor, Secure Sockets Layer (SSL).  They are used to encrypt data communicated over networks. Encryption protects sensitive data transmitted over the internet, such as passwords. Even if the web server sends non-sensitive information to the client, encryption prevents third parties from intercepting and modifying the data before it reaches its destination. In most cases, SSL certificates are useful for encrypting data to and from a web server over the internet.


There are two types of certificates: third-party signed certificates and self-signed certificates. In most cases, you'll want a third-party signed certificate, which is easily obtained using Let's Encrypt (explained below). However, other situations (like testing) are better suited to self-signed certificates.  This encrypts data to and from your server; however, no third party confirms the validity of your web server, and a warning will be displayed when connecting. You probably don't want to use a self-signed certificate if your site is publicly accessible.

Once you have the certificate and the web server is configured, it is recommended to use the `ssltest` tool from ssllabs, which can suggest configuration changes to enhance security.


## Signed Certificates

A certificate signed by a Certificate Authority (CA) allows website users to verify that a third party (the CA) confirms the site's identity, preventing man-in-the-middle attacks.

Many CAs charge annual fees, unlike Let's Encrypt. An SSL certificate signed by this CA can be created and renewed automatically using the Certbot tool, which also configures your web server to use the certificate. For a quick start, see the [Certbot main page](link-to-certbot-main-page). Details are in the [Certbot documentation](link-to-certbot-documentation).

If configuring Certbot via Apache, open port 443 (TCP ingress) so Certbot can connect to the site (this is not covered in the Certbot documentation).


## Self-Signed Certificates

This section describes creating a self-signed SSL certificate and configuring Apache for encryption.  Using a self-signed certificate on a production website is not recommended; however, they are suitable for restricted local-use sites or testing environments.

The following steps describe the procedure on Ubuntu.  There will be some differences on other operating systems, particularly regarding commands, locations, and configuration file names.


### Enabling the SSL Module

Install Apache (see [Installing Apache](link-to-installing-apache)), then enable the SSL module:

```bash
sudo a2enmod ssl
sudo service apache2 restart
```

### Creating a Self-Signed SSL Certificate

```bash
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout /etc/ssl/private/server.key -out /etc/ssl/certs/server.crt
```

If prompted for a passphrase, ensure you rerun the command with the correct syntax, including the `-nodes` option. You will then need to answer the following questions (examples provided):

* Country Name (2 letter code) [AU]: CA
* State or Province Name (full name) [Some-State]: Nova Scotia
* Locality Name (eg, city) []: Halifax
* Organization Name (eg, company) [Internet Widgits Pty Ltd]: Alliance
* Organizational Unit Name (eg, section) []: ACENET
* Common Name (e.g. server FQDN or YOUR name) []: XXX-XXX-XXX-XXX.cloud.computecanada.ca
* Email Address []: `<your email>`

The answer to `Common Name` is the most important; it's your server's domain name. For a virtual machine on our cloud, replace the Xs in the example with the virtual machine's floating IP address.


### Setting Ownership and Permissions

Set the owner and permissions for the private key:

```bash
sudo chown root:ssl-cert /etc/ssl/private/server.key
sudo chmod 640 /etc/ssl/private/server.key
```

### Configuring Apache to Use the Certificate

Modify the SSL configuration file:

```bash
sudo vim /etc/apache2/sites-available/default-ssl.conf
```

Replace these lines:

```
SSLCertificateFile      /etc/ssl/certs/ssl-cert-snakeoil.pem
SSLCertificateKeyFile /etc/ssl/private/ssl-cert-snakeoil.key
```

With these lines:

```
SSLCertificateFile      /etc/ssl/certs/server.crt
SSLCertificateKeyFile /etc/ssl/private/server.key
SSLCertificateChainFile /etc/ssl/certs/server.crt
```

Ensure the `DocumentRoot` path matches the path defined in `/etc/apache2/sites-available/000-default.conf`, if SSL applies to that site.


### Enhancing Security

Redirect HTTP requests to HTTPS; require newer SSL versions; use better encryption options.  First, modify the file:

```bash
sudo vim /etc/apache2/sites-available/default-ssl.conf
```

Then add the following inside the `<VirtualHost>` tag:

```
ServerName XXX-XXX-XXX-XXX.cloud.computecanada.ca
SSLProtocol all -SSLv2 -SSLv3
SSLCipherSuite HIGH:MEDIUM:!aNULL:!MD5:!SEED:!IDEA:!RC4
SSLHonorCipherOrder on
```

Replace the Xs with the virtual machine's IP (using hyphens instead of dots). Add a redirect to the virtual server by modifying the default website configuration file:

```bash
sudo vim /etc/apache2/sites-available/000-default.conf
```

Add this line inside the `<VirtualHost>` tag:

```
Redirect permanent / https://XXX-XXX-XXX-XXX.cloud.computecanada.ca
```

### Enabling the Secure Site

```bash
sudo a2ensite default-ssl.conf
sudo service apache2 restart
```

**(Remember to replace placeholder IP addresses and email addresses with your actual values.)**
