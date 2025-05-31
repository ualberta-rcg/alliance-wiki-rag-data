# FTP Server in the Cloud

This page is a translated version of the page FTP server in the Cloud and the translation is 100% complete.

Other languages:

* English
* français

## Better Alternatives to FTP

If you can use a protocol other than FTP, there are better alternatives.

In a context where FTP accepts anonymous connections for read-only access, use HTTP (see Creating a web server on a cloud); for read/write access, as it is extremely risky to accept files transferred anonymously, contact technical support; knowing your specific case, we can help you find a secure solution.

If you want FTP users to be authenticated with usernames and passwords, SFTP is a more secure and easier option; FTPS, an extension of FTP, uses TLS (Transport Layer Security) to encrypt incoming and outgoing data.

When authentication is done by password, the transmitted data should be encrypted to prevent a skilled person from decoding the password. We strongly recommend not allowing password access to your instance (or VM for virtual machine) since any machine connected to the internet is at risk of brute-force attacks. Authentication by SSH keys is preferable and works with SFTP.


## Configuring an FTP Server

If you must use FTP, consult one of the following guides, depending on the operating system:

* Ubuntu
* CentOS 6

The ports of an instance used by FTP must be opened; see Security Groups to learn how to open ports. FTP uses port 21 to initiate the file transfer request, but the transfer itself can take place on a random port above port 1025; however, the details vary depending on the FTP operating mode (for example, port 20 could be used). This means that to allow FTP access to your instance, you must open port 21, possibly port 20, and probably ports above 1025. Each open port represents a security risk and protocols other than FTP should be preferred.

For more information on the ports used by FTP, read this article.

Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=FTP_server_in_the_Cloud/fr&oldid=140196](https://docs.alliancecan.ca/mediawiki/index.php?title=FTP_server_in_the_Cloud/fr&oldid=140196)"
