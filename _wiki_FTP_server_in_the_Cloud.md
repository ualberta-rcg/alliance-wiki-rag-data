# Better Alternatives to FTP

If you have the freedom to choose an alternative to FTP, consider the following options:

## If you are considering anonymous FTP...

*   **For read-only access:** Use HTTP (see [Creating a web server on a cloud](link-to-webserver-doc)).
*   **For read/write access:** The security risks of accepting anonymous incoming file transfers are very great. Please [contact us](link-to-contact-page) and describe your use case so we can help you find a secure solution.

## If you plan to authenticate FTP users (that is, require usernames and passwords)...

*   A safer and easier alternative is **SFTP**.
*   Another alternative is **FTPS**, which is an extension of FTP that uses **TLS** to encrypt data sent and received.

When authenticating users via passwords, the transmitted data should be encrypted; otherwise, an eavesdropper could discover the password. We strongly recommend that you not allow password logins to your VM, as automated brute-force attempts to crack passwords can be expected on any machine connected to the internet. Instead, use SSH key authentication (see [SSH Keys](link-to-ssh-keys-doc)). SFTP can be configured to use SSH key authentication.

# Setting up FTP

If you do not have the freedom to choose an alternative to FTP, see the guide which best matches your operating system:

*   [Ubuntu guide](link-to-ubuntu-guide)
*   [CentOS 6 guide](link-to-centos-guide)

The ports that FTP uses must be open on your VM; see [this page](link-to-ports-page) for information about opening ports. FTP uses port 21 to initiate file transfer requests, but the actual transfer can take place on a randomly chosen port above port 1025, though the details of this can vary depending on which mode FTP operates. For example, port 20 can also be involved. This means that to allow FTP access on your VM, you must open port 21, possibly port 20, and probably ports 1025 and above. Every open port represents a security risk, which is why other protocols are preferred to FTP. See [this article](link-to-ftp-ports-article) for more details on ports used by FTP.


**(Note:  Please replace bracketed placeholders like `[link-to-webserver-doc]` with actual links to the relevant documentation.)**
