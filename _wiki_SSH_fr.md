# SSH

The SSH (Secure Shell) protocol is frequently used to obtain a secure connection to a remote machine. An SSH connection is entirely encrypted, including the credentials entered to connect (username and password). The SSH protocol is used to connect to our clusters to run commands, check task progress, or, in some cases, transfer files.

Software implementations of the SSH protocol exist for most major operating systems.

On macOS and Linux, the most commonly used is OpenSSH, a command-line application installed by default.

With recent versions of Windows, SSH is available via the PowerShell terminal, in the `cmd` prompt, or via WSL (Windows Subsystem for Linux). Other SSH clients are also offered by third parties such as PuTTY, MobaXTerm, WinSCP, and Bitvise.


To use these SSH implementations correctly, you must:

* Know the name of the machine you want to connect to; the format looks like `cedar.alliancecan.ca` or `niagara.alliancecan.ca`.
* Know your username; the format looks like `ansmith`. Your username is *not* your email address, nor your CCI (e.g., `abc-123`), nor a CCRI (e.g., `abc-123-01`).
* Know your password or use an SSH key. Your password is the same as the one you use to log in to the CCDB portal. We recommend creating and using an SSH key, which is more secure than a password.
* Register for multi-factor authentication and remember your second factor.

In a command-line client (e.g., `/Applications/Utilities/Terminal.app` on macOS; `cmd` or PowerShell on Windows), use the `ssh` command like so:

```bash
[name@server ~]$ ssh username@machine_name
```

For more information on graphical clients like MobaXterm or PuTTY, see:

* [Connecting to a server with MobaXTerm](link-to-mobaxterm-doc)
* [Connecting to a server with PuTTY](link-to-putty-doc)

On your first connection to a machine, you will be asked to save a copy of its host key; this key is a unique identifier with which the SSH client verifies that it is the same machine when you connect subsequently.

For more information on how to generate key pairs, see [SSH Keys](link-to-ssh-keys-doc).


## X11 for graphical applications

SSH supports graphical applications via the X protocol, known as X11. To use X11, an X11 server must be installed on your computer. On Linux, an X11 server will usually already be installed, but on macOS you will generally need to install an external package such as XQuartz. On Windows, MobaXterm comes with an X11 server; with PuTTY, the server is VcXsrv.

In the SSH command line, add the `-Y` option to allow X11 communications.

```bash
[name@server ~]$ ssh -Y username@machine_name
```

## Connection errors

You may receive an error message when connecting to a cluster:

* `no matching cipher found`
* `no matching MAC found`
* `unable to negotiate a key exchange method`
* `couldn't agree a key exchange algorithm`
* `remote host identification has changed`

This last message may indicate a man-in-the-middle attack or a security update for the cluster you are trying to connect to. If this message is displayed, check if the fingerprint of the host key mentioned corresponds to one of the valid host keys; if so, you can proceed with the connection.

If the host key is not in the list, close the connection and contact technical support.

Niagara users had [actions to take](link-to-niagara-actions) following the May 31, 2019 security update. Similar updates were performed on the other clusters towards the end of September 2019; for more information, see the [wiki page on security improvements](link-to-security-improvements-wiki).

In the case of other error messages, you will need to update your operating system and/or your SSH client to allow for more robust encryption, key exchange protocols, and MAC (message authentication code) algorithms.

These errors are known for the following versions that will need to be updated:

* OpenSSH on CentOS/RHEL 5
* PuTTY v0.64 and below, on Windows


**(Retrieved from "https://docs.alliancecan.ca/mediawiki/index.php?title=SSH/fr&oldid=171831")**
