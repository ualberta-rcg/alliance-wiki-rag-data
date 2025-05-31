# Secure Shell (SSH)

Secure Shell (SSH) is a widely used standard for securely connecting to remote machines.  The SSH connection is encrypted, including the username and password. SSH is the standard way to connect to execute commands, submit jobs, check job progress, and in some cases, transfer files.

Various SSH implementations exist for most major operating systems.

* **macOS and Linux:** The most widely used client is OpenSSH, a command-line application installed by default.
* **Windows:** SSH is available in the PowerShell terminal, the `cmd` prompt, or through Windows Subsystem for Linux (WSL). Popular third-party clients include PuTTY, MobaXTerm, WinSCP, and Bitvise.

To use SSH successfully, you must:

1. Know the machine name (e.g., `cedar.alliancecan.ca` or `niagara.alliancecan.ca`).
2. Know your username (e.g., `ansmith`).  This is *not* your CCI (like `abc-123`), CCRI (like `abc-123-01`), or email address.
3. Know your password or have an SSH key. Your password is the same as your CCDB login password.  Using an SSH key is highly recommended for improved security.
4. Be registered for multifactor authentication and have your second factor available.


## Connecting via Command Line

From a command-line client (e.g., `/Applications/Utilities/Terminal.app` for macOS, `cmd` or PowerShell for Windows), use the `ssh` command like this:

```bash
ssh username@machine_name
```

For graphical clients like MobaXterm or PuTTY, see:

* [Connecting with MobaXTerm](<link_to_mobaxterm_docs>)
* [Connecting with PuTTY](<link_to_putty_docs>)


The first connection to a machine prompts you to store its host key, a unique identifier verifying the machine on subsequent connections.


## SSH Keys

For more information on generating key pairs, see:

* [SSH Keys](<link_to_ssh_keys_docs>)
* [Generating SSH keys in Windows](<link_to_windows_ssh_keygen_docs>)
* [Using SSH keys in Linux](<link_to_linux_ssh_key_docs>)


## SSH Tunneling

For using SSH to allow communication between compute nodes and the internet, see:

[SSH tunnelling](<link_to_ssh_tunneling_docs>)


## SSH Configuration File

For simplifying the login procedure using an SSH configuration file, see:

[SSH configuration file](<link_to_ssh_config_docs>)


## X11 for Graphical Applications

SSH supports graphical applications via the X protocol (X11). You need an X11 server installed:

* **Linux:** Usually pre-installed.
* **macOS:** Install a package like XQuartz.
* **Windows:** Included with MobaXterm; use VcXsrv with PuTTY.

To enable X11 communications using the SSH command line, add the `-Y` option:

```bash
ssh -Y username@machine_name
```

## Connection Errors

Connection errors might include:

* `no matching cipher found`
* `no matching MAC found`
* `unable to negotiate a key exchange method`
* `couldn't agree a key exchange algorithm`
* `remote host identification has changed`

The last error might indicate a man-in-the-middle attack or a cluster security upgrade.  Verify the host key fingerprint against the published list at [SSH host keys](<link_to_host_keys_docs>). If it matches, proceed; otherwise, terminate the connection and contact support.

A security upgrade on the Niagara cluster occurred on May 31, 2019. See [this page](<link_to_niagara_upgrade_docs>) for required actions. Further upgrades across all clusters happened in September/October 2019; see [SSH security improvements](<link_to_security_improvements_docs>) for details.

Other errors require upgrading your OS and/or SSH client to support strong ciphers, key exchange protocols, and MAC algorithms.  Known failing versions include:

* OpenSSH on CentOS/RHEL 5
* PuTTY v0.64 and earlier on Windows


**(Note:  Replace bracketed `<link_to...>` placeholders with actual links to relevant documentation pages.)**
