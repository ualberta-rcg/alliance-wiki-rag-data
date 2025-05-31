# SSH

Secure Shell (SSH) is a widely used standard to connect to remote machines securely. The SSH connection is encrypted, including the username and password. SSH is the standard way for you to connect in order to execute commands, submit jobs, check the progress of jobs, and in some cases, transfer files.

Various implementations of the SSH standard exist for most major operating systems.

On macOS and Linux, the most widely used client is OpenSSH, a command-line application installed by default.

For recent versions of Windows, SSH is available in the PowerShell terminal, in the `cmd` prompt, or through Windows Subsystem for Linux (WSL). There are also 3rd-party SSH clients that are popular, such as PuTTY, MobaXTerm, WinSCP, and Bitvise.

To use any of these implementations of SSH successfully, you must:

*   Know the name of the machine to which you want to connect. This will be something like `cedar.alliancecan.ca` or `niagara.alliancecan.ca`.
*   Know your username, typically something like `ansmith`. The `username` is *not* your CCI, like `abc-123`, nor a CCRI like `abc-123-01`, nor your email address.
*   Know your password, or have an SSH key. Your password is the same one you use to log in to CCDB. You may register and use an SSH key instead of a password; we highly recommend this since it provides better security.
*   Be registered for multifactor authentication and have your 2nd factor available.

From a command-line client (e.g., `/Applications/Utilities/Terminal.app` for macOS, `cmd` or PowerShell for Windows), use the `ssh` command like this:

```bash
ssh username@machine_name
```

For graphical clients such as MobaXterm or PuTTY, see:

*   [Connecting with MobaXTerm](link-to-mobaxterm-doc)
*   [Connecting with PuTTY](link-to-putty-doc)

The first time that you connect to a machine you'll be asked to store a copy of its host key, a unique identifier that allows the SSH client to verify, when connecting next time, that this is the same machine.

For more on generating key pairs, see:

*   [SSH Keys](link-to-ssh-keys-doc)
*   [Generating SSH keys in Windows](link-to-windows-ssh-keys-doc)
*   [Using SSH keys in Linux](link-to-linux-ssh-keys-doc)

For how to use SSH to allow communication between compute nodes and the internet, see:

*   [SSH tunnelling](link-to-ssh-tunnelling-doc)

For how to use an SSH configuration file to simplify the login procedure, see:

*   [SSH configuration file](link-to-ssh-config-doc)

## X11 for graphical applications

SSH supports graphical applications via the X protocol, now usually called "X11". In order to use X11 you must have an X11 server installed on your computer. Under Linux, an X11 server will normally already be installed, but users of macOS will typically need to install an external package such as XQuartz. Under Windows, MobaXterm comes with an X11 server, while for PuTTY users, there is VcXsrv.

Using the SSH command line, add the `-Y` option to enable X11 communications:

```bash
ssh -Y username@machine_name
```

## Connection errors

While connecting to one of our clusters, you might get an error message such as:

*   `no matching cipher found`
*   `no matching MAC found`
*   `unable to negotiate a key exchange method`
*   `couldn't agree a key exchange algorithm`
*   `remote host identification has changed`

The last of these error messages can point to a man-in-the-middle attack, or to an upgrade of security of the cluster you are trying to connect to.

If you get this, verify that the host key fingerprint mentioned in the message matches one of the host key fingerprints published at [SSH host keys](link-to-ssh-host-keys-doc).

If it does, it is safe to continue connecting. If the host key fingerprint does not appear on our published list, terminate the connection and contact support.

One such upgrade occurred on the Niagara cluster on May 31, 2019. See [this page](link-to-niagara-upgrade-doc) for the one-time action required from users after the security upgrade. Further upgrades of this type were made on all clusters in September/October 2019; see [SSH security improvements](link-to-ssh-security-improvements-doc) for more information.

If you see any of the other error messages, you will have to upgrade your OS and/or SSH client that supports strong ciphers, key exchange protocols and MAC (message authentication code) algorithms.

Here are known versions that will fail and will have to be upgraded:

*   OpenSSH on CentOS/RHEL 5
*   PuTTY v0.64 and earlier on Windows


**(Remember to replace bracketed placeholders like `[link-to-mobaxterm-doc]` with actual links to relevant documentation.)**
