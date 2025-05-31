# SSH Configuration File

Under Linux and macOS, you can modify your local SSH configuration file to change the behavior of `ssh` and simplify the connection procedure. For example, to connect to `narval.alliancecan.ca` as `username` with an SSH key, you might need to use the command:

```bash
[name@yourLaptop ~] ssh -i ~/.ssh/your_private_key username@narval.alliancecan.ca
```

To avoid entering this command every time you connect to Narval, add the following to `~/.ssh/config` on your local computer:

```
Host narval
   User username
   HostName narval.alliancecan.ca
   IdentityFile ~/.ssh/your_private_key
```

You can now connect to Narval by entering:

```bash
[name@yourLaptop ~] ssh narval
```

This also changes the behavior of `sftp`, `scp`, and `rsync`, and you can now transfer files by entering, for example:

```bash
[name@yourLaptop ~] scp local_file narval:work/
```

If you frequently connect to different clusters, modify the `Host` block above instead of adding an entry for each cluster.

```
Host narval beluga graham cedar
   [...]
   HostName %h.alliancecan.ca
   [...]
```

Note that you must install your public SSH key on each cluster, or use the CCDB instead.

Other options of the `ssh` command have corresponding parameters that can be entered in your computer's `~/.ssh/config` file. In particular, `-X` (X11 forwarding), `-Y` (X11 forwarding without security checks), and `-A` (agent forwarding) can be defined in the corresponding sections of the configuration file by adding the lines:

```
ForwardX11 yes
ForwardX11Trusted yes
ForwardAgent yes
```

However, this is not recommended because:

*   Enabling X11 forwarding by default for all your connections can slow down your sessions, especially if your computer's X11 client is misconfigured.
*   Enabling X11 forwarding without security extensions presents a risk, and we recommend using it only when you have no other option. If the server you are connecting to is compromised, someone with root permissions could detect your computer's keyboard activity.
*   Although agent forwarding is convenient and more secure than entering a password on a remote computer, it does present a risk. In the event that the server you are connecting to is compromised, a user with root privileges could use your agent to connect to another host without your knowledge. We recommend using agent forwarding only when necessary.  Furthermore, if you use this feature, combine it with `ssh-askpass` so that each use of your agent triggers a prompt on your computer to warn you that your agent is being used.


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=SSH_configuration_file/fr&oldid=171783](https://docs.alliancecan.ca/mediawiki/index.php?title=SSH_configuration_file/fr&oldid=171783)"
