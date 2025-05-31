# Automation in the Context of Multifactor Authentication

Other languages: English, français

An automated workflow involving an external machine connecting to a cluster without human intervention cannot use a second authentication factor.  To execute such a workflow now that MFA is required, you must request access to an **automation node**. An automation node doesn't require a second factor but is more limited than a regular login node regarding accepted authentication types and actions.


## Increased Security Measures

### Available Only by Request

If you need an automated workflow for your research, contact our [technical support](link-to-technical-support) and request access to an automation node.  When contacting us, detail the intended automation, commands to be executed, and tools/libraries used.

### Available Only Through Constrained SSH Keys

Automation nodes only accept SSH keys uploaded to the CCDB. Keys in your `.ssh/authorized_keys` file are not accepted.  Additionally, SSH keys *must* have these constraints:

#### `restrict`

This disables port forwarding, agent forwarding, X11 forwarding, and the pseudo-teletype (PTY), blocking most interactive workloads.  This is because automation nodes aren't for long-running or interactive processes; use regular login nodes instead.

#### `from="pattern-list"`

This specifies that the key can only be used from IP addresses matching the patterns. This ensures the key isn't used from unintended computers. The pattern list must include only IP addresses fully specifying at least the network class, network, and subnet (the first three elements). For example, `x.y.*.*` is unacceptable, but `x.y.z.*` is acceptable.  The IP address must be a *public* IP address;  `10.0.0.0 – 10.255.255.255`, `172.16.0.0 – 172.31.255.255`, and `192.168.0.0 – 192.168.255.255` are incorrect. Use a site like [What Is My IP Address?](link-to-ip-address-site) or the shell command `curl ifconfig.me` to find your public IP address.

#### `command="COMMAND"`

This forces the execution of `COMMAND` when a connection is established, restricting usable commands with this key.

### Convenience Wrapper Scripts for `command=`

`command` constraints can specify any command but are most useful with a wrapper script accepting or rejecting commands based on the called command. You can write your own, but we provide several for common actions in [this git repository](link-to-git-repository):

*   `/cvmfs/soft.computecanada.ca/custom/bin/computecanada/allowed_commands/transfer_commands.sh`: Allows only file transfers (e.g., `scp`, `sftp`, `rsync`).
*   `/cvmfs/soft.computecanada.ca/custom/bin/computecanada/allowed_commands/archiving_commands.sh`: Allows commands to archive files (e.g., `gzip`, `tar`, `dar`).
*   `/cvmfs/soft.computecanada.ca/custom/bin/computecanada/allowed_commands/file_commands.sh`: Allows commands to manipulate files (e.g., `mv`, `cp`, `rm`).
*   `/cvmfs/soft.computecanada.ca/custom/bin/computecanada/allowed_commands/git_commands.sh`: Allows the `git` command.
*   `/cvmfs/soft.computecanada.ca/custom/bin/computecanada/allowed_commands/slurm_commands.sh`: Allows some Slurm commands (e.g., `squeue`, `sbatch`).
*   `/cvmfs/soft.computecanada.ca/custom/bin/computecanada/allowed_commands/allowed_commands.sh`: Allows all of the above.


### Examples of Accepted SSH Keys

Accepted SSH keys must include all three constraints.  Examples of accepted keys:

This key only allows file transfers (using `scp`, `sftp`, or `rsync`):

```
restrict,from="216.18.209.*",command="/cvmfs/soft.computecanada.ca/custom/bin/computecanada/allowed_commands/transfer_commands.sh" ssh-ed25519 AAAAC3NzaC1lZDI1NTE6AACAIExK9iTTDGsyqKKzduA46DvIJ9oFKZ/WN5memqG9Invw
```

This key only allows Slurm commands (`squeue`, `scancel`, `sbatch`, `scontrol`, `sq`):

```
restrict,from="216.18.209.*",command="/cvmfs/soft.computecanada.ca/custom/bin/computecanada/allowed_commands/slurm_commands.sh" ssh-ed25519 AAAAC3NzaC1lZDI1NTE6AACAIExK9iTTDGsyqKKzduA46DvIJ9oFKZ/WN5memqG9Invw
```

**Warning:** Add constraints directly as text before your key before uploading the complete string to your account.


## Automation Nodes for Each Cluster

Here are hostnames for unattended connections on each cluster:

*   **Cedar:** `robot.cedar.alliancecan.ca`
*   **Graham:** `robot.graham.alliancecan.ca`
*   **Béluga:** `robot.beluga.alliancecan.ca`
*   **Narval:** `robot.narval.alliancecan.ca`
*   **Niagara:** `robot.niagara.alliancecan.ca` (or `robot1.niagara.alliancecan.ca` or `robot2.niagara.alliancecan.ca`; one is the fallback for the other)


## Using the Right Key

If you have multiple keys, use the correct one by passing parameters to your command. Examples:

With `ssh` or `scp`:

```bash
ssh -i .ssh/private_key_to_use ...
scp -i .ssh/private_key_to_use ...
```

With `rsync`:

```bash
rsync -e "ssh -i .ssh/private_key_to_use" ...
```

It's more convenient to put these parameters in your `~/.ssh/config` file:

```
host robot
    hostname robot.cluster.alliancecan.ca
    user myrobot
    identityfile ~/.ssh/my-robot-key
    identitiesonly yes
    requesttty no
```

Then, these commands will work:

```bash
ssh robot /usr/bin/ls
rsync -a datadir/a robot:scratch/testdata
```


## IPv4 vs IPv6 Issue

Your SSH client might use IPv6 over IPv4, more likely in Windows.  Ensure your `restrict,from=` key field IP address mask matches your computer's connection type. Check your addresses using [https://test-ipv6.com/](https://test-ipv6.com/).

An IPv4 address looks like `199.241.166.5`. An IPv6 address looks like `2620:123:7002:4::5`.  If you use an IPv4 mask (`199.241.166.*`) in the CCDB SSH key, but your SSH client connects using IPv6, the source address won't match the key's mask, and the key will be rejected.


### How to Identify the Problem

If you have SSH connection issues, try this:

```bash
ssh -i ~/.ssh/automation_key -vvv username@robot.graham.alliancecan.ca "ls -l"
```

This connects to Graham's automation node, executes `ls -l`, and prints the output. The `-vvv` option provides verbose debug output. Look for the `Connecting to...` message:

*   `debug1: Connecting to robot.graham.alliancecan.ca [199.241.166.5] port 22.` (IPv4)
*   `debug1: Connecting to robot.graham.alliancecan.ca [2620:123:7002:4::5] port 22.` (IPv6)


### Possible Solutions

*   Make the SSH client explicitly use IPv4 (`-4`) or IPv6 (`-6`) to match your CCDB key format.
*   Use an IP address instead of the hostname (e.g., `ssh -i ~/.ssh/automation_key -vvv username@199.241.166.5 "ls -l"` to force IPv4).
*   Disable IPv6 on your computer (last resort; Microsoft doesn't recommend this).  How to disable IPv6 depends on your operating system.


## Automation Using Python and Paramiko

If you use the Paramiko Python module, here's how to work with automation nodes:

```python
#!/usr/bin/env python3
import os
import paramiko

key = paramiko.Ed25519Key.from_private_key_file("/home/username/.ssh/cc_allowed")
user = "username"
host = "robot.graham.alliancecan.ca"
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(hostname=host, username=user, pkey=key)
cmd = "ls -l"
stdin, stdout, stderr = ssh.exec_command(cmd)
print("".join(stdout.readlines()))
ssh.close()
```

This connects to Graham's automation node using a CCDB-specified key, executes `ls -l`, and prints the output.  Install Paramiko with `pip install paramiko[all]` to ensure Ed25519 key support.


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Automation_in_the_context_of_multifactor_authentication&oldid=155797](https://docs.alliancecan.ca/mediawiki/index.php?title=Automation_in_the_context_of_multifactor_authentication&oldid=155797)"
