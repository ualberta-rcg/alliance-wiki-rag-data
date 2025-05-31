# SSH Security Improvements

**Other languages:** English, français

[SSH security improvements flowchart](placeholder_for_flowchart.png) *(click to enlarge)*


SSH is the software protocol used to connect to Compute Canada clusters. It secures communication by verifying the server's and your identities against known data and encrypting the connection.  Because security risks evolve, Compute Canada will soon end support for certain SSH options no longer considered secure.  To continue using our clusters, you'll need to make some changes, outlined in the flowchart and detailed below.


## SSH Changes (September-October 2019)

An email explaining these changes (with "IMPORTANT" in the subject line) was sent to all users on July 29, and with more detail on September 16.


### What is Changing?

The following SSH security improvements were implemented on September 24, 2019, on Graham, and one week later on Béluga and Cedar:

*   Disable certain encryption algorithms.
*   Disable certain public key types.
*   Regenerate the cluster's host keys.

If you don't understand "encryption algorithms," "public keys," or "host keys," don't worry.  Simply follow the steps below. If testing indicates you need to update or change your SSH client, you may find [this page](placeholder_for_link.com) useful.

These changes do not affect Arbutus users, who connect via a web interface, not SSH.

Earlier, less comprehensive updates were made to Niagara (May 31, 2019; see [here](placeholder_for_link.com)) and Graham (early August), which triggered similar messages and errors.


### Updating Your Client's Known Host List

The first login to a Compute Canada cluster after the changes will likely display a warning like this:

```
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
IT IS POSSIBLE THAT SOMEONE IS DOING SOMETHING NASTY!
Someone could be eavesdropping on you right now (man-in-the-middle attack)!
It is also possible that a host key has just been changed.
The fingerprint for the ED25519 key sent by the remote host is
SHA256:mf1jJ3ndpXhpo0k38xVxjH8Kjtq3o1+ZtTVbeM0xeCk.
Please contact your system administrator.
Add correct host key in /home/username/.ssh/known_hosts to get rid of this message.
Offending ECDSA key in /home/username/.ssh/known_hosts:109
ED25519 host key for graham.computecanada.ca has changed and you have requested strict checking.
Host key verification failed.
Killed by signal 1.
```

This warning appears because the cluster's host keys (e.g., Graham) changed, and your SSH client remembers the old ones (to prevent man-in-the-middle attacks). This will happen for each device you connect from.  You might also see a "DNS spoofing" warning, related to the same change.


#### MobaXterm, PuTTY, or WinSCP

If using MobaXterm, PuTTY, or WinSCP on Windows, the warning appears in a pop-up, letting you accept the new host key by clicking "Yes."

**Only click "Yes" if the fingerprint matches one listed in [SSH host key fingerprints](#ssh-host-key-fingerprints) below.  If it doesn't match, do not accept the connection and contact Technical Support with details.**


#### macOS, Linux, GitBash, or Cygwin

For the command-line `ssh` command on macOS, Linux, GitBash, or Cygwin, "forget" the old host key using these commands:

**Graham:**

```bash
for h in 2620:123:7002:4::{2..5} 199.241.166.{2..5} {gra-login{1..3},graham,gra-dtn,gra-dtn1,gra-platform,gra-platform1}.{sharcnet,computecanada}.ca; do ssh-keygen -R $h; done
```

**Cedar:**

```bash
for h in 206.12.124.{2,6} cedar{1,5}.cedar.computecanada.ca cedar.computecanada.ca; do ssh-keygen -R $h; done
```

**Beluga:**

```bash
for h in beluga{,{1..4}}.{computecanada,calculquebec}.ca 132.219.136.{1..4}; do ssh-keygen -R $h; done
```

**Mp2:**

```bash
for h in ip{15..20}-mp2.{computecanada,calculquebec}.ca 204.19.23.2{15..20}; do ssh-keygen -R $h; done
```

The next `ssh` connection to the cluster will prompt you to confirm the new host keys, e.g.:

```
$ ssh graham.computecanada.ca
The authenticity of host 'graham.computecanada.ca (142.150.188.70)' can't be established.
ED25519 key fingerprint is SHA256:mf1jJ3ndpXhpo0k38xVxjH8Kjtq3o1+ZtTVbeM0xeCk.
ED25519 key fingerprint is MD5:bc:93:0c:64:f7:e7:cf:d9:db:81:40:be:4d:cd:12:5c.
Are you sure you want to continue connecting (yes/no)?
```

**Only type "yes" if the fingerprint matches one in [SSH host key fingerprints](#ssh-host-key-fingerprints) below. If not, do not connect and contact Technical Support.**


### Troubleshooting


#### Can I Test My SSH Client Before the Change?

Yes.  A test server, `ssh-test.computecanada.ca`, lets you test your SSH client with the new protocols using your regular Compute Canada credentials.  Successful connection means your client will connect to our clusters after the SSH changes; however, you'll still need to update your client's locally stored SSH host keys. See the [SSH host key fingerprints](#ssh-host-key-fingerprints) list.


#### My SSH Key No Longer Works

If prompted for a password despite previously using SSH keys on the same system, 1024-bit DSA & RSA keys were likely disabled. You need to generate a new, stronger key.  The process depends on your OS ([Windows](placeholder_for_link.com) or [Linux/macOS](placeholder_for_link.com)). These instructions also show how to add your client's new public key to the remote host for key authentication instead of passwords.


#### I Can't Connect!

Errors like these:

```
Unable to negotiate with 142.150.188.70 port 22: no matching cipher found.
Unable to negotiate with 142.150.188.70 port 22: no matching key exchange method found.
Unable to negotiate with 142.150.188.70 port 22: no matching mac found.
```

require upgrading your SSH client to a compatible one (see below).


#### Which Clients Are Compatible With the New Configuration?

This list isn't exhaustive, but we've tested these clients.  Earlier versions may or may not work. Update your OS and SSH client to the latest version compatible with your hardware.


##### Linux Clients

*   OpenSSH\_7.4p1, OpenSSL 1.0.2k-fips (CentOS 7.5, 7.6)
*   OpenSSH\_6.6.1p1 Ubuntu-2ubuntu2.13, OpenSSL 1.0.1f (Ubuntu 14)


##### OS X Clients

Use `ssh -V` to determine your OS X SSH client version.

*   OpenSSH 7.4p1, OpenSSL 1.0.2k (Homebrew)
*   OpenSSH 7.9p1, LibreSSL 2.7.3 (OS X 10.14.5)


##### Windows Clients

*   MobaXterm Home Edition v11.1
*   PuTTY 0.72
*   Windows Services for Linux (WSL) v1
    *   Ubuntu 18.04 (OpenSSH\_7.6p1 Ubuntu-4ubuntu0.3, OpenSSL 1.0.2n)
    *   openSUSE Leap 15.1 (OpenSSH\_7.9p1, OpenSSL 1.1.0i-fips)


##### iOS Clients

*   Termius, 4.3.12


## SSH Host Key Fingerprints

To retrieve host fingerprints remotely:

```bash
ssh-keyscan <hostname> | ssh-keygen -E md5 -l -f -
ssh-keyscan <hostname> | ssh-keygen -E sha256 -l -f -
```

The following are SSH fingerprints for our clusters. If your fingerprint doesn't match, do not connect and contact Technical Support.


### Béluga

*   **ED25519**
    *   SHA256: lwmU2AS/oQ0Z2M1a31yRAxlKPcMlQuBPFP+ji/HorHQ
    *   MD5: 2d:d7:cc:d0:85:f9:33:c1:44:80:38:e7:68:ce:38:ce
*   **RSA**
    *   SHA256: 7ccDqnMTR1W181U/bSR/Xg7dR4MSiilgzDlgvXStv0o
    *   MD5: 7f:11:29:bf:61:45:ae:7a:07:fc:01:1f:eb:8c:cc:a4


### Cedar

*   **ED25519**
    *   SHA256: a4n68wLDqJhxtePn04T698+7anVavd0gdpiECLBylAU
    *   MD5: f8:6a:45:2e:b0:3a:4b:16:0e:64:da:fd:68:74:6a:24
*   **RSA**
    *   SHA256: 91eMtc/c2vBrAKM0ID7boyFySo3vg2NEcQcC69VvCg8
    *   MD5: 01:27:45:a0:fd:34:27:9e:77:66:b0:97:55:10:0e:9b


### Graham

*   **ED25519**
    *   SHA256: mf1jJ3ndpXhpo0k38xVxjH8Kjtq3o1+ZtTVbeM0xeCk
    *   MD5: bc:93:0c:64:f7:e7:cf:d9:db:81:40:be:4d:cd:12:5c
*   **RSA**
    *   SHA256: tB0gbgW4PV+xjNysyll6JtDi4aACmSaX4QBm6CGd3RM
    *   MD5: 21:51:ca:99:15:a8:f4:92:3b:8e:37:e5:2f:12:55:d3


### Narval

*   **ED25519**
    *   SHA256: pTKCWpDC142truNtohGm10+lB8gVyrp3Daz4iR5tT1M
    *   MD5: 79:d5:b2:8b:c6:2c:b6:3b:79:d2:75:0e:3b:31:46:17
*   **RSA**
    *   SHA256: tC0oPkkY2TeLxqYHgfIVNq376+RfBFFUZaswnUeeOnw
    *   MD5: bc:63:b5:f9:e6:48:a3:b7:0d:4a:23:26:a6:31:19:ef


### Niagara

*   **ED25519**
    *   SHA256: SauX2nL+Yso9KBo2Ca6GH/V9cSFLFXwxOECGWXZ5pxc
    *   MD5: b4:ae:76:a5:2b:37:8d:57:06:0e:9a:de:62:00:26:be
*   **RSA**
    *   SHA256: k6YEhYsI73M+NJIpZ8yF+wqWeuXS9avNs2s5QS/0VhU
    *   MD5: 98:e7:7a:07:89:ef:3f:d8:68:3d:47:9c:6e:a6:71:5e


### ssh-test.computecanada.ca

*   **ED25519 (256b)**
    *   SHA256: Tpu6li6aynYkhmB83Q9Sh7x8qdhT8Mbw4QcDxTaHgxY
    *   MD5: 33:8f:f8:57:fa:46:f9:7f:aa:73:e2:0b:b1:ce:66:38
*   **RSA (4096b)**
    *   SHA256: DMSia4nUKIyUhO5axZ/As4I8uqlaX0jPcJvcK93D2H0
    *   MD5: a7:08:00:7c:eb:81:f2:f7:2f:5a:92:b0:85:e3:e8:5d


### Mp2

*   **ED25519 (256b)**
    *   SHA256: hVAo6KoqKOEbtOaBh6H6GYHAvsStPsDEcg4LXBQUP50
    *   MD5: 44:71:28:23:9b:a1:9a:93:aa:4b:9f:af:8d:9b:07:01
*   **RSA (4096b)**
    *   SHA256: XhbK4jWsnoNNjoBudO6zthlgTqyKkFDtxiuNY9md/aQ
    *   MD5: 88:ef:b0:37:26:75:a2:93:91:f6:15:1c:b6:a7:a9:37


### Siku

*   **ED25519 (256b)**
    *   SHA256: F9GcueU8cbB0PXnCG1hc4URmYYy/8JbnZTGo4xKflWU
    *   MD5: 44:2b:1d:40:31:60:1a:83:ae:1d:1a:20:eb:12:79:93
*   **RSA (2048b)**
    *   SHA256: cpx0+k52NUJOf8ucEGP3QnycnVkUxYeqJQMp9KOIFrQ
    *   MD5: eb:44:dc:42:70:32:f7:61:c5:db:3a:5c:39:04:0e:91


**(Retrieved from "https://docs.alliancecan.ca/mediawiki/index.php?title=SSH_security_improvements&oldid=83271")**
