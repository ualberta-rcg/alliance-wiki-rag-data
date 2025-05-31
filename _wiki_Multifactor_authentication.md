# Multifactor Authentication

Other languages: English, français

Multifactor authentication (MFA) enhances account security beyond passwords.  Once enabled, you'll need your username, password, and a second action (the second factor) to access most services.

Second factor options:

* Approve a notification on a smart device via the Duo Mobile application.
* Enter an on-demand generated code.
* Push a button on a hardware key (YubiKey).

This feature's rollout is gradual; not all services will immediately support it.


## Contents

1. Recorded Webinars
2. Registering Factors
    * Registering Multiple Factors
    * Use a Smartphone or Tablet
    * Use a YubiKey
        * Configuring your YubiKey for Yubico OTP
3. Using Your Second Factor
    * When Connecting via SSH
        * Configuring your SSH client with ControlMaster
            * Linux and MacOS
            * Windows
    * When Authenticating to Our Account Portal
4. Configuring Common SSH Clients
    * FileZilla
        * Niagara Special Case
    * MobaXTerm
        * Prompt on File Transfer
        * Use SSH Key Instead of Password
        * Known Issues with MFA
    * PuTTY
    * WinSCP
    * PyCharm
    * Cyberduck
5. Frequently Asked Questions
    * Can I use Authy/Google Authenticator/Microsoft Authenticator?
    * I do not have a smartphone or tablet, and I do not want to buy a YubiKey
    * Why can't you send me one-time passcodes through SMS?
    * Why can't you send me one-time passcodes through email?
    * I have an older Android phone and I cannot download the Duo Mobile application from the Google Play site. Can I still use Duo?
    * I want to disable multifactor authentication. How do I do this?
    * I do not have a smartphone or tablet, or they are too old. Can I still use multifactor authentication?
    * I have lost my second factor device. What can I do?
    * Which SSH clients can be used when multifactor authentication is configured?
    * I need to have automated SSH connections to the clusters through my account. Can I use multifactor authentication?
    * Why have I received the message "Access denied. Duo Security does not provide services in your current location"?
6. Advanced Usage
    * Configuring your YubiKey for Yubico OTP using the Command Line (`ykman`)


## Recorded Webinars

Two webinars (October 2023) are available:

* Authentification multifacteur pour la communauté de recherche (French)
* Multifactor authentication for researchers (English)


## Registering Factors

### Registering Multiple Factors

When enabling MFA, configure at least two second factors (e.g., phone and single-use codes; phone and hardware key; two hardware keys). This ensures continued access if one factor is lost.


### Use a Smartphone or Tablet

1. Install the Duo Mobile authentication application from the Apple Store or Google Play.  Use the correct application (see icon below).  TOTP apps (Aegis, Google Authenticator, Microsoft Authenticator) are *not* compatible.
2. Go to the CCDB, log in, and select `My account → Multifactor authentication management`.
3. Under `Register a device`, click `Duo Mobile`.
4. Name your device, click `Continue`. A QR code will appear.
5. In Duo Mobile, tap `Set up account` or the "+" sign.
6. Tap `Use a QR code`.
7. Scan the CCDB QR code.  Ensure your device is online while scanning.

**(Include images for steps 1-7 here)**


### Use a YubiKey

A YubiKey (Yubico) is an alternative if you lack a smartphone/tablet or frequently cannot use one.  Note that not all YubiKey models are compatible (they must support "Yubico OTP"). We recommend the YubiKey 5 Series, but older models might work (see [Yubico identification page](link_to_yubico_page)).

A YubiKey 5 is USB-sized and costs $67-$100. Models support USB-A, USB-C, Lightning, and some offer NFC.  Our clusters use Yubico One-Time Passwords (OTP).  Touching the YubiKey generates an authentication string.

To register, you'll need the Public ID, Private ID, and Secret Key. If you have this information, go to the [Multifactor authentication management page](link_to_mfa_management_page). Otherwise, follow these steps:

#### Configuring your YubiKey for Yubico OTP

1. Download and install the YubiKey Manager from the [Yubico website](link_to_yubico_website).
2. Insert your YubiKey and launch the manager.
3. Select `Applications`, then `OTP`.
4. Select `Configure` for slot 1 (short touch, 1-2.5 seconds) or slot 2 (long touch, 3-5 seconds). Slot 1 is typically pre-registered for Yubico cloud mode. If used for other services, use slot 2 or click `Swap` to transfer the configuration.
5. Select `Yubico OTP`.
6. Select `Use serial`, then generate a private ID and secret key.
7. Securely save the Public ID, Private ID, and Secret Key before clicking `Finish`.
8. Log into the CCDB to register your YubiKey on the [Multifactor authentication management page](link_to_mfa_management_page).

**(Include images for steps 3-8 here)**

Test your YubiKey by pressing its button; it should generate a code.


## Using Your Second Factor

### When Connecting via SSH

After providing your password or SSH key, you'll be prompted for your second factor:

```
[name@server ~]$ ssh cluster.computecanada.ca
Duo two-factor login for name

Enter a passcode or select one of the following options:
1. Duo Push to My phone (iOS)
Passcode or option (1-1):
```

Select your device (if multiple are enrolled). You'll receive a notification to accept.  For YubiKeys, touch it at the "Passcode" prompt. For backup or time-based codes (Duo Mobile), paste or type them.

```
[name@server ~]$ ssh cluster.computecanada.ca
Duo two-factor login for name

Enter a passcode or select one of the following options:
1. Duo Push to My phone (iOS)
Passcode or option (1-1): vvcccbhbllnuuebegkkbcfdftndjijlneejilrgiguki
Success.
Logging you in ...
```

#### Configuring your SSH client with ControlMaster

This reduces second factor prompts. Edit your `.ssh/config` to add:

```
Host HOSTNAME
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlMaster auto
    ControlPersist 10m
```

Replace `HOSTNAME` with the server's hostname.  The first SSH session will request both factors; subsequent connections (within 10 minutes) reuse the first session's connection.

*Note:* ControlMaster (Multiplexing) doesn't work natively on Windows; Windows Subsystem for Linux (WSL) is required.  See [Configuring WSL as a ControlMaster relay server](link_to_wsl_config).


### When Authenticating to Our Account Portal

After entering your username and password, you'll see a prompt to select your second factor.  (Note: This screen will be updated.)


## Configuring Common SSH Clients

Command-line clients usually support MFA without extra configuration; graphical clients often require it.

### FileZilla

FileZilla prompts for password and second factor per transfer (independent connections close after idle time). To avoid this, limit connections per site to 1 in "Site Manager" => "Transfer Settings".  (You'll lose server browsing during transfers).

1. Launch FileZilla and select "Site Manager".
2. Create (or edit) a site.
3. On the "General" tab:
    * Protocol: "SFTP – SSH File Transfer Protocol"
    * Host: [cluster hostname]
    * Logon Type: "Interactive"
    * User: [your username]
4. On the "Transfer Settings" tab:
    * Limit number of simultaneous connections: [checked]
    * Maximum number of connections: 1
5. Select "OK".
6. Test the connection.

#### Niagara Special Case

FileZilla only supports either SSH keys or interactive prompts, not both.  Since Niagara needs both, use a different SCP client or try this workaround:

1. Connect with an SSH key (it'll fail due to the second factor prompt, but FileZilla will remember your key).
2. Change the login method to interactive and connect again (you'll get the 2FA prompt).


### MobaXTerm (Version 23.1 or later)

Version 23.5 (on Archive.org) is the latest version for which the following instructions work for most users.

#### Prompt on File Transfer

MobaXterm uses two connections by default (terminal and remote file browser). The SFTP file browser causes a second factor prompt. Switch the SSH-browser type to "SCP (enhanced speed)" or "SCP (normal speed)" in the session's Advanced SSH settings.

#### Use SSH Key Instead of Password

To allow downloads and use an SSH passphrase instead of the password, change SSH settings ("SSH" tab in Settings):

* Uncheck "GSSAPI Kerberos"
* Uncheck "Use external Pageant"
* Check "Use internal SSH agent "MobAgent""
* Use the "+" button to select the SSH key file.

#### Known Issues with MFA

MobaXTerm might show strange behavior after MFA adoption. Files might open via the terminal, but opening, downloading, or uploading using the left-side navigation bar hangs.  This is because three independent sessions need authentication (terminal, folder display, file transfer). Hidden MFA-Duo windows might be waiting for authentication. Each folder navigation starts another MFA transaction.  Some MobaXterm versions handle this better.


### PuTTY (Version 0.72 or later)

### WinSCP

Use SSH Keys.


### PyCharm

Setup SSH Keys before connecting.  When connecting, enter your username and host. You'll be prompted for a "One-time password"; use your YubiKey or Duo password.


### Cyberduck

Cyberduck opens a new connection per file transfer, prompting for the second factor. In preferences, under "Transfers" -> "General", select "Use browser connection" for "Transfer Files" and uncheck "Segmented downloads with multiple connections per file".

**(Include image here)**


## Frequently Asked Questions

### Can I use Authy/Google Authenticator/Microsoft Authenticator?

No, only Duo Mobile works.


### I do not have a smartphone or tablet, and I do not want to buy a YubiKey

You won't be able to use our services when MFA becomes mandatory. A YubiKey is the cheapest MFA option and should be covered by research funding.  MFA is a funder requirement.


### Why can't you send me one-time passcodes through SMS?

SMS costs money and is considered insecure by security experts.


### Why can't you send me one-time passcodes through email?

Duo doesn't support email-based one-time codes.


### I have an older Android phone and I cannot download the Duo Mobile application from the Google Play site. Can I still use Duo?

Yes, download it from the Duo website:

* Android 8 and 9: [DuoMobile-4.33.0.apk](link_to_apk)
* Android 10: [DuoMobile-4.56.0.apk](link_to_apk)

SHA-256 checksums: [link_to_checksums]
Installation instructions: [link_to_instructions]


### I want to disable multifactor authentication. How do I do this?

MFA is mandatory and cannot be disabled by users. Exceptions are only for automation.  If MFA is inconvenient, use the SSH client configurations above or refer to the recorded webinars for tips.


### I do not have a smartphone or tablet, or they are too old. Can I still use multifactor authentication?

Yes, use a YubiKey.


### I have lost my second factor device. What can I do?

If you have bypass codes or multiple devices, use another method to access your account on the [account portal](link_to_account_portal) and delete the lost device. Then register a new one.

If you've lost all devices and bypass codes, email support@tech.alliancecan.ca with this information:

* Primary account email address
* Account active duration
* Research area
* IP address ([link_to_ip_address])
* Principal investigator
* Group members
* Contact for validation
* Most used clusters
* Most loaded software modules
* Last job run date
* Recent batch job IDs
* Recent technical support ticket topics and IDs


### Which SSH clients can be used when multifactor authentication is configured?

Most command-line clients (Linux, macOS), Cyberduck, FileZilla, JuiceSSH (Android), MobaXTerm, PuTTY, PyCharm, Termius (iOS), VSCode, WinSCP.


### I need to have automated SSH connections to the clusters through my account. Can I use multifactor authentication?

Login nodes for automated, unattended SSH connections are being deployed.  See [here](link_to_automated_ssh) for more information.


### Why have I received the message "Access denied. Duo Security does not provide services in your current location"?

Duo blocks authentications from IP addresses in countries/regions under economic/trade sanctions. See [Duo help](link_to_duo_help).


## Advanced Usage

### Configuring your YubiKey for Yubico OTP using the Command Line (`ykman`)

1. Install `ykman` following instructions for your OS from Yubico's [ykman guide](link_to_ykman_guide).
2. Insert your YubiKey and run `ykman info`.
3. Run `ykman otp info`.
4. Select a slot and run `ykman otp yubiotp` to program it.
5. Securely save the Public ID, Private ID, and Secret Key.
6. Register your YubiKey in the CCDB's [Multifactor authentication management page](link_to_mfa_management_page).

```bash
[name@yourLaptop]$ ykman otp yubiotp -uGgP vvcccctffclk 2
Using a randomly generated private ID: bc3dd98eaa12
Using a randomly generated secret key: ae012f11bc5a00d3cac00f1d57aa0b12
Upload credential to YubiCloud? [y/N]: y
Upload to YubiCloud initiated successfully.
Program an OTP credential in slot 2? [y/N]: y
Opening upload form in browser: https://upload.yubico.com/proceed/4567ad02-c3a2-1234-a1c3-abe3f4d21c69
```

Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Multifactor_authentication&oldid=177414](https://docs.alliancecan.ca/mediawiki/index.php?title=Multifactor_authentication&oldid=177414)"

**(Remember to replace all bracketed placeholders like `[link_to_yubico_page]` with actual links.)**
