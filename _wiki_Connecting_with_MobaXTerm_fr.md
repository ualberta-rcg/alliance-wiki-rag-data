# Connecting with MobaXTerm

This page is a translated version of the page Connecting with MobaXTerm and the translation is 100% complete.

Other languages:

* English
* français

## Creating the SSL Session *(Click to enlarge)*

## Remote Connection *(Click to enlarge)*

## X11 Redirection *(Click to enlarge)*

## Private Key Identification *(Click to enlarge)*

While MobaXterm and PuTTY connection tools (see [Connecting to a server with PuTTY](placeholder_link_to_putty_doc)) are similar, MobaXTerm offers more features.  It includes a built-in SFTP client and an X11 server, allowing you to run graphical programs remotely without needing another server. MobaXTerm can use already saved PuTTY sessions without requiring you to redefine the parameters.


To connect to a server you haven't previously connected to with MobaXTerm or PuTTY: under `Sessions->New session`, select `SSH` then enter the server address and your username (if necessary, check `Specify username`). Click `OK`. MobaXTerm saves this information for future connections to the server and establishes the SSH connection. After entering your password, the displayed window shows two panels:

* The left panel is the terminal where you enter commands.
* The right panel shows the list of files saved on the server; you can drag and drop your files from your computer to the server and vice versa.


## X11 Redirection

To enable X11 redirection and allow the use of graphical applications from the server:

1. Verify that X11 redirection is enabled for the specific session. To do this, click on the session name and select `Edit session`. In the `Session settings` window, select `Advanced SSH settings`; the `X11-Forwarding` box must be checked.
2. Verify that the `X server` icon is green (top right corner of the main window). If the icon is not green, the server is not activated. To start it, click on the red X icon.
3. Test X11 redirection. To do this, start the session by double-clicking on the session in the left panel and enter your password. Then run a simple command, for example `xclock`; the display of a pop-up window showing a clock indicates that X11 redirection is probably working.


## SSH Key Pair

In the left panel, right-click on the session name and select `Edit session`; this displays the `Session settings` window. Select `Advanced SSH settings` and check `Use private key`. To select the private key you want to use, click on the monitor icon in the right part of the window; this displays the list of keys to choose from. To create a key pair, see [Generating SSH keys in Windows](placeholder_link_to_keygen_doc).


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Connecting_with_MobaXTerm/fr&oldid=70302](https://docs.alliancecan.ca/mediawiki/index.php?title=Connecting_with_MobaXTerm/fr&oldid=70302)"
