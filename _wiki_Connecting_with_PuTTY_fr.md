# Connecting with PuTTY

This page is a translated version of the page Connecting with PuTTY and the translation is 100% complete.

Other languages: English, français

## Connecting to the Server

Enter the server name or address (click to enlarge)

Enter the username; this field is not mandatory since the name can be entered at login (click to enlarge)

Enable X11 forwarding (click to enlarge)

Create the SSH key (click to enlarge)


Start PuTTY and enter the name or address of the server you want to connect to.

Parameters can be saved for future use: enter the name in the *Save Session* field and click the *Save* button to the right of the list of names.

You can also save the username for connection to a particular server: under *Category->Connection->Data*, enter the username in the *Auto-login username* field.  It will no longer be necessary to enter the username to connect.


## X11 Forwarding

To use graphical applications, enable X11 forwarding: under *Connection->SSH->X11*, check *Enable X11 forwarding*.

The X11 forwarding function requires an X window server such as Xming or, for recent versions of Windows, VcXsrv. The X window server should be running before establishing the SSH connection. To test the redirection, open a PuTTY session and run a simple command, for example `xclock`. The display of a pop-up window showing a clock indicates that X11 forwarding is probably working.


## SSH Key Pair

To locate the private key: under *Category->Connection->SSH->Auth*, click the *Browse* button.

PuTTY uses files with the .ppk suffix; these suffixes are generated via PuTTYGen (see [Generating SSH keys in Windows](link-to-generating-ssh-keys-page) to learn how to create these keys).

In newer versions of PuTTY, you must click the "+" sign next to *Auth*, then select *Credentials* to be able to search for the *Private key file for authentication*. In this newer interface, the *Certificate to use* and *Plugin to provide authentication response* fields must be empty.


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Connecting_with_PuTTY/fr&oldid=134700](https://docs.alliancecan.ca/mediawiki/index.php?title=Connecting_with_PuTTY/fr&oldid=134700)"
