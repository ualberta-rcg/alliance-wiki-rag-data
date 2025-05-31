# Connecting with MobaXterm

Connecting with MobaXterm works similarly to PuTTY (see [Connecting with PuTTY](Connecting_with_PuTTY)). However, MobaXterm offers more integrated functionality. It includes a built-in SFTP client for file transfer and a built-in X11 server, eliminating the need for a third-party X11 server to run graphical programs remotely.  If you've used PuTTY and saved sessions, MobaXterm will utilize those, avoiding the need to re-enter settings.


To connect to a new machine, go to `Sessions` -> `New session`, select an "SSH" session, enter the remote host address and your username (you might need to check the "Specify username" box). Click "OK". MobaXterm saves this session information and opens an SSH connection, prompting for your password.  Upon successful authentication, you'll have a terminal and an SFTP client (left pane) to view and transfer files via drag-and-drop.


## X11 Forwarding

To enable X11 forwarding for graphical applications:

1.  **Enable X11 Forwarding for the Session:** Right-click the session, select "Edit Session," go to "Advanced SSH settings," and check the "X11-Forwarding" box.
2.  **Ensure X Server is Running:** Verify the "X server" icon (top-right corner of the main window) is green. If red, click it to start the X server.
3.  **Test X11 Forwarding:** Open the session, enter your password, and run a simple GUI program like `xclock`. A clock popup confirms successful X11 forwarding.


## Using a Key Pair

1. Right-click the session in the left "Sessions" pane and select "Edit Session".
2. In the session settings window, select "Advanced SSH settings" and check the "Use private key" checkbox.
3. Click the icon to the right of the text box to browse and select your private key file.
4. To create a key pair, see [Generating SSH keys in Windows](Generating_SSH_keys_in_Windows).


**(Images omitted as per instructions.  Original images were: Creating an SSH session, Connected to a remote host, Enabling X11 Forwarding, Specifying a private key)**
