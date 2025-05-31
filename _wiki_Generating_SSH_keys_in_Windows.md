# Generating SSH Keys in Windows

## Generating a Key Pair

The process of generating a key is nearly the same whether you are using PuTTY or MobaXTerm.

With MobaXTerm, go to the menu item Tools->MobaKeyGen (SSH key generator).

With PuTTY, run the PuTTYGen executable.

Both methods display a window for generating a new key or loading an existing one.  The PuTTY window is illustrated below (the MobaXTerm window looks almost identical).

[PuTTYgen before generating a key (Click for larger image)]

[PuTTYgen after generating a key (Click for larger image)]

For "Type of key to generate," select "Ed25519". (Type "RSA" is also acceptable, but set the "Number of bits" to 2048 or greater.)

Click the "Generate" button.  You will then be asked to move your mouse around to generate random data for the key.

Enter a passphrase for your key. Remember this passphrase; you will need it every time you reload PuTTY or MobaXTerm to use this key pair.

Click "Save private key" and choose a meaningful file name; the extension `.ppk` is added to the file name (e.g., `compute_canada.ppk`).

Click "Save public key". It's conventional to save the public key with the same name as the private key, but with the extension `.pub`.


## Installing the Public Part of the Key Pair

### Installing via CCDB

We encourage registering your SSH public key with the CCDB. This lets you use it to log in to any of our HPC clusters. Copy the contents of the box titled "Public key for pasting into OpenSSH ..." and paste it into the box at [CCDB -> Manage SSH Keys](link-to-ccdb-ssh-keys). For more information, see [SSH Keys: Using CCDB](link-to-ccdb-ssh-keys-page).


### Installing Locally

If you don't want to use the CCDB method, you can upload your public key to *each* cluster as follows:

Copy the contents of the box titled "Public key for pasting into OpenSSH ..." and paste it as a single line at the end of `/home/USERNAME/.ssh/authorized_keys` on the cluster you wish to connect to.

Ensure the permissions and ownership of the `~/.ssh` directory and files within are correct, as described in [these instructions](link-to-permissions-instructions).

You can also use `ssh-copy-id` if it's available on your personal computer.


## Connecting Using a Key Pair

Test the new key by connecting to the server using SSH. See:

* [Connecting with PuTTY using a key pair](link-to-putty-instructions)
* [Connecting with MobaXTerm using a key pair](link-to-mobaxterm-instructions)
* [Connecting with WinSCP](link-to-winscp-instructions)

Key generation and usage with PuTTY are demonstrated in this video: [Easily setup PuTTY SSH keys for passwordless logins using Pageant](link-to-video)


## Converting an OpenStack Key

When a key is created on OpenStack, you obtain a key with a ".pem" extension. This key can be converted to a format used by PuTTY by clicking the "Load" button in PuTTYGen. Then select the "All Files (*.*)" filter, select the ".pem" file you downloaded from OpenStack, and click "Open". You should also add a "Key passphrase" at this point to use when accessing your private key and then click "Save private key".

This private key can be used with PuTTY to connect to a VM created with OpenStack. For more about this, see "Launching a VM" on the [Cloud Quick Start](link-to-cloud-quickstart) page.


**(Note:  Please replace bracketed `link-to-...` placeholders with actual links.)**
