# Recovering Data from a Compromised VM

This is a translated version of the page [Recovering data from a compromised VM](link-to-english-version-if-available) and the translation is 100% complete.

## What Happens When a Virtual Machine is Compromised?

This is confirmed by our technical support team after analyzing traffic logs and other sources.  The virtual machine is shut down at the sysadmin level. You will receive an email to this effect.

## Why Do You Need to Rebuild the Virtual Machine?

You cannot start a virtual machine that has been locked at the sysadmin level. The content of the virtual machine is no longer intact, but it is relatively safe to extract the data from it. You must build a new virtual machine.


## How to Proceed

1. **Email:** Write to nuage@tech.alliancecan.ca explaining your recovery plan. If access to the file systems is necessary, the volume will be unlocked by our technical support team.

2. **Connect to OpenStack Console:** Log in to the OpenStack console.

3. **Launch a New Instance:** Launch a new instance to be used for recovery.

4. **Detach the Volume:** In "Volumes," select "Manage attachments" from the dropdown menu to the right of the compromised volume and click the "Detach volume" button.

5. **Attach the Volume:** In "Volumes," select "Manage attachments" from the dropdown menu to the right of the compromised volume and click the "Attach volume" button (select the instance you just launched).

6. **SSH Connection:** Connect via SSH to the new instance. The compromised volume is the `vdb` disk.

7. **Mount the Filesystem:** Mounting the correct filesystem from a partition or an LVM (logical volume manager) depends heavily on how the base operating system image was created. You will need an experienced person to complete the data recovery.


**(Retrieved from "https://docs.alliancecan.ca/mediawiki/index.php?title=Recovering_data_from_a_compromised_VM/fr&oldid=137251")**
