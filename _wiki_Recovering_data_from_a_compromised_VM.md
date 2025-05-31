# Recovering Data from a Compromised VM

This page outlines the steps to recover data from a compromised virtual machine (VM).  The information provided is not exhaustive.

## What Happens When We Detect a Compromised VM?

1. Our support team investigates network traffic logs and other sources to confirm the compromise.
2. The VM is shut down and locked at the sysadmin level.
3. You are notified by email.

## Why Do You Need to Rebuild?

* You cannot start an administratively locked VM.
* The contents of the compromised VM are no longer trustworthy.  However, it is relatively safe to extract the data.
* You must build a new VM.

## What Steps Should You Take?

1. **Email Cloud Support:** Send an email to `cloud@tech.alliancecan.ca` outlining your recovery plan. If filesystem access is required, the cloud support team will unlock the volume.

2. **Log in to OpenStack:** Log in to the OpenStack admin console.

3. **Launch a Recovery Instance:** Launch a new instance that will be used for data rescue operations.

4. **Detach the Compromised Volume:**
    * Under "Volumes", select "Manage Attachments" from the dropdown menu for the compromised volume.
    * Click the "Detach Volume" button.

5. **Attach to Recovery Instance:**
    * Under "Volumes", select "Manage Attachments" for the compromised volume.
    * Select "Attach To Instance" and choose the recovery instance you just launched.

6. **SSH into Recovery Instance:**  SSH into your recovery instance. The old, compromised volume will be available as the "vdb" disk.

7. **Mount the Filesystem:** Mounting the appropriate filesystem (partition or LVM logical volume) depends on how the base OS image was created.  Instructions vary greatly; contact someone with experience to proceed with this step.
