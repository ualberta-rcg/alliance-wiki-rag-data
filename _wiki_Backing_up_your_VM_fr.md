# Backing Up Your VM

There are several strategies for backing up a VM instance and restoring it in case of problems; the choice of strategy depends on your specific needs and situation.  It is strongly recommended to create backups outside the cloud. A frequently applied backup rule is the 3-2-1 rule: three copies of your data on at least two different media types, with one copy offsite. We discuss some common methods for backing up your instance and preserving its state, and give an example of a combination of these methods that represents a complete backup strategy.

## File Backup

Many strategies used with physical computers also apply to virtual instances; for example, `rsync`, `duplicity`, `borg`, and `restic` are tools that can remotely back up your instance's data.

## Automated Configuration

Provisioning tools like `ansible`, `puppet`, `chef`, and `saltstack` can be used to automate the configuration of software and the operating system. With the appropriate specification files for each of these tools, it is very easy to recreate an instance.  Specification files can be managed by a version control application like `git`. Provisioning and orchestration tools (e.g., Heat and Terraform) can be used together to automate the entire process of creating an instance and configuring software; see [Automating VM creation](link-to-automating-vm-creation-page); data that would not then be generated or created will need to be backed up using a method mentioned in the [File Backup](#file-backup) section.

## OpenStack Backup Methods

OpenStack offers two options for storage:

*   Storage in a volume, with triple replication; this option protects data in case of hardware problems, but not in case of accidental or malicious deletion.
*   Ephemeral storage on a local node; this option also protects against hardware problems, but should not be used with critical data; it is mainly used temporarily.

OpenStack also offers tools to create disk images and instance snapshots. The main instance templates (persistent and compute) have different behaviors; we recommend different backup procedures for each template.

### Persistent Instances

Persistent instances are designed to be booted from a volume. A backup is created when a copy of the volume(s) associated with the instance is created. However, this does not include the instance template, its public IP, and its security rules. The best way to create a backup of a volume is therefore to create an image of that volume. This image can then be downloaded and reused to create several new instances; you can access it via VirtualBox from your personal computer; or upload it to another cloud.

To create an image from a volume, that volume must be detached from the instance.  Furthermore, if the volume is the root volume of the instance, it cannot be detached without the instance being deleted. You can delete your instance without losing data provided you have not checked "Delete volume when deleting the instance" when creating the instance; note that OpenStack will not signal you that this box has been checked. One way to get around this is to create a snapshot of the volume; however, make sure your storage quota allows it since snapshots are counted. Since a volume cannot be deleted if a snapshot of that volume has been created, the volume will not be deleted if you delete the instance, regardless of whether you have checked the box in question.

The status of all volumes you want to create an image from should then be Available. To create an image from a volume, select "Load into image" from the dropdown menu for the volume. Select the QCOW2 format and enter a name for the image. There are several formats for disk images, but QCOW2 works well with OpenStack and typically takes up less space than Raw format images. The other formats vmdk and vdi are useful when working with other visualization tools like VirtualBox.

Once you have created the images of all the volumes you want to back up, you can then recreate the instance from the original root volume of the instance and, if necessary, attach the additional volumes that you would have attached to the original instance.

#### Volume Snapshot

You can also create a snapshot of the volume to preserve its current state; however, this is not an ideal backup solution since the original volume should not be modified.  Furthermore, it is not possible to download a snapshot since it depends on the original volume. It is however possible to create a new volume from one of its snapshots if, for example, some files have been modified since the snapshot was created and the modifications do not need to be saved, or if modifications to the original instance should not be propagated to other instances.

#### Instance Snapshot

The behavior of an instance snapshot depends on the template of that instance. In the case of a persistent instance, OpenStack creates a nearly empty image that contains pointers to the volume snapshots. These pointers point to the snapshots of the boot volume of the persistent instance and the other volumes that were created when the instance snapshot was created. You can then create a new instance (Start from an image (creates a volume)), which creates new volumes from the previously created snapshots, starts a new instance from the root volume, and attaches any other duplicated volume.

### Compute Instances

As with creating a persistent instance, the main goal is to create an image of the root disk at least, but also if necessary of the volumes attached to it. However, the process for creating an image is different in the case of type c (compute) templates. Unlike persistent instances, these are not designed to boot from a volume accessed over the network, but rather from disk images that reside on the computer where the instance is running. This means there is no volume in the OpenStack dashboard that you can click on to create the image of your root disk. To do this, you must click on "Create a snapshot" in the "Overview" tab of the instance. As this happens when creating an image with a persistent instance, this creates an image; in this case, however, the image is not as empty (i.e., it does not only contain pointers to the volume snapshots), but contains a copy of the root disk of the instance.

Compute instances have an additional data disk mounted on `/mnt` whose data is not part of the instance image.  It is therefore necessary to proceed differently to back up this data, for example by copying it from the disk before the instance is terminated.

## Example Backup Strategy

It can be difficult to manage images larger than 10-20GB which take a long time to download and create instances. A good strategy is to isolate large datasets from the operating system and software. A backup of the operating system and software can be done with a disk image or they can be recreated with a provisioning application (see [Automated Configuration](#automated-configuration)). The datasets can then be copied to a remote location using a common backup method. If you are using database software such as MySQL or PostgreSQL, you will want to dump your databases including the backup.  Finally and most importantly, perform tests to see if your backups have successfully restored what was required.

## See Also

*   [Command Line Clients](link-to-command-line-clients-page)
*   [Creating an Image from an Instance](link-to-creating-image-page)
*   [Downloading an Image](link-to-downloading-image-page)
*   [Uploading an Image](link-to-uploading-image-page)
*   [Synchronizing Data](link-to-synchronizing-data-page)


**(Remember to replace the bracketed links with the actual links to the relevant pages.)**
