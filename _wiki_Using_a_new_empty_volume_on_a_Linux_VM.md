# Using a New Empty Volume on a Linux VM

Other languages: English, français

On most Linux distributions, the following steps can be used to partition, format, and mount a newly created volume.  **NOTE:** If this is not a newly created volume, skip the partition and format steps (as they will result in data loss) and proceed only to the mounting steps.

## Create a Partition

```bash
sudo fdisk /dev/vdb
```

`fdisk` will prompt for commands. Use this sequence of single-character commands:

* `n` => new partition
* `p` => primary (only one partition on the disk)
* `1` => partition number 1
* `<return>` => first sector (use default)
* `<return>` => last sector (use default)
* `w` => write partition table and exit


## Format the Partition

```bash
sudo mkfs -t ext4 /dev/vdb1
```

## Create a Mount Point

```bash
sudo mkdir /media/data
```

## Mount the Volume

```bash
sudo mount /dev/vdb1 /media/data
```

## Auto-Mount on Boot

To automatically mount the volume on boot, edit `/etc/fstab` and add a line like this:

```
/dev/vdb1 /media/data ext4 defaults 0 2
```

For more details about the `/etc/fstab` file, see this [Wikipedia article](link-to-wikipedia-article-here).  If you are not rebooting, you can mount the device just added to `/etc/fstab` with:

```bash
sudo mount -a
```

## Unmounting a Volume

To unmount a volume (e.g., to create an image or attach it to a different VM), it's best to unmount it first to prevent data corruption.  To unmount the volume mounted above:

```bash
sudo umount /media/data
```

This command will only work if no files are being accessed by the operating system or any other program. If files are in use, you'll receive a message indicating that the volume is busy and cannot be unmounted.
