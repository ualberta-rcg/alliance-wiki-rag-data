# CephFS

Other languages: English, français

CephFS provides a common filesystem that can be shared amongst multiple OpenStack VM hosts. Access to the service is granted via requests to `cloud@tech.alliancecan.ca`.

This is a fairly technical procedure that assumes basic Linux skills for creating/editing files, setting permissions, and creating mount points. For assistance in setting up this service, write to `cloud@tech.alliancecan.ca`.

## Procedure

### Request access to shares

If you do not already have a quota for the service, you will need to request this through `cloud@tech.alliancecan.ca`. In your request, please provide the following:

* OpenStack project name
* Amount of quota required (in GB)
* Number of shares required

### OpenStack configuration: Create a CephFS share

1. Create the share. In `Project --> Share --> Shares`, click on `+Create Share`.
2. **Share Name**: Enter a name that identifies your project (e.g., `project-name-shareName`).
3. **Share Protocol**: CephFS
4. **Size**: Size you need for this share
5. **Share Type**: cephfs
6. **Availability Zone**: nova
7. Do not check "Make visible for all", otherwise the share will be accessible by all users in all projects.
8. Click on the `Create` button.

### Configuration of CephFS on Horizon GUI

1. Create an access rule to generate an access key. In `Project --> Share --> Shares --> Actions` column, select `Manage Rules` from the drop-down menu.
2. Click on the `+Add Rule` button (right of the page).
3. **Access Type**: cephx
4. **Access Level**: Select `read-write` or `read-only` (you can create multiple rules for either access level if required).
5. **Access To**: Select a key name that describes the key. This name is important because it will be used in the cephfs client configuration on the VM; on this page, we use `MyCephFS-RW`.

**Properly configured CephFS**

Note the share details which you will need later. In `Project --> Share --> Shares`, click on the name of the share. In the `Share Overview`, note the three elements circled in red in the "Properly configured" image: `Path`, which will be used in the `mount` command on the VM, the `Access Rules`, which will be the client name and the `Access Key` that will let the VM's client connect.


### Attach the CephFS network to your VM

#### On Arbutus

On Arbutus, the cephFS network is already exposed to your VM; there is nothing to do here, go to the VM configuration section.

#### On SD4H/Juno

On SD4H/Juno, you need to explicitly attach the cephFS network to the VM.

**With the Web GUI:**

For each VM you need to attach, select `Instance --> Action --> Attach interface`, select the CephFS-Network, leave the `Fixed IP Address` box empty.

**With the Openstack client:**

1. List the servers and select the ID of the server you need to attach to the CephFS:
   ```bash
   openstack server list
   ```
2. Select the ID of the VM you want to attach (example using the first one):
   ```bash
   openstack server add network 1b2a3c21-c1b4-42b8-9016-d96fc8406e04 CephFS-Network
   ```
3. Verify the attachment:
   ```bash
   openstack server list
   ```

### VM configuration: install and configure CephFS client

#### Required packages for the Red Hat family (RHEL, CentOS, Fedora, Rocky, Alma)

Check the available releases at [https://download.ceph.com/](https://download.ceph.com/) and look for recent `rpm-*` directories. As of July 2024, `quincy` is the latest stable release. The compatible distributions (distros) are listed at [https://download.ceph.com/rpm-quincy/](https://download.ceph.com/rpm-quincy/).

Here we show configuration examples for Enterprise Linux 8 and Enterprise Linux 9.

Install relevant repositories for access to ceph client packages:

**Enterprise Linux 8 - el8:**

```
File: /etc/yum.repos.d/ceph.repo
[Ceph]
name = Ceph packages for $basearch
baseurl = http://download.ceph.com/rpm-quincy/el8/$basearch
enabled = 1
gpgcheck = 1
type = rpm-md
gpgkey = https://download.ceph.com/keys/release.asc
[Ceph-noarch]
name = Ceph noarch packages
baseurl = http://download.ceph.com/rpm-quincy/el8/noarch
enabled = 1
gpgcheck = 1
type = rpm-md
gpgkey = https://download.ceph.com/keys/release.asc
[ceph-source]
name = Ceph source packages
baseurl = http://download.ceph.com/rpm-quincy/el8/SRPMS
enabled = 1
gpgcheck = 1
type = rpm-md
gpgkey = https://download.ceph.com/keys/release.asc
```

**Enterprise Linux 9 - el9:**

```
File: /etc/yum.repos.d/ceph.repo
[Ceph]
name = Ceph packages for $basearch
baseurl = http://download.ceph.com/rpm-quincy/el9/$basearch
enabled = 1
gpgcheck = 1
type = rpm-md
gpgkey = https://download.ceph.com/keys/release.asc
[Ceph-noarch]
name = Ceph noarch packages
baseurl = http://download.ceph.com/rpm-quincy/el9/noarch
enabled = 1
gpgcheck = 1
type = rpm-md
gpgkey = https://download.ceph.com/keys/release.asc
[ceph-source]
name = Ceph source packages
baseurl = http://download.ceph.com/rpm-quincy/el9/SRPMS
enabled = 1
gpgcheck = 1
type = rpm-md
gpgkey = https://download.ceph.com/keys/release.asc
```

The epel repo also needs to be in place:

```bash
sudo dnf install epel-release
```

You can now install the ceph lib, cephfs client and other dependencies:

```bash
sudo dnf install -y libcephfs2 python3-cephfs ceph-common python3-ceph-argparse
```

#### Required packages for the Debian family (Debian, Ubuntu, Mint, etc.)

1. Get the repository codename with:
   ```bash
   lsb_release -sc
   ```
2. Add the repository:
   ```bash
   sudo apt-add-repository 'deb https://download.ceph.com/debian-quincy/ {codename} main'
   ```
3. Install the packages:
   ```bash
   sudo apt-get install -y libcephfs2 python3-cephfs ceph-common python3-ceph-argparse
   ```

#### Configure ceph client

Once the client is installed, you can create a `ceph.conf` file. Note the different `mon host` for the different clouds.

**Arbutus:**

```
File: /etc/ceph/ceph.conf
[global]
admin socket = /var/run/ceph/$cluster-$name-$pid.asok
client reconnect stale = true
debug client = 0/2
fuse big writes = true
mon host = 10.30.201.3:6789,10.30.202.3:6789,10.30.203.3:6789
[client]
quota = true
```

**SD4H/Juno:**

```
File: /etc/ceph/ceph.conf
[global]
admin socket = /var/run/ceph/$cluster-$name-$pid.asok
client reconnect stale = true
debug client = 0/2
fuse big writes = true
mon host = 10.65.0.10:6789,10.65.0.12:6789,10.65.0.11:6789
[client]
quota = true
```

You can find the monitor information in the share details `Path` field that will be used to mount the volume. If the value of the web page is different than what is seen here, it means that the wiki page is out of date.

You also need to put your client name and secret in the `ceph.keyring` file:

```
File: /etc/ceph/ceph.keyring
[client.MyCephFS-RW]
key = <access Key>
```

Again, the access key and client name (here MyCephFS-RW) are found under the access rules on your project web page. Look for `Project --> Share --> Shares`, then click on the name of the share.

Retrieve the connection information from the share page for your connection. Open up the share details by clicking on the name of the share in the `Shares` page. Copy the entire path of the share to mount the filesystem.

**Mount the filesystem:**

Create a mount point directory somewhere in your host (`/cephfs` is used here):

```bash
mkdir /cephfs
```

You can use the ceph driver to permanently mount your CephFS device by adding the following in the VM fstab:

**Arbutus:**

```
File: /etc/fstab
:/volumes/_nogroup/f6cb8f06-f0a4-4b88-b261-f8bd6b03582c /cephfs/ ceph name=MyCephFS-RW 0  2
```

**SD4H/Juno:**

```
File: /etc/fstab
:/volumes/_nogroup/f6cb8f06-f0a4-4b88-b261-f8bd6b03582c /cephfs/ ceph name=MyCephFS-RW,mds_namespace=cephfs_4_2,x-systemd.device-timeout=30,x-systemd.mount-timeout=30,noatime,_netdev,rw 0  2
```

Notice the non-standard `:` before the device path. It is not a typo! The mount options are different on different systems. The namespace option is required for SD4H/Juno, while other options are performance tweaks.

You can also do the mount directly from the command line:

**Arbutus:**

```bash
sudo mount -t ceph :/volumes/_nogroup/f6cb8f06-f0a4-4b88-b261-f8bd6b03582c /cephfs/ -o name=MyCephFS-RW
```

**SD4H/Juno:**

```bash
sudo mount -t ceph :/volumes/_nogroup/f6cb8f06-f0a4-4b88-b261-f8bd6b03582c /cephfs/ -o name=MyCephFS-RW,mds_namespace=cephfs_4_2,x-systemd.device-timeout=30,x-systemd.mount-timeout=30,noatime,_netdev,rw
```

CephFS can also be mounted directly in user space via ceph-fuse.

1. Install the ceph-fuse lib:
   ```bash
   sudo dnf install ceph-fuse
   ```
2. Let the fuse mount be accessible in userspace by uncommenting `user_allow_other` in the `fuse.conf` file:

   ```
   File: /etc/fuse.conf
   # mount_max = 1000
   user_allow_other
   ```

3. Mount in user's home:
   ```bash
   mkdir ~/my_cephfs
   ceph-fuse my_cephfs/ --id=MyCephFS-RW --conf=~/ceph.conf --keyring=~/ceph.keyring --client-mountpoint=/volumes/_nogroup/f6cb8f06-f0a4-4b88-b261-f8bd6b03582c
   ```

   Note that the client name is here the `--id`. The `ceph.conf` and `ceph.keyring` content are exactly the same as for the ceph kernel mount.


## Notes

A particular share can have more than one user key provisioned for it. This allows a more granular access to the filesystem, for example, if you needed some hosts to only access the filesystem in a read-only capacity. If you have multiple keys for a share, you can add the extra keys to your host and modify the above mounting procedure. This service is not available to hosts outside of the OpenStack cluster.


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=CephFS&oldid=170412](https://docs.alliancecan.ca/mediawiki/index.php?title=CephFS&oldid=170412)"
