# Cloud Storage Options

Other languages: English, français

The existing storage types available in our clouds are:

## Volume Storage

The standard storage unit for cloud computing; can be attached to and detached from an instance.

## Ephemeral/Disk Storage

Virtual local disk storage tied to the lifecycle of a single instance on a hypervisor's local disk ("c" flavor local disk can be lost).

## Object Storage

Non-hierarchical storage where data is created or uploaded in whole-file form.

## Shared Filesystem Storage

Private network attached storage space (similar to NFS/SMB shares); must be configured on each instance where it is mounted.


## Comparison of Storage Types

Attributes of each storage type are compared in the following table:

| Attribute                                      | Volume Storage | Ephemeral/Disk Storage | Object Storage | Shared Filesystem Storage |
|-------------------------------------------------|-----------------|-------------------------|-----------------|--------------------------|
| Default storage option                         | Yes              | Yes                     | No              | No                       |
| Can be accessed via a web browser              | No              | No                      | Yes             | No                       |
| Access can be restricted for specific source IP ranges | N/A             | N/A                     | Yes (S3 ACL)     | N/A                      |
| Can be mounted on a single VM                   | Yes              | Yes                     | No              | Yes                      |
| Can be mounted on multiple VMs (and across projects) simultaneously | No              | No                      | No              | Yes                      |
| Automatic backups                              | No (manually with snapshots) | No                      | No              | Yes (nightly to tape)     |
| Suitable for write once, read only, and public access | No              | No                      | Yes             | No                       |
| Suitable for data/files that change frequently | Yes              | Yes                     | No              | Yes                      |
| Hierarchical filesystem                        | Yes              | Yes                     | No              | Yes                      |
| Suitable for long-term storage                 | Yes              | No                      | Yes             | Yes                      |
| Suitable mountable dedicated storage for individual servers | Yes              | Only for temporary data | No              | No                       |
| Deleted automatically upon deletion of VM      | No              | Yes                     | No              | No                       |
| Standard magnitude of allocation               | GB              | GB                      | TB              | TB                       |
| Multi-disk fault tolerance                     | Yes              | c-flavors No; p-flavors Yes | Yes             | Yes                      |
| Physical disk-level encryption                 | No              | No                      | No              | No                       |


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Cloud_storage_options&oldid=157618](https://docs.alliancecan.ca/mediawiki/index.php?title=Cloud_storage_options&oldid=157618)"
