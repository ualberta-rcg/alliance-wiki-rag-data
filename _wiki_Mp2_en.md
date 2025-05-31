# Mp2

This page is a translated version of the page Mp2 and the translation is 100% complete.

**Other languages:**

*   English
*   français

**Availability:** February 2012 - April 1, 2020

**Login Node:**

mp2.calculcanada.ca

**Globus Endpoint:**

computecanada#mammouth

**Data Transfer Node (rsync, scp, sftp,...):**

mp2.calculcanada.ca

Mp2 is now exclusive to researchers from the Université de Sherbrooke.

Mammouth-Mp2 is a heterogeneous and polyvalent cluster designed for ordinary computations; it is located at the Université de Sherbrooke.


## Site-specific policies

Each job must have a duration of at least one hour (at least five minutes for test jobs) and a user cannot have more than 1000 jobs (running and queued) at any given time. The maximum duration of a job is 168 hours (seven days).

No GPUs.


## Storage

### HOME

*   **Filesystem:** Lustre
*   **Total Space:** 79.6 TB
*   **Description:** This space is small and cannot be expanded; you should use your project space for substantial storage needs.
*   **Per-user quota:** 50 GB of space and 500K files.
*   **Backup:** Daily backup.

### SCRATCH

*   **Filesystem:** Lustre
*   **Total Space:** 358.3 TB
*   **Description:** Large space for storing temporary files during computations.
*   **Per-user quota:** 20 TB of space and 1M files.
*   **Backup:** No backup system in place.

### PROJECT

*   **Filesystem:** Lustre
*   **Total Space:** 716.6 TB
*   **Description:** This space is designed for sharing data among the members of a research group and for storing large amounts of data.
*   **Per-group quota:** 1 TB of space and 500K files.
*   **Backup:** No backup system in place.

For transferring data by Globus, you should use the endpoint `computecanada#mammouth`, whereas tools like rsync and scp can simply use an ordinary login node.


## High-performance interconnect

The Mellanox QDR (40 Gb/s) Infiniband network links together all of the cluster's nodes and is non-blocking for groups of 216 nodes and 5:1 for the rest of the cluster.


## Node characteristics

| Quantity | Cores | Available Memory | CPU Type                                                              | Storage     | GPU Type |
| -------- | ----- | ------------------ | --------------------------------------------------------------------- | ----------- | -------- |
| 1588     | 24    | 31 GB (31744 MB)    | 12 cores/socket, 2 sockets/node. AMD Opteron Processor 6172 @ 2.1 GHz | 1 TB SATA   | -        |
| 20       | 48    | 251 GB (257024 MB)   | 12 cores/socket, 4 sockets/node. AMD Opteron Processor 6174 @ 2.2 GHz | 1 TB SATA   | -        |
| 2        | 48    | 503 GB (515072 MB)   | 12 cores/socket, 4 sockets/node. AMD Opteron Processor 6174 @ 2.2 GHz | 1 TB SATA   | -        |


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Mp2/en&oldid=164517](https://docs.alliancecan.ca/mediawiki/index.php?title=Mp2/en&oldid=164517)"
