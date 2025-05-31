# Mammouth-Mp2 Cluster Documentation

**Other languages:** English, français

**Availability:** February 2012 - April 2020

**Connection Node:** mp2.calculcanada.ca

**Globus Endpoint:** computecanada#mammouth

**Copy Node (rsync, scp, sftp, ...):** mp2.calculcanada.ca

Mp2 is now exclusive to researchers from the Université de Sherbrooke.

Mammouth-Mp2 is a heterogeneous and versatile cluster designed for general-purpose computing; it is located at the Université de Sherbrooke.


## Contents

1. Particularities
2. Storage
3. High-Performance Networking
4. Node Types and Characteristics


## Particularities

Each task should be at least one hour long (at least five minutes for test tasks), and a user can have no more than 1000 tasks (running and pending) at a time. The maximum duration of a task is 7 days (168 hours).

No GPUs.


## Storage

### HOME

*   File system: Lustre
*   Total space: 79.6 TB
*   This space is small and cannot be expanded: you will need to use your `project` space for large storage needs.
*   50 GB of space and 500K files per user.
*   Automatic backup once a day.

### SCRATCH

*   File system: Lustre
*   Total space: 358.3 TB
*   Large space for storing temporary files during calculations.
*   20 TB of space and 1M files per user.
*   No automatic backup system.

### PROJECT

*   File system: Lustre
*   Total space: 716.6 TB
*   This space is designed for data sharing between group members and for storing large amounts of data.
*   1 TB of space and 500K files per group.
*   No automatic backup system.

For data transfers via Globus, the endpoint `computecanada#mammouth` should be used, while for tools like rsync and scp, a connection node can be used.


## High-Performance Networking

The Mellanox Infiniband QDR (40 Gb/s) network connects all nodes in the cluster, non-blocking on 216 nodes, 5:1 for the rest.


## Node Types and Characteristics

| Quantity | Cores | Available Memory | CPU Type                                      | Storage      | GPU Type |
| -------- | ----- | ------------------ | ---------------------------------------------- | ------------- | -------- |
| 1588     | 24    | 31 GB (31744 MB)    | 12 cores/socket, 2 sockets/node. AMD Opteron Processor 6172 @ 2.1 GHz | 1TB SATA disk | -        |
| 20       | 48    | 251 GB (257024 MB)   | 12 cores/socket, 4 sockets/node. AMD Opteron Processor 6174 @ 2.2 GHz | 1TB SATA disk | -        |
| 2        | 48    | 503 GB (515072 MB)   | 12 cores/socket, 4 sockets/node. AMD Opteron Processor 6174 @ 2.2 GHz | 1TB SATA disk | -        |


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Mp2&oldid=164513](https://docs.alliancecan.ca/mediawiki/index.php?title=Mp2&oldid=164513)"
