# Béluga Cluster Documentation

**Availability:** March, 2019

**Login node:** beluga.alliancecan.ca

**Globus Endpoint:** computecanada#beluga-dtn

**Data Transfer Node (rsync, scp, sftp,...):** beluga.alliancecan.ca

**Portal:** https://portail.beluga.calculquebec.ca/


Béluga is a general-purpose cluster designed for a variety of workloads, located at the École de technologie supérieure in Montreal.  The cluster is named in honor of the St. Lawrence River's Beluga whale population.


## Site-Specific Policies

By policy, Béluga's compute nodes cannot access the internet. If you need an exception, contact technical support explaining your needs.

Crontab is not offered on Béluga.

Each job should have a duration of at least one hour (five minutes for test jobs). A user cannot have more than 1000 jobs (running and queued) at any given time. The maximum job duration is 7 days (168 hours).


## Storage

### HOME

*   Lustre filesystem, 105 TB of space
*   Location of home directories; each has a small fixed quota. Use the PROJECT space for larger storage needs.
*   Small fixed quota per user.
*   Daily backups of home directories.

### SCRATCH

*   Lustre filesystem, 2.6 PB of space
*   Large space for temporary files during computations.
*   No backup system.
*   Large fixed quota per user.
*   Automated purge of older files.

### PROJECT

*   Lustre filesystem, 25 PB of space
*   Designed for sharing data among research group members and storing large datasets.
*   Large adjustable quota per group.
*   Daily backups of the project space.

For Globus data transfers, use the endpoint `computecanada#beluga-dtn`. For tools like rsync and scp, use the login node.


## High-Performance Interconnect

A Mellanox Infiniband EDR (100 Gb/s) network connects all cluster nodes. A central 324-port switch links the cluster's island topology with a maximum blocking factor of 5:1. Storage servers have a non-blocking connection.  The architecture allows multiple parallel jobs with up to 640 cores (or more) due to the non-blocking network. For jobs requiring greater parallelism, the blocking factor is 5:1, but even across multiple islands, interconnection remains high-performance.


## Node Characteristics

Turbo mode is activated on all compute nodes.

| Nodes | Cores | Available Memory      | CPU                                      | Storage     | GPU                                     |
|-------|-------|-----------------------|-------------------------------------------|-------------|-----------------------------------------|
| 160   | 40    | 92G or 95000M         | 2 x Intel Gold 6148 Skylake @ 2.4 GHz     | 1 x SSD 480G | -                                        |
| 579   | 40    | 186G or 191000M        | 2 x Intel Gold 6148 Skylake @ 2.4 GHz     | 1 x SSD 480G | -                                        |
| 10    |       |                       |                                           | 6 x SSD 480G |                                         |
| 51    | 40    | 752G or 771000M        | 2 x Intel Gold 6148 Skylake @ 2.4 GHz     | 1 x SSD 480G | -                                        |
| 2     |       |                       |                                           | 6 x SSD 480G |                                         |
| 172   | 40    | 186G or 191000M        | 2 x Intel Gold 6148 Skylake @ 2.4 GHz     | 1 x NVMe SSD 1.6T | 4 x NVidia V100SXM2 (16G memory), NVLink |


To increase `$SLURM_TMPDIR` space, submit jobs with `--tmp=xG`, where `x` is between 350 and 2490.


## Monitoring Jobs

Monitor CPU and GPU usage (past and current) in real time via the [portal](https://portail.beluga.calculquebec.ca/) to maximize resource use and reduce queue wait times.  For each job, you can monitor compute core, memory, and GPU usage.  If resources are underutilized, adjust your requests accordingly (e.g., if requesting four cores but only using one).
