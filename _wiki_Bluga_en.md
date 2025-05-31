# Béluga Cluster Documentation

**Availability:** March, 2019

**Login node:** beluga.alliancecan.ca

**Globus Endpoint:** computecanada#beluga-dtn

**Data Transfer Node (rsync, scp, sftp, ...):** beluga.alliancecan.ca

**Portal:** https://portail.beluga.calculquebec.ca/

Béluga is a general-purpose cluster designed for a variety of workloads, located at the École de technologie supérieure in Montreal.  The cluster is named in honor of the St. Lawrence River's Beluga whale population.


## Site-Specific Policies

By policy, Béluga's compute nodes cannot access the internet. If you need an exception, contact technical support explaining your needs.

Crontab is not offered on Béluga.

Each job should have a duration of at least one hour (five minutes for test jobs). A user cannot have more than 1000 jobs (running and queued) at any given time. The maximum job duration is 7 days (168 hours).


## Storage

### HOME

*   **Filesystem:** Lustre
*   **Space:** 105 TB
*   **Description:** Location of home directories; each has a small, fixed quota. Use project space for larger storage needs.
*   **Quota:** Small fixed quota per user.
*   **Backup:** Daily backup of home directories.

### SCRATCH

*   **Filesystem:** Lustre
*   **Space:** 2.6 PB
*   **Description:** Large space for temporary files during computations.
*   **Backup:** No backup system.
*   **Quota:** Large fixed quota per user.
*   **Purge:** Automated purge of older files.

### PROJECT

*   **Filesystem:** Lustre
*   **Space:** 25 PB
*   **Description:** Designed for sharing data among research group members and storing large datasets.
*   **Quota:** Large adjustable quota per group.
*   **Backup:** Daily backup of project space.

For Globus data transfer, use the endpoint `computecanada#beluga-dtn`. For tools like rsync and scp, use the login node.


## High-Performance Interconnect

A Mellanox Infiniband EDR (100 Gb/s) network connects all cluster nodes. A central 324-port switch links the cluster's island topology with a maximum blocking factor of 5:1. Storage servers have a non-blocking connection.  The architecture allows multiple parallel jobs with up to 640 cores (or more) due to the non-blocking network. For jobs requiring greater parallelism, the blocking factor is 5:1, but even across multiple islands, interconnection remains high-performance.


## Node Characteristics

Turbo mode is activated on all compute nodes.

| Nodes | Cores | Available Memory     | CPU                                     | Storage      | GPU                                      |
|-------|-------|----------------------|------------------------------------------|---------------|-------------------------------------------|
| 160   | 40    | 92G or 95000M         | 2 x Intel Gold 6148 Skylake @ 2.4 GHz | 1 x SSD 480G | -                                           |
| 579   | 40    | 186G or 191000M        | 2 x Intel Gold 6148 Skylake @ 2.4 GHz | 1 x SSD 480G | -                                           |
| 10    |       |                       |                                          | 6 x SSD 480G |                                           |
| 51    | 40    | 752G or 771000M        | 2 x Intel Gold 6148 Skylake @ 2.4 GHz | 1 x SSD 480G | -                                           |
| 2     |       |                       |                                          | 6 x SSD 480G |                                           |
| 172   | 40    | 186G or 191000M        | 2 x Intel Gold 6148 Skylake @ 2.4 GHz | 1 x NVMe SSD 1.6T | 4 x NVidia V100SXM2 (16G memory), NVLink |


To increase `$SLURM_TMPDIR` space, submit jobs with `--tmp=xG`, where `x` is between 350 and 2490.


## Monitoring Jobs

To optimize resource use and reduce queue wait times, monitor past and current CPU and GPU tasks in real time via the [portal](https://portail.beluga.calculquebec.ca/).

For each job, you can monitor:

*   Compute core usage
*   Memory usage
*   GPU usage

If compute resources are underutilized, adjust your requests accordingly. For example, if you request four cores but only use one, modify your submission file.
