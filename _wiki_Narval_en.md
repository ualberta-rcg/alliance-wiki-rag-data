# Narval Cluster Documentation

This document provides information about the Narval compute cluster.

## Availability

Since October 2021

## Login Node

`narval.alliancecan.ca`

## Globus Collection

Compute Canada - Narval

## Data Transfer Node (rsync, scp, sftp, ...)

`narval.alliancecan.ca`

## Portal

https://portail.narval.calculquebec.ca/

## Overview

Narval is a general-purpose cluster located at the École de technologie supérieure in Montreal.  It's named after the narwhal, a whale species sometimes seen in the Gulf of St. Lawrence.


## Site-Specific Policies

*   **Internet Access:** Compute nodes cannot access the internet. Contact technical support for exceptions.
*   **Crontab:** Not available on Narval.
*   **Job Duration:** Minimum job duration is one hour (five minutes for test jobs).  Maximum duration is 7 days (168 hours).
*   **Job Limits:**  A maximum of 1000 jobs (running or queued) are permitted at any time.


## Storage

### HOME

*   Lustre filesystem
*   40 TB of space
*   Location of home directories
*   Small, fixed quota per user
*   Daily backups


### SCRATCH

*   Lustre filesystem
*   5.5 PB of space
*   For temporary files
*   No backups
*   Large quota per user
*   Automated purge of older files


### PROJECT

*   Lustre filesystem
*   19 PB of space
*   For sharing data within research groups and storing large datasets
*   Large, adjustable quota per group
*   Daily backups


**Data Transfer:** Use the Globus endpoint specified at the top of this page. For `rsync` and `scp`, use the login node.


## High-Performance Interconnect

The cluster uses a Mellanox HDR InfiniBand network.  Each hub (40 HDR ports at 200 Gb/s) connects up to 66 nodes with HDR100 (100 Gb/s) using 33 HDR links divided in two by special cables. Seven remaining HDR links connect the hub to a rack containing seven central HDR InfiniBand hubs.  This creates islands of nodes connected with a maximum blocking factor of 33:7 (4.7:1). Storage servers have a much lower blocking factor for optimized performance.

Narval racks contain islands of 48 or 56 regular CPU nodes.  Parallel jobs can utilize up to 3584 cores with a non-blocking network. For larger or fragmented jobs, the blocking factor is 4.7:1.


## Node Characteristics

| Nodes | Cores | Available Memory      | CPU                                         | Storage     | GPU                                      |
|-------|-------|-----------------------|----------------------------------------------|-------------|------------------------------------------|
| 1145  | 64    | 249 GB or 255000 MB   | 2 x AMD EPYC 7532 (Zen 2) @ 2.40 GHz, 256 MB L3 cache | 1 x 960 GB SSD |                                          |
| 33    |       | 2009 GB or 2057500 MB | 3                                            | 4000 GB     |                                          |
| 159   | 48    | 498 GB or 510000 MB   | 2 x AMD EPYC 7413 (Zen 3) @ 2.65 GHz, 128 MB L3 cache | 1 x 3.84 TB SSD | 4 x Nvidia A100SXM4 (40 GB memory), NVLink |


## AMD Processors

### Supported Instruction Sets

Narval uses 2nd and 3rd generation AMD EPYC processors supporting the AVX2 instruction set. This is the same as Intel processors on Béluga, Cedar, Graham, and Niagara.  AVX512 is *not* supported (unlike Béluga, Niagara, and some Cedar/Graham nodes).

Applications compiled on Broadwell nodes (Cedar/Graham) will run on Narval.  However, applications compiled on Béluga, Niagara, or Skylake/Cascade Lake nodes (Cedar/Graham) will require recompilation.


### Intel Compilers

Intel compilers can build for Narval's AMD processors using AVX2 and earlier instruction sets. Use the `-march=core-avx2` option for compatibility with both Intel and AMD processors.

Avoid options like `-xXXXX` (e.g., `-xCORE-AVX2`) as these add Intel-specific checks that will fail on Narval.  `-xHOST` and `-march=native` are equivalent to `-march=pentium` and should not be used.


### Software Environments

The standard environment is StdEnv/2023. Older versions (2016 and 2018) are blocked. Contact technical support for older software needs.


### BLAS and LAPACK Libraries

While Intel MKL works, FlexiBLAS is preferred. See the BLAS and LAPACK documentation for details.


## Monitoring Jobs

The Narval portal allows real-time monitoring of CPU and GPU usage for current and past jobs. This helps optimize resource usage and reduce queue times.  Monitor compute node, memory, and GPU usage.  Adjust job requests if resources are underutilized.
