# Cedar

This page is a translated version of the page Cedar and the translation is 100% complete.

Other languages: English, français

Availability: Since June 30, 2017, as part of the 2017 resource allocation competition.

Connection node: cedar.alliancecan.ca

Globus endpoint: computecanada#cedar-globus

Cluster status: https://status.alliancecan.ca/

Cedar is a heterogeneous cluster suitable for several types of tasks; it is located at Simon Fraser University. Its name recalls the Western red cedar, the official tree of British Columbia whose spiritual significance is important to the First Nations of the region.

The supplier is Scalar Decisions Inc.; the nodes are Dell products; the high-performance /scratch storage file system is from DDN; the networking is from Intel. A liquid cooling system uses heat exchangers on the back doors.

**IMPORTANT:** Globus version 4 no longer supports endpoints and `computecanada#cedar-dtn` is no longer available. Please use the version 5 endpoint, `computecanada#cedar-globus`.

## Introduction to Cedar

*   [Run tasks](placeholder_link_to_run_tasks)
*   [Transfer data](placeholder_link_to_transfer_data)


## Contents

1.  Storage
2.  High-performance networking
3.  Node characteristics
    *   Select a node type
4.  Changes to the job submission and execution policy
    *   Performance


## Storage

`/home` space

*   Total volume: 526 TB
*   Location of `/home` directories: Each `/home` directory has a small fixed quota, not allocated via the rapid access service or the resource allocation competition; large-scale storage is done on `/project`.
*   Backed up daily.

`/scratch` space

*   Total volume: 5.4 PB
*   High-performance parallel file system.
*   Active or temporary storage.
*   Not allocated.
*   Large fixed quota per user.
*   Inactive data is purged.

`/project` space

*   Total volume: 23 PB
*   External persistent storage.
*   Not suitable for parallel read and write tasks; use `/scratch` instead.
*   Large adjustable quota per project.
*   Backed up daily.


The temporary storage (/scratch) is a Lustre file system based on DDN ES14K technology. It consists of 640 8TB NL-SAS disks, with a dual metadata controller whose disks are SSDs.


## High-performance networking

Intel OmniPath networking (version 1, 100 Gbit/s bandwidth).

A low-latency, high-performance network for all compute nodes and temporary storage.

The architecture has been planned to support multiple parallel tasks using up to 1024 Broadwell cores (32 nodes) or 1536 Skylake cores (32 nodes) or 1536 Cascade Lake cores (32 nodes) thanks to a non-blocking network. For larger tasks, the network has a 2:1 blocking factor. Even for tasks of several thousand cores, Cedar is a good option.


## Node characteristics

Cedar offers 100,400 CPU cores for computation and 1352 GPUs. TurboBoost is disabled on all nodes.

| Nodes | Cores | Available Memory | CPU                                         | Storage     | GPU                                      |
|-------|-------|--------------------|---------------------------------------------|-------------|-------------------------------------------|
| 256   | 32    | 125G or 128000M     | 2 x Intel E5-2683 v4 Broadwell @ 2.1GHz     | 2 x SSD 480G | -                                           |
| 256   | 32    | 250G or 257000M     | 2 x Intel E5-2683 v4 Broadwell @ 2.1GHz     | 2 x SSD 480G | -                                           |
| 40    | 32    | 502G or 515000M     | 2 x Intel E5-2683 v4 Broadwell @ 2.1GHz     | 2 x SSD 480G | -                                           |
| 16    | 32    | 1510G or 1547000M    | 2 x Intel E5-2683 v4 Broadwell @ 2.1GHz     | 2 x SSD 480G | -                                           |
| 6     | 32    | 4000G or 409600M    | 2 x AMD EPYC 7302 @ 3.0GHz                 | 2 x SSD 480G | -                                           |
| 2     | 40    | 6000G or 614400M    | 4 x Intel Gold 5215 Cascade Lake @ 2.5GHz   | 2 x SSD 480G | -                                           |
| 96    | 24    | 125G or 128000M     | 2 x Intel E5-2650 v4 Broadwell @ 2.2GHz     | 1 x SSD 800G | 4 x NVIDIA P100 Pascal (12G HBM2 memory) |
| 32    | 24    | 250G or 257000M     | 2 x Intel E5-2650 v4 Broadwell @ 2.2GHz     | 1 x SSD 800G | 4 x NVIDIA P100 Pascal (16G HBM2 memory) |
| 192   | 32    | 187G or 192000M     | 2 x Intel Silver 4216 Cascade Lake @ 2.1GHz | 1 x SSD 480G | 4 x NVIDIA V100 Volta (32G HBM2 memory)   |
| 608   | 48    | 187G or 192000M     | 2 x Intel Platinum 8160F Skylake @ 2.1GHz  | 2 x SSD 480G | -                                           |
| 768   | 48    | 187G or 192000M     | 2 x Intel Platinum 8260 Cascade Lake @ 2.4GHz | 2 x SSD 480G | -                                           |


Note that the amount of available memory is less than the rounded value suggested by the hardware configuration. For example, `base 128G` nodes actually have 128 GB of RAM, but a certain amount is permanently used by the kernel and the operating system. To avoid the time loss incurred by swapping or paging, the scheduler will never allocate a task whose requirements exceed the amount of available memory indicated in the table above.

All nodes have temporary local storage space. Compute nodes (except GPU nodes) have two 480GB SSDs for a total capacity of 960GB. GPU nodes have an 800GB or 480GB SSD. Use local storage on the node through the directory created for the task by the scheduler. See [Local storage on compute nodes](placeholder_link_to_local_storage).


### Select a node type

A number of 48-core nodes are reserved for tasks requiring entire nodes. No 32-core nodes are reserved for whole-node computations.

Tasks requiring less than 48 cores per node may therefore have to share nodes with other tasks.

Most applications can be run on Broadwell, Skylake or Cascade Lake nodes and the performance difference should not be significant compared to waiting times. We recommend that you do not specify the node type for your tasks. However, if it is necessary to request a particular type, use `--constraint=cascade`, `--constraint=skylake` or `--constraint=broadwell`. If you need an AVX512 node, use `--constraint=[skylake|cascade]`.


## Changes to the job submission and execution policy

Since April 17, 2019, tasks can no longer be executed in the `/home` file system. This change aims to reduce the load and improve response time in interactive mode in `/home`. If the message "Submitting jobs from directories residing in /home is not permitted" is displayed, transfer the files to your `/project` or `/scratch` directory and submit the task from the new location.


### Performance

The maximum theoretical double-precision performance is 6547 teraflops for CPUs plus 7434 teraflops for GPUs, for a total of nearly 14 petaflops.

The network topology is a composition of islands with a 2:1 blocking factor between each. Most islands have 32 nodes fully connected by a non-blocking Omni-Path fabric.

Most islands have 32 nodes:

*   16 islands of 32 Broadwell nodes each with 32 cores, or 1024 cores per island;
*   43 islands of 32 Skylake or Cascade Lake nodes each with 48 cores, or 1536 cores per island;
*   4 islands with 32 P100 GPU nodes;
*   6 islands with 32 V100 GPU nodes;
*   2 islands of 32 large memory Broadwell nodes each; of these 64 nodes, 40 are 0.5 TB, 16 are 1.5 TB, 6 are 4 TB and 2 are 6 TB.


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Cedar/fr&oldid=164581](https://docs.alliancecan.ca/mediawiki/index.php?title=Cedar/fr&oldid=164581)"
