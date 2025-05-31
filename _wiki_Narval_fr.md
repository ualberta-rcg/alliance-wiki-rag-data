# Narval Cluster Documentation

**Availability:** Since October 2021

**Login Node:** narval.alliancecan.ca

**Globus Collection:** Compute Canada - Narval

**Copy Node (rsync, scp, sftp, ...):** narval.alliancecan.ca

**Portal:** https://portail.narval.calculquebec.ca/

Narval is a heterogeneous and versatile cluster designed for a wide variety of small and medium-sized scientific computations. Narval is located at the École de technologie supérieure. Its name is a reference to the narwhal, a marine mammal that has sometimes been observed in the waters of the St. Lawrence River.


## Specific Features

Our policy is that Narval compute nodes do not have internet access. For exceptions, please contact technical support explaining your needs and reasons. Note that the `crontab` tool is not available.

Each task should be at least one hour long (at least five minutes for test tasks), and you cannot have more than 1000 tasks (running and pending) at a time. The maximum duration of a task is 7 days (168 hours).


## Storage

**HOME:** Lustre file system, 40 TB of total space. This space is small and cannot be expanded; you will need to use your `project` space for large storage needs. Small, fixed quotas per user. Automatic backup once a day.

**SCRATCH:** Lustre file system, 5.5 PB of total space. Large space for storing temporary files during calculations. No automatic backup system. Large, fixed quotas per user. Automatic purge of old files in this space.

**PROJECT:** Lustre file system, 19 PB of total space. This space is designed for data sharing between group members and for storing large amounts of data. Large, adjustable quotas per project. Automatic backup once a day.

At the beginning of this page, a table shows several connection addresses. For data transfers via Globus, use the Globus Endpoint. However, for tools like `rsync` and `scp`, use the Copy Node address.


## High-Performance Networking

The Mellanox HDR InfiniBand network connects all nodes in the cluster. Each 40-port HDR switch (200 Gb/s) allows connecting up to 66 nodes in HDR100 (100 Gb/s) with 33 HDR links divided into two (2) by special cables. The seven (7) remaining HDR links are used to connect the switch of a cabinet to each of the seven (7) HDR switches of the central InfiniBand network. The node islands are therefore connected with a maximum blocking factor of 33:7 (4.7:1). However, the storage servers are connected with a significantly lower blocking factor for maximum performance.

In practice, Narval cabinets contain islands of 48 or 56 regular CPU nodes. It is therefore possible to run parallel tasks using up to 3584 cores and non-blocking networking. For larger or more fragmented tasks on the network, the blocking factor is 4.7:1.  The interconnection remains high-performance nonetheless.


## Node Characteristics

| Nodes | Cores | Available Memory | CPU                                                              | Storage      | GPU                                                                  |
|-------|-------|--------------------|--------------------------------------------------------------------|--------------|-----------------------------------------------------------------------|
| 1145  | 64    | 249G or 255000M     | 2 x AMD EPYC 7532 (Zen 2) @ 2.40 GHz, 256M L3 cache             | 1 x 960G SSD |                                                                       |
| 33    |       | 2009G or 2057500M | 3                                                                   |             |                                                                       |
|       |       | 4000G or 4096000M |                                                                    |             |                                                                       |
| 159   | 48    | 498G or 510000M     | 2 x AMD EPYC 7413 (Zen 3) @ 2.65 GHz, 128M L3 cache             | 1 x 3.84T SSD | 4 x NVidia A100SXM4 (40G memory), connected via NVLink              |


## AMD Processor Specifics

### Supported Instruction Sets

The Narval cluster is equipped with 2nd and 3rd generation AMD EPYC processors that support AVX2 instructions. This instruction set is the same as that of the Intel processors found on the Beluga, Cedar, Graham, and Niagara nodes.

However, Narval does not support AVX512 instructions, unlike Beluga and Niagara nodes and some Cedar and Graham nodes. AVX2 is supported where nodes have Broadwell-type CPUs, while both instruction sets (AVX2 and AVX512) are supported where CPUs are Skylake or Cascade Lake type.  Therefore, an application compiled on a node with a Broadwell CPU from Cedar or Graham, including their login nodes, can be executed on Narval, but cannot be executed if compiled on Beluga or Niagara, or on a Skylake or Cascade Lake node from Cedar or Graham. In the latter case, the application will need to be recompiled (see Intel Compilers below).


### Intel Compilers

Intel compilers can compile applications for Narval's AMD processors very well, limiting themselves to AVX2 and older instruction sets. To do this, use the `-march=core-avx2` option of the Intel compiler, which allows obtaining executables that are compatible with both Intel and AMD processors.

However, if you have compiled code on a system using Intel processors and used one or more `-xXXXX` options, such as `-xCORE-AVX2`, the compiled applications will not work on Narval, because Intel compilers add additional instructions to verify that the processor used is an Intel product. On Narval, the `-xHOST` and `-march=native` options are equivalent to `-march=pentium` (the old Pentium from 1993) and should not be used.


### Available Software Environments

The standard software environment `StdEnv/2023` is the default environment on Narval. Older versions (2016 and 2018) have been intentionally blocked. If you need software that is only available on an older version of the standard environment, please send a request to our technical support.


### BLAS and LAPACK Libraries

The Intel MKL library works on AMD processors, but it is not optimal. We now favor the use of FlexiBLAS. For more details, see the BLAS and LAPACK page.


## Task Monitoring

From the portal, you can monitor your CPU and GPU compute tasks in real time or past tasks to maximize resource utilization and reduce your queue wait times.

You will be able to visualize for a task:

* The use of computing cores;
* The memory used;
* The use of GPUs.

It is important to use the allocated resources and correct your requests when computing resources are underutilized or not used at all. For example, if you request four cores (CPU) but only use one, you should adjust your submission file accordingly.

Retrieved from "https://docs.alliancecan.ca/mediawiki/index.php?title=Narval/fr&oldid=164853"
