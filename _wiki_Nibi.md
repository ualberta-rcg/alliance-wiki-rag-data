# Nibi Supercomputer

**Other languages:** English, français

**Availability:** Spring 2025

**Login node:** nibi.sharcnet.ca (for testing)

**Globus endpoint:** To be determined

**Data transfer node (rsync, scp, sftp,...):** Use login nodes

**Portal:** To be determined

Nibi, the Anishinaabemowin word for water, is the successor of Graham and a general-purpose cluster of 134,400 CPU cores and 288 H100 NVIDIA GPUs.  It was built by [Hypertec](https://www.hypertec.com/), hosted and operated by [SHARCNET](https://www.sharcnet.ca/) at the University of Waterloo.  It is expected that Nibi will come online by July 1, 2025.

## Storage

**Parallel storage:** 25 petabytes, all SSD from [VAST Data](https://www.vastdata.com/) for `/home`, `/project`, and `/scratch`. This is the same storage as used by miniGraham.

## Interconnect Fabric

*   Nokia 200/400G ethernet
*   200 Gbit/s network bandwidth for CPU nodes.
*   400 Gbit/s non-blocking network bandwidth between all Nvidia GPU nodes.
*   200 Gbit/s network bandwidth between all AMD GPU nodes.
*   24x100 Gbit/s connection to the VAST storage nodes.
*   2:1 blocking at 400 Gbit/s uplinks for all compute nodes.

## Node Characteristics

| Nodes | Cores | Available Memory | CPU                                      | GPU                                                |
|-------|-------|--------------------|-------------------------------------------|------------------------------------------------------|
| 700   | 192   | 768GB DDR5          | 2 x Intel 6972P @ 2.4 GHz, 384MB cache L3 | 10                                                   |
| 192   | 6TB   | DDR5                | 2 x Intel 6972P @ 2.4 GHz, 384MB cache L3 | 36                                                   |
| 112   | 2TB   | DDR5                | 1 x Intel 8570 @ 2.1 GHz, 300MB cache L3 | 8 x Nvidia H100 SXM (80 GB memory)                   |
| 96    | 512GB |                     | 4 x AMD MI300A @ 2.1GHz                   | 4 x AMD CDNA 3 (unified memory: 512 GB HBM3 total) |


**(Retrieved from "https://docs.alliancecan.ca/mediawiki/index.php?title=Nibi&oldid=177225")**
