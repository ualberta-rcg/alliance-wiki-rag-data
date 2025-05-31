# Killarney Cluster

**Other languages:** English, français

**Availability:** TBA

**Login node:** TBA

**Globus endpoint:** TBA

**System Status Page:** TBA


Killarney is a cluster dedicated to the needs of the Canadian scientific Artificial Intelligence community.  Located at the University of Toronto, it is managed by the Vector Institute and SciNet. It is named after the Killarney Ontario Provincial Park, located near Georgian Bay. This cluster is part of the Pan-Canadian AI Compute Environment (PAICE).


## Killarney Hardware Specifications

| Performance Tier | Nodes | Model       | CPU                     | Cores | System Memory | GPUs per node | Total GPUs |
|-----------------|-------|-------------|--------------------------|-------|----------------|----------------|-------------|
| Standard Compute | 168   | Dell 750xa  | 2 x Intel Xeon Gold 6338 | 64    | 512 GB         | 4 x NVIDIA L40s 48GB | 672         |
| Performance Compute | 10    | Dell XE9680 | 2 x Intel Xeon Gold 6442Y | 48    | 2048 GB        | 8 x NVIDIA H100 SXM 80GB | 80          |


## Storage System

Killarney's storage system is an all-NVME VastData platform with a total usable capacity of 1.7PB.

* **Home space:** xxxTB total volume. Location of `/home` directories. Each `/home` directory has a small fixed quota. Not allocated via RAS or RAC. Larger requests go to the `/project` space. Has daily backup.
* **Scratch space:** xPB total volume. Parallel high-performance filesystem. For active or temporary (scratch) storage. Not allocated. Large fixed quota per user. Inactive data will be purged.
* **Project space:** xPB total volume. External persistent storage. Not designed for parallel I/O workloads. Use `/scratch` space instead. Large adjustable quota per project. Has daily backup.


## Network Interconnects

Standard Compute nodes are interconnected with Infiniband HDR100 for 100Gbps throughput, while Performance Compute nodes are connected with 2 x HDR 200 for 400Gbps aggregate throughput.


## Scheduling

The Killarney cluster uses the Slurm scheduler to run user workloads. The basic scheduling commands are similar to the other national systems.


## Software

Module-based software stack. Both the standard Alliance software stack as well as cluster-specific software.


**(Retrieved from "https://docs.alliancecan.ca/mediawiki/index.php?title=Killarney&oldid=174874")**
