# Killarney

Killarney is a cluster that meets the needs of the Canadian scientific community in artificial intelligence. It is located at the University of Toronto and managed by the Vector Institute and SciNet. Its name is reminiscent of Killarney Provincial Park, located near Georgian Bay, Ontario.

Killarney is part of ECPIA, the Pan-Canadian Computing Environment for Artificial Intelligence.


## Hardware

### Performance

| Nodes | Model             | CPU                     | Cores | System Memory | GPU per Node | Total GPUs |
|-------|----------------------|--------------------------|-------|----------------|---------------|------------|
| 168   | Dell 750xa          | 2 x Intel Xeon Gold 6338 | 64    | 512 GB         | 4 x NVIDIA L40s 48GB | 672        |
| 10    | Dell XE9680         | 2 x Intel Xeon Gold 6442Y | 48    | 2048 GB        | 8 x NVIDIA H100 SXM 80GB | 80         |


## Storage

The storage system is a NVME VastData platform with a usable capacity of 1.7PB.

* **/home**: Total volume xxxTB.  Location of `/home` directories. Fixed quota per directory. Not allocated via the rapid access service or the resource allocation competition; requests for more space are directed to `/project`. Daily backup.
* **/scratch**: Total volume xPB. High-performance parallel file system designed for active or temporary storage. Not allocated. Fixed quota per user. Inactive data is purged.
* **/project**: Total volume xPB. Persistent external storage. Not designed for parallel I/O tasks (use `/scratch` space instead). Fixed quota per project. Daily backup.


## Networking

* Standard compute nodes: Infiniband HDR100, 100Gbps throughput.
* Performance compute nodes: 2 x HDR 200, aggregated throughput of 400Gbps.


## Scheduling

The Slurm scheduler runs user-submitted jobs. Basic Slurm commands are similar to those for other national systems.


## Software

Modular software stack. Standard Alliance software stack and cluster-specific software.
