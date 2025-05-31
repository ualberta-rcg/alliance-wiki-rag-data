# tamIA

**Other languages:** English, français

**Availability:** March 31, 2025

**Login node:** tamia.alliancecan.ca

**Globus Collection:** TamIA's Globus v5 Server

**Copy node (rsync, scp, sftp, ...):** tamia.alliancecan.ca

**Portal:** To be announced

tamIA is a cluster dedicated to the needs of the Canadian scientific community in artificial intelligence. tamIA is located at Université Laval and is co-managed with Mila and Calcul Québec. Its name refers to the tamia, a rodent present in North America. This cluster is part of the Pan-Canadian AI Computing Environment (PCAICE).


## Particularities

Our policy is that tamIA compute nodes do not have internet access. For exceptions, please contact technical support explaining your needs and why.

Note that the `crontab` tool is not available.

Each task should be at least one hour long (at least five minutes for test tasks), and you cannot have more than 1000 tasks (running and pending) at a time.

The maximum duration of a task is one day (24 hours).

Each task must use 4 GPUs, or one full node.


## Access

To access the compute cluster, each researcher must complete an access request in the CCDB.  Effective access to the cluster may take up to one hour after completing the access request.

Eligible Principal Investigators are members of an AIP-type RAP (prefix `aip-`).

The procedure for sponsoring other researchers is as follows:

1. On the CCDB homepage, consult the "Project with resource allocation" table.
2. Search for the RAP of the `aip-` project and click on it to be redirected to the RAP management page.
3. At the bottom of the RAP management page, click on "Manage project membership".
4. In the "Add members" section, enter the CCRI of the member to add.


## Storage

**HOME:** Lustre file system. This space is small and cannot be expanded: you will need to use your `project` space for large storage needs. Small, fixed quotas per user. There is currently no automatic backup (planned for summer 2025).

**SCRATCH:** Lustre file system. Large space to store temporary files during calculations. No automatic backup system. Large, fixed quotas per user. There is automatic purging of old files in this space.

**PROJECT:** Lustre file system. This space is designed for data sharing between group members and for storing large amounts of data. Large, adjustable quotas per project. There is an automatic daily backup.

At the very beginning of this page, a table shows several connection addresses. For data transfers via Globus, use the Globus Endpoint. However, for tools like `rsync` and `scp`, use the Copy Node address.


## High-Performance Networking

The Nvidia NDR InfiniBand network connects all nodes in the cluster. Each H100 GPU is connected to an NDR200 port via a Nvidia ConnectX-7 card. Each server therefore has 4 NDR200 ports connected to the Infiniband fabric.

The Infiniband network is non-blocking for compute servers and consists of 2 layers of switches arranged in a "fat-tree" topology. Storage and management nodes are connected via 4 400Gb/s connections to the heart of the network.


## Node Characteristics

| Nodes | Cores | Available Memory | CPU                                         | Storage     | GPU                                                                     |
|-------|-------|-----------------|---------------------------------------------|-------------|--------------------------------------------------------------------------|
| 42    | 48    | 512GB           | 2 x Intel Xeon Gold 6442Y 2.6 GHz, 24C     | 1 x 7.68TB SSD | 4 x NVIDIA HGX H100 SXM 80GB HBM3 700W, connected via NVLink             |
| 4     | 64    | 512GB           | 2 x Intel Xeon Gold 6438M 2.2G, 32C/64T | 1 x 7.68TB SSD | None                                                                      |


## Available Software Environments

The standard software environment `StdEnv/2023` is the default environment on tamIA.


## Task Monitoring

The portal is not yet available.

From the portal, you will be able to monitor your GPU and CPU compute tasks in real time or past tasks to maximize resource utilization and reduce your queue wait times.  You will be able to visualize, for a task:

* CPU core usage
* Memory used
* GPU usage

It is important to use the allocated resources and to correct your requests when computing resources are underutilized or not used at all. For example, if you request four cores (CPU) but only use one, you must adjust your submission file accordingly.

Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=TamIA&oldid=177751](https://docs.alliancecan.ca/mediawiki/index.php?title=TamIA&oldid=177751)"
