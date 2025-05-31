# tamIA: A Canadian AI Cluster

tamIA is a cluster dedicated to artificial intelligence for the Canadian scientific community. Located at Université Laval, tamIA is co-managed with Mila and Calcul Québec.  The cluster is named for the eastern chipmunk, a common species found in eastern North America. tamIA is part of PAICE, the Pan-Canadian AI Compute Environment.

**Availability:** March 31, 2025

**Login node:** tamia.alliancecan.ca

**Globus collection:** TamIA's Globus v5 Server

**Data transfer node (rsync, scp, sftp,...):** tamia.alliancecan.ca

**Portal:** To be announced


## Site-specific policies

* By policy, tamIA's compute nodes cannot access the internet. If you need an exception, contact technical support explaining your needs.
* `crontab` is not offered on tamIA.
* Each job should be at least one hour long (at least five minutes for test jobs). You can't have more than 1000 jobs (running and pending) at a time.
* The maximum duration of a task is one day (24 hours).
* Each task must use 4 GPUs, or 1 full node.


## Access

To access the cluster, each researcher must complete an access request in the [CCDB](link-to-ccdb-here). Access to the cluster may take up to one hour after completing the access request.

Eligible principal investigators are members of an AIP-type RAP (prefix `aip-`).

The procedure for sponsoring other researchers is as follows:

1. In the [CCDB home page](link-to-ccdb-home-page-here), go to the *Resource Allocation Projects* table.
2. Look for the RAPI of the `aip-` project and click on it to be redirected to the RAP management page.
3. At the bottom of the RAP management page, click on *Manage RAP memberships*.
4. To add a new member, go to *Add Members* and enter the CCRI of the user you want to add.


## Storage

* **HOME:** Lustre file system. Location of home directories, each with a small fixed quota. Use the *PROJECT* space for larger storage needs. Small per-user quota. There is currently no backup of the home directories (ETA Summer 2025).
* **SCRATCH:** Lustre file system. Large space for storing temporary files during computations. No backup system in place. Large quota per user. There is an automated purge of older files in this space.
* **PROJECT:** Lustre file system. This space is designed for sharing data among members of a research group and for storing large amounts of data. Large and adjustable per-group quota. There is currently no backup of the project directories (ETA Summer 2025).

For transferring data via Globus, use the endpoint specified at the top of this page. For tools like `rsync` and `scp`, use the login node.


## High-performance interconnect

The InfiniBand NVIDIA NDR network links together all the nodes of the cluster. Each H100 GPU is connected to a single NDR200 port through an NVIDIA ConnectX-7 HCA. Each GPU server has 4 NDR200 ports connected to the InfiniBand fabric.

The InfiniBand network is non-blocking for compute servers and is composed of two levels of switches in a fat-tree topology. Storage and management nodes are connected via four 400Gb/s connections to the network core.


## Node characteristics

| Nodes | Cores | Available Memory | CPU                                      | Storage      | GPU                                                                     |
|-------|-------|-----------------|-------------------------------------------|---------------|--------------------------------------------------------------------------|
| 42    | 48    | 512GB           | 2 x Intel Xeon Gold 6442Y 2,6 GHz, 24C | 1 x 7.68TB SSD | 4 x NVIDIA HGX H100 SXM 80GB HBM3 700W, connected via NVLink             |
| 4     | 64    | 512GB           | 2 x Intel Xeon Gold 6438M 2.2G, 32C/64T | 1 x 7.68TB SSD | None                                                                      |


## Software environments

`StdEnv/2023` is the standard environment on tamIA.


## Monitoring jobs

The portal is not yet available.

From the tamIA portal, you can monitor your jobs using CPUs and GPUs in real time or examine jobs that have run in the past. This can help you optimize resource usage and shorten wait time in the queue. You can monitor your usage of compute nodes, memory, and GPUs.

It is important that you use the allocated resources and correct your requests when compute resources are less used or not used at all. For example, if you request 4 cores (CPUs) but use only one, you should adjust the script file accordingly.
