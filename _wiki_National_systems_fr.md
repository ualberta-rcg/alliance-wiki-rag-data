# National Systems

## Compute Clusters

Most of our clusters are general purpose and designed to run many types of jobs. They have nodes with different characteristics, classified into three groups:

*   **Base nodes:** Typically around 4GB of memory per core.
*   **Large-memory nodes:** Typically more than 8GB per core.
*   **GPU nodes:**  Have Graphics Processing Units (GPUs).

The Niagara cluster is designed to support massively parallel jobs (requiring more than 1000 CPU cores across multiple nodes), but it also accepts jobs that only require a single node. Its nodes have uniform characteristics and its network is optimized for running demanding jobs.

All clusters have high-performance storage.  Click on a cluster name in the table below for details on the number of nodes available, CPU and GPU counts and models, memory, and storage.

### List of Compute Clusters

| Cluster   | Type                     | Subsystems             | Status             |
| --------- | ------------------------ | ----------------------- | ------------------ |
| Béluga    | General Purpose          | beluga-compute, beluga-gpu, beluga-storage | In Production       |
| Cedar     | General Purpose          | cedar-compute, cedar-gpu, cedar-storage     | In Production       |
| Fir       | General Purpose          | fir-compute, fir-gpu, fir-storage           | Installing          |
| Graham    | General Purpose          | graham-compute, graham-gpu, graham-storage   | In Production       |
| Narval    | General Purpose          | narval-compute, narval-gpu, narval-storage   | In Production       |
| Niagara   | Massively Parallel Jobs | niagara-compute, niagara-storage, hpss-storage | In Production       |
| Nibi      | General Purpose          | nibi-compute, nibi-storage, nibi-storage     | Installing          |
| Rorqual   | General Purpose          | rorqual-compute, rorqual-gpu, rorqual-storage | Installing          |
| Trillium  | Massively Parallel Jobs | trillium-compute, trillium-gpu, trillium-storage | Installing          |


## Cloud (IaaS)

Our cloud service is offered as an Infrastructure as a Service (IaaS) model based on OpenStack.

### Cloud

| Cloud          | Subsystems                      | Description                                                                                                       | Status             |
| --------------- | -------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------ |
| Arbutus Cloud   | arbutus-compute-cloud, arbutus-persistent-cloud, arbutus-dcache                                         | vCPU, VGPU, RAM, Ephemeral local disk, Volume and snapshot storage, Shared filesystem storage, Object storage, Floating IP addresses, dCache storage | In Production       |
| Béluga Cloud    | beluga-compute-cloud, beluga-persistent-cloud                                                             | vCPU, RAM, Ephemeral local disk, Volume and snapshot storage, Floating IP addresses                             | In Production       |
| Cedar Cloud     | cedar-persistent-cloud, cedar-compute-cloud                                                              | vCPU, RAM, Ephemeral local disk, Volume and snapshot storage, Floating IP addresses                             | In Production       |
| Graham Cloud    | graham-persistent-cloud                                                                                   | vCPU, RAM, Ephemeral local disk, Volume and snapshot storage, Floating IP addresses                             | In Production       |


## EIPIA Clusters

The Pan-Canadian Artificial Intelligence (EIPIA) clusters are systems dedicated to the emerging needs of the Canadian artificial intelligence research community.

| Name       | Institute      | Status             |
| ---------- | --------------- | ------------------ |
| TamIA      | Mila            | Installing          |
| Killarney  | Vector Institute | Installing          |
| Vulcan     | Amii            | Installing          |


**(Retrieved from "https://docs.alliancecan.ca/mediawiki/index.php?title=National_systems/fr&oldid=177204")**
