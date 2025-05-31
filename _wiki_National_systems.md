# National Systems

## Compute Clusters

A general-purpose cluster is designed to support a wide variety of types of jobs, and is composed of a mixture of different nodes. We broadly classify the nodes as:

*   **base** nodes, containing typically about 4GB of memory per core;
*   **large-memory** nodes, containing typically more than 8GB memory per core;
*   **GPU** nodes, which contain graphic processing units.

The large parallel cluster Niagara is designed to support multi-node parallel jobs requiring more than 1000 CPU cores, although jobs as small as a single node are also supported there. Niagara is composed of nodes of a uniform design, with an interconnect optimized for large jobs.

All clusters have large, high-performance storage attached. For details about storage, memory, CPU model and count, GPU model and count, and the number of nodes at each site, please click on the cluster name in the table below.

### List of Compute Clusters

| Name and link | Type             | Sub-systems           | Status             |
| -------------- | ---------------- | --------------------- | ------------------ |
| Béluga         | General-purpose  | beluga-compute, beluga-gpu, beluga-storage | In production       |
| Cedar          | General-purpose  | cedar-compute, cedar-gpu, cedar-storage     | In production       |
| Fir            | General-purpose  | fir-compute, fir-gpu, fir-storage           | Installation in progress |
| Graham         | General-purpose  | graham-compute, graham-gpu, graham-storage   | In production       |
| Narval         | General-purpose  | narval-compute, narval-gpu, narval-storage   | In production       |
| Niagara        | Large parallel   | niagara-compute, niagara-storage, hpss-storage | In production       |
| Nibi           | General-purpose  | nibi-compute, nibi-storage, nibi-storage     | Installation in progress |
| Rorqual        | General-purpose  | rorqual-compute, rorqual-gpu, rorqual-storage | Installation in progress |
| Trillium       | Large parallel   | trillium-compute, trillium-gpu, trillium-storage | Installation in progress |


## Cloud - Infrastructure as a Service

Our cloud systems are offering an Infrastructure as a Service (IaaS) based on OpenStack.

| Name and link      | Sub-systems                               | Description                                                                                                                        | Status             |
| ------------------ | ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| Arbutus cloud      | arbutus-compute-cloud, arbutus-persistent-cloud, arbutus-dcache | VCPU, VGPU, RAM, Local ephemeral disk, Volume and snapshot storage, Shared filesystem storage (backed up), Object storage, Floating IPs, dCache storage | In production       |
| Béluga cloud       | beluga-compute-cloud, beluga-persistent-cloud | VCPU, RAM, Local ephemeral disk, Volume and snapshot storage, Floating IPs                                                        | In production       |
| Cedar cloud        | cedar-persistent-cloud, cedar-compute-cloud | VCPU, RAM, Local ephemeral disk, Volume and snapshot storage, Floating IPs                                                        | In production       |
| Graham cloud       | graham-persistent-cloud                    | VCPU, RAM, Local ephemeral disk, Volume and snapshot storage, Floating IPs                                                        | In production       |


## PAICE Clusters

Pan-Canadian AI Compute Environment (PAICE) clusters are systems dedicated to the current and emerging AI needs of Canada’s research community.

| Name and link | Institute        | Status             |
| -------------- | ---------------- | ------------------ |
| TamIA          | Mila             | Installation in progress |
| Killarney      | Vector Institute | Installation in progress |
| Vulcan         | Amii             | Installation in progress |


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=National_systems&oldid=177172](https://docs.alliancecan.ca/mediawiki/index.php?title=National_systems&oldid=177172)"
