# Cloud RAS Allocations

Any Digital Research Alliance of Canada user can access modest quantities of resources as soon as they have an Alliance account. The Rapid Access Service (RAS) allows users to experiment and to start working right away. Many research groups can meet their needs with the Rapid Access Service only. Users requiring larger resource quantities can apply to our annual Resource Allocation Competition (RAC). Primary Investigators (PIs) with a current RAC allocation are also able to request resources via RAS.

Using cloud resources, researchers can create cloud instances (also known as virtual machines or VMs). There are two options available for cloud resources:

## Compute Instances

These are instances that have a limited life-time (wall-time) and typically have constant high CPU requirements. They are sometimes referred to as batch instances. Users may need a large number of compute instances for production activities. Maximum wall-time for compute instances is one month. Upon reaching their life-time limit these instances will be scheduled for deactivation and their owners will be notified in order to ensure they clean up their instances and download any required data. Any grace period is subject to resources availability at that time.

## Persistent Instances

These are instances that are meant to run indefinitely and would include Web servers, database servers, etc. In general, these instances provide a persistent service and use less CPU power than compute instances.

## vGPU

Arbutus currently offers V100 GPUs in a single flavor (`g1-8gb-c4-22gb`). This flavor has 8GB GPU memory, 4 vCPUs and 22GB of memory. In the future, alternative GPU flavors will be available; researcher feedback on useful resource combinations for those new flavors is welcomed. For more information on setting up your VM to use vGPUs, see [Using cloud vGPUs](link-to-vgpu-doc).


## Cloud RAS Resources Limits

| Attributes             | Compute Instances | Persistent Instances |
|-------------------------|--------------------|----------------------|
| **May be requested by** | PIs only           | PIs only             |
| vCPUs (see VM flavours) | 80                 | 25                   |
| vGPUs                  | 1                  | 1                    |
| Instances              | 20                 | 10                   |
| Volumes                | 2                  | 10                   |
| Volume snapshots       | 2                  | 10                   |
| RAM (GB)               | 300                | 50                   |
| Floating IP            | 2                  | 2                    |
| Persistent storage (TB) | 10                 | 10                   |
| Object storage (TB)     | 10                 | 10                   |
| Shared filesystem storage (TB) | 10                 | 10                   |
| Default duration       | 1 year, with 1 month wall-time | 1 year (renewable)   |
| Default renewal        | April               | April                 |


## Requesting RAS

To request RAS, please [fill out this form](link-to-form).

## Notes

1. Users may request both a compute and persistent allocation to share a single project. Storage is shared between the two allocations and is limited to 10TB/PI per storage type. PIs may request a 1-year renewal of their cloud RAS allocations an unlimited number of times; however, allocations will be given based on available resources and are not guaranteed. Requests made after January 1 will expire March of the following year and therefore may be longer than 1 year. Allocation requests made between May-December will be less than 1 year. Renewals will take effect in April.

2. Currently only available at Arbutus.

3. This is a metadata quota and not a hard limit, users can request an increase beyond these values without a RAC request.

4. This is to align with the RAC allocation period of April-March.


**(Retrieved from "https://docs.alliancecan.ca/mediawiki/index.php?title=Cloud_RAS_Allocations&oldid=143400")**
