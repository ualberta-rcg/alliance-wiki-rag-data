# Important Update to Our Advanced Computing Infrastructure

This major update to our advanced computing infrastructure, starting in the winter of 2024-2025, will improve our high-performance computing and cloud services to support research in Canada.  Most new systems are scheduled to be commissioned by summer 2025. The content of this page will be updated as information becomes available.

Nearly 80% of our current equipment, which is nearing the end of its life, will be replaced. The new hardware will offer faster processing speeds, greater storage capacity, and improved reliability.


## New Systems

| New System | Replaced System | Documentation |
|---|---|---|
| Arbutus (cloud) |  (no change to this virtual infrastructure) | [see this page](link-to-page-needed) |
| Rorqual | Béluga | [see this page](link-to-page-needed) |
| Fir | Cedar | [see this page](link-to-page-needed) |
| Trillium | Niagara & Mist | [see this page](link-to-page-needed) |
| Nibi | Graham | [see this page](link-to-page-needed) |


## System Capacity, Downgrades, and Outages

During the installation and transition to the new systems, we may need to suspend or reduce services due to power or space constraints. Please take these possibilities into account when planning your research program, thesis or dissertation defenses, etc.

[Click here for a list of completed work](link-to-page-needed).

| Start | End | Status | System | Type | Description |
|---|---|---|---|---|---|
| June 6, 2025, 9 AM (EDT) | June 10, 2025, 12 PM (EDT) (4 days) | In preparation | Béluga, Narval, Juno (other than HD zone) | Service Outage | Scheduled electrical maintenance requires us to shut down the Béluga and Narval compute nodes from 9 AM (noon) on June 6th until 12 PM (noon) on June 10th, 2025 (EDT). Juno cloud instances that are not in the high-availability zone will also be shut down. Jobs scheduled to finish after 9 AM on June 6th will remain in the queue until the clusters return to service. |
|  |  |  |  | Short interruptions for network and storage maintenance: | Network maintenance work may cause short interruptions of access to Béluga cloud instances and Juno cloud instances that are not in the high-availability zone. Béluga and Narval storage systems will remain accessible via Globus and login nodes; however, network and storage maintenance work may cause intermittent access interruptions. |
| January 22, 2025 | Ongoing |  | Cedar (70%) | Service Downgrade | Cluster capacity will be reduced to approximately 70% from January 22nd until the Fir cluster comes online in summer 2025. |
| February 25, 2025 | Ongoing |  | Graham (25%) | Service Downgrade | Update March 21: The Graham cluster is now operating at reduced capacity. The Graham cloud is operating as usual. |
| January 6, 2025 | Ongoing |  | Niagara (50%), Mist (35%) | Service Downgrade | Computing capacity will be reduced to 50% for Niagara and 35% for Mist until the commissioning of Trillium, scheduled for summer 2025. Mist was down for a few hours on January 6, 2025. |


## Resource Allocation Competition

The resource allocation competition will be affected by this transition, but the application procedure remains unchanged.

Allocations for 2024-2025 will remain in effect on the clusters being replaced as long as those clusters are in service. Allocations for 2025-2026 will all be available once all the new systems are in service.

If you hold both 2024 and 2025 allocations, there will be a period when these allocations will not be available since most of the replaced clusters will be out of service before all the new systems are commissioned. However, you will be able to use your default allocation (`def-xxxxxx`) on each new system as soon as it is commissioned, but the 2025 allocations will only be available once all the new systems are commissioned.


## Training Tools

| Title | Organization | Presented by | Date | Description | Target Audience | Format | Registration |
|---|---|---|---|---|---|---|---|
| Workflow Hacks for Large Datasets in HPC | Simon Fraser University / West DRI | Alex Razoumov | Tuesday, May 20, 2025, 10 AM PT | Over the years, we have prepared webinars on tools that facilitate workflows with large datasets. We will present some of these tools here.  • In-situ visualization: for interactive rendering of large in-memory vectors, without the need to save them to disk. • Lossy 3D data compression: to facilitate storage and archiving by reducing the size of 3D datasets by about 100X, without visible artifacts. • Distributed storage: for better management of a lot of data located in several different places. • DAR (Disk ARchiver): a modern and efficient alternative to TAR that offers indexing, differential archives, and faster extraction. Recordings and materials for previous webinars are available (free of charge) at [https://training.westdri.ca](https://training.westdri.ca). | Users with large datasets | Webinar; recordings and materials from past webinars are available free of charge at [https://training.westdri.ca](https://training.westdri.ca) |  |
| Mastering GPU Efficiency | SHARCNET | Sergey Mashchenko | Anytime | This self-paced online course provides basic training on using GPUs on our national systems. Modern GPUs (such as NVIDIA A100 and H100) are massively parallel and very expensive resources. Most GPU tasks are unable to use these GPUs efficiently, either because the problem size is too small to saturate the GPU, or because of the intermittent (bursty) usage pattern of the GPU. You will learn how to measure the GPU utilization by your tasks to use both NVIDIA technologies - MPS (Multi-Process Service) and MIG (Multi-Instance GPU) to improve GPU utilization. | Potential users of upgraded systems | 1-hour online course with certificate | Access the course here (an Alliance account is required) |
| Introduction to the Fir cluster | Simon Fraser University / West DRI | Alex Razoumov | September 2025 (date postponed) | The new Fir cluster at Simon Fraser University is expected to be operational in the summer of 2025. We will present an overview of the cluster and its hardware; the different file systems and their recommended use; the job submission policies; and best practices on using the cluster. | Fir cluster users | Webinar | Registration details to come |
| Survival guide for the upcoming GPU upgrades | SHARCNET | Sergey Mashchenko | ONLINE | Our national systems will undergo major upgrades in the coming months. In particular, the old GPUs (P100, V100) will be replaced by the new NVIDIA H100 GPUs. The total computing power of the GPUs will increase by a factor of 3.5, but the number of GPUs will decrease considerably, from 3200 to 2100. This will pose a significant challenge, as the usual practice of using a whole GPU for each process or MPI rank will no longer be possible in most cases. Fortunately, NVIDIA offers two powerful technologies to mitigate this situation: MPS (Multi-Process Service) and MIG (Multi-Instance GPU). We will discuss these two technologies and how they can be used on our clusters. We will see how to determine the approach that will work best for a particular code and a demonstration will be performed at the end. | Potential users of upgraded systems, or those needing to use a significant amount of H100 resources (e.g., multiple GPUs at a time and/or for more than 24 hours of runtime) | Video and slides (duration, 1 hour) (presentation given November 20, 2024 from 12 PM to 1 PM) |  |


## Frequently Asked Questions

**Will my data be migrated to the new system?**

Data migration is the responsibility of each of the national host sites; you will receive information on the actions to take.

**Will my files be deleted if the data center hosting my system closes during the transition?**

No, your files will not be deleted. During the renewal activities, each national host site will migrate the `/project` and `/home` data from the existing storage system to the new storage system when it is installed. These migrations generally occur during service outages, but the specific details may vary by site. Each national host site will inform you of any action that may affect your work.

In addition, tape systems for backups and `/nearline` data are not being replaced, so backups and `/nearline` data will remain unchanged.

For other technical questions, contact [our technical support](link-to-support-needed).

**Are service outages predictable?**

Each of the national host sites manages the service outages that will be required during installation and transition; they will be reported on [our system status webpage](link-to-status-page-needed). This wiki page will be updated as information becomes available, and you will periodically receive email notices and updates.

**Who can answer my questions about the transition?**

[Technical support](link-to-support-needed) will try to inform you, but they may not yet know the information.

**Are the new systems compatible with my tasks and applications?**

Generally, yes. Some applications may need to be recompiled or reconfigured for the new CPUs and GPUs. You will receive information as the transition progresses.

**Will the software on existing systems still be available?**

Yes, our [standard software environment](link-to-software-environment-needed) will be available on the new systems.

**Will commercially licensed software be migrated to the new systems?**

Yes. As much as possible, you will have the same access for this type of application (Gaussian, AMS/ADF, etc.). Suppliers may change the terms and conditions, but the risk is low. We will inform you of any cases that may arise.

**Will service outages be staggered?**

We will do everything possible to limit overlapping service outages, but as we are very constrained by delivery schedules and funding deadlines, there will likely be periods when several of our systems are offline simultaneously. We will inform you as soon as possible.

**Is it possible to buy the equipment that will be removed from the infrastructure?**

Most of the equipment is owned by the host institutions, which dispose of it according to their own standards. Generally, the equipment is sent for recycling. Contact the host institution to see if there is a possibility of obtaining some.


**(Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Infrastructure_renewal/fr&oldid=178365](https://docs.alliancecan.ca/mediawiki/index.php?title=Infrastructure_renewal/fr&oldid=178365)")**
