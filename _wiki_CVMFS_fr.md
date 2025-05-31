# CVMFS

This page is a translated version of the page [CVMFS](https://docs.alliancecan.ca/mediawiki/index.php?title=CVMFS&oldid=151027) and the translation is 100% complete.

Other languages:

* [English](https://docs.alliancecan.ca/mediawiki/index.php?title=CVMFS&oldid=151027)
* français


We use CVMFS (CERN Virtual Machine File System) to distribute software, data, and other content. For more information, see the [website](https://cvmfs.readthedocs.io/en/latest/) and the [documentation section](https://cvmfs.readthedocs.io/en/latest/doc/). To learn how to configure a CVMFS client, see our wiki page [Accessing CVMFS](https://docs.alliancecan.ca/mediawiki/index.php?title=Acc%C3%A8s_%C3%A0_CVMFS&oldid=151026).


## Introduction

CVMFS is a read-only distributed software distribution system implemented as a user-space POSIX file system (FUSE) using HTTP transport. It was originally developed for the Large Hadron Collider experiments at CERN to provide software to virtual machines and replace various shared software installation areas and package management systems across numerous computing sites. Designed to provide software in a fast, scalable, and reliable manner, its use has rapidly expanded in recent years to include dozens of projects, ~10<sup>10</sup> files and directories, ~10<sup>2</sup> computing sites, and ~10<sup>5</sup> clients worldwide. The [CernVM Monitor](https://cernvm-monitor.web.cern.ch/) shows several research groups using CVMFS and the sites of the strata that replicate their repositories.


## Description

* A single copy of the software needs to be maintained and can be propagated and used across multiple sites. Commonly used software can be installed on CVMFS to minimize remote software management.
* Software applications and their prerequisites can be run from CVMFS, eliminating any requirements on the Linux distribution type or version level of a client node.
* The project software stack and operating system can be decoupled. In the particular case of the cloud, this allows accessing the software within a virtual machine without being inside the virtual machine image, allowing images and software to be updated and distributed separately.
* Content versioning is done via repository catalog revisions. Updates are committed in transactions and can be rolled back to a previous state.
* Updates are propagated to clients automatically and atomically.
* Clients can see historical versions of the repository content.
* Files are retrieved using standard HTTP. Client nodes do not require opening ports or firewalls.
* Fault tolerance and reliability are achieved using multiple redundant proxy servers and strata servers. Clients seamlessly switch to the next available proxy or server.
* Hierarchical caching makes the CVMFS model highly scalable and robust, and minimizes network traffic. There can be multiple levels in the content distribution and caching hierarchy:
    * Strata 0 contains the master copy of the repository.
    * Multiple strata 1 servers replicate the repository content from strata 0.
    * HTTP proxy servers cache network requests from clients to strata 1 servers.
    * The CVMFS client downloads files on demand into the local client cache(s).
    * Two levels of local cache can be used, e.g., a fast SSD cache and a large HDD cache. A cluster file system can also be used as a shared cache for all nodes.
* CVMFS clients have read-only access to the file system.
* Using Merkle trees and content-addressable storage, and encoding metadata in catalogs, all metadata is treated as data, is virtually all immutable, and lends itself perfectly to caching.
* Metadata storage and operations scale using nested catalogs, allowing metadata query resolution to be performed locally by the client.
* File integrity and authenticity are verified using signed cryptographic hashes, preventing data corruption or tampering.
* On the server-side, automatic deduplication and compression minimize storage usage. On the client-side, file segmentation and on-demand access minimize storage usage.
* Versatile configurations can be deployed by writing authorization handlers or cache extensions to interact with external authorization or storage providers.


## References

* 2018-01-31 [Compute Canada Software Installation and Distribution](https://www.computecanada.ca/), workshop (2018)
* 2019-06-03 [CVMFS at Compute Canada](https://www.computecanada.ca/), workshop (2019)
* 2019-06-20 [Providing A Unified User Environment for Canada’s National Advanced Computing Centers](https://www.canheit.ca/), CANHEIT (2019)
* 2019-07-28 [Providing a Unified Software Environment for Canada’s National Advanced Computing Centers](https://dl.acm.org/doi/10.1145/3341282.3341297), Practice and Experience in Advanced Research Computing (2019)
    * [PDF version](https://arxiv.org/pdf/1907.09269.pdf)
* 2019-08-01 [Providing a Unified Software Environment for Canada’s National Advanced Computing Centers](https://dl.acm.org/doi/10.1145/3367281.3370024), PEARC (2019)
* 2020-09-24 [Distributing software across campuses and the world with CVMFS](https://bcnet.ca/), BCNET Connect (2020)
* 2021-01-26 [CVMFS Tutorial](https://easybuild.readthedocs.io/en/latest/), EasyBuild User Meeting (2021)
    * [Slides](https://easybuild.readthedocs.io/en/latest/_static/presentations/2021-01-26-CVMFS-Tutorial.pdf)
* [Unlimited scientific libraries and applications in Kubernetes, instantly!](https://towardsdatascience.com/unlimited-scientific-libraries-and-applications-in-kubernetes-instantly-a7a60698667a), article in Towards Data Science (2021-09-27)
    * Demonstrates Compute Canada's approach to distributing research applications (the deployment described is used for a single demonstration cluster and uses CephFS rather than CVMFS)
* 2022-02-16 [EESSI: A cross-platform ready-to-use optimised scientific software stack](https://onlinelibrary.wiley.com/doi/full/10.1002/spe.3014), Journal of Software: Practice and Experience (2022)
    * Demonstrates Compute Canada's approach to distributing software to the broader research community, with more hardware-related support
* 2022-09-13 [CVMFS in Canadian Advanced Research Computing](https://www.computecanada.ca/), workshop (2022)


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=CVMFS/fr&oldid=151027](https://docs.alliancecan.ca/mediawiki/index.php?title=CVMFS/fr&oldid=151027)"
