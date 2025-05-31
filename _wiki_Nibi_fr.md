# Nibi

This page is a translated version of the page Nibi and the translation is 100% complete.

Other languages: English, français

**Availability:** Spring 2025
**Login Node:** nibi.sharcnet.ca (for testing)
**Globus Endpoint:** To be determined
**Copy Node (rsync, scp, sftp, etc.):** Use the login nodes
**Portal:** To be determined

In the Anishinaabe language, *Nibi* is a term that designates water. This new versatile cluster will replace Graham by July 1, 2025. Designed by [Hypertec](https://www.hypertec.ca/en/) and hosted and operated by [SHARCNET](https://www.sharcnet.ca/) at the University of Waterloo, the cluster has 134,400 CPU cores and 288 NVIDIA H100 GPUs.


## Storage

**Parallel Storage:** 25PB, SSD (Solid-State Drive) from [VAST Data](https://www.vastdata.com/) for `/home`, `/project`, and `/scratch` (same as on miniGraham).


## Interconnect

*   Nokia ethernet, 200/400G bandwidth for CPU nodes
*   200 Gbit/s non-blocking bandwidth for all Nvidia GPU nodes
*   400 Gbit/s bandwidth for all AMD GPU nodes
*   200 Gbit/s connection to VAST storage nodes
*   400 Gbit/s uplinks for all nodes; 2:1 blocking


## Node Characteristics

| Nodes | Cores | Available Memory | CPU                                      | GPU                                                              |
|-------|-------|--------------------|-------------------------------------------|--------------------------------------------------------------------|
| 700   | 192   | 768GB DDR5          | 2 x Intel 6972P @ 2.4 GHz, 384MB L3 cache |                                                                    |
| 192   | 6TB   | DDR5                | 2 x Intel 6972P @ 2.4 GHz, 384MB L3 cache |                                                                    |
| 36    | 112   | 2TB DDR5            | 1 x Intel 8570 @ 2.1 GHz, 300MB L3 cache | 8 x Nvidia H100 SXM (80GB memory)                               |
| 6     | 96    | 512GB              | 4 x AMD MI300A @ 2.1 GHz                  | 4 x AMD CDNA 3 (unified memory: 512GB HBM3 total)                   |


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Nibi/fr&oldid=178196](https://docs.alliancecan.ca/mediawiki/index.php?title=Nibi/fr&oldid=178196)"
