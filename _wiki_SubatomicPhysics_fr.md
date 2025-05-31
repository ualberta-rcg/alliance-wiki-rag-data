# Subatomic and High-Energy Physics Software

This page was prepared by the national subatomic physics team and last updated in July 2021.  See also the [Astronomy and High Energy Physics Interactive Analysis Facility](link-to-other-page-needed).

Most experimental subatomic physics groups use CVMFS repositories from CERN, the Open Science Grid consortium, and experiment-specific repositories.

The CCenv configuration available to regular users may conflict with some configurations of these repositories because access is done with Nix and EasyBuild from the Compute Canada CVMFS repository `soft.computecanada.ca` rather than the software installed on the base operating system of the compute nodes.

ATLAS users will find useful information on the TRIUMF twiki pages.

**NOTE:** Use the recommended Tier-3 configurations rather than reinventing the techniques described below.

* [https://twiki.atlas-canada.ca/bin/view/AtlasCanada/ComputeCanadaTier3s](https://twiki.atlas-canada.ca/bin/view/AtlasCanada/ComputeCanadaTier3s)
* [https://twiki.atlas-canada.ca/bin/view/AtlasCanada/Containers](https://twiki.atlas-canada.ca/bin/view/AtlasCanada/Containers)

Many configurations assume that the base nodes are configured with the `HEP_OSLibs` [1] package, which is not the case for our compute nodes. It would be possible to work with some simple configurations of the `sft.cern.ch` repository, but the recommended approach is to use Singularity containers where the required RPMs are installed (see below), which also allows the use of multiple base operating systems (e.g., SL6) on the Compute Canada CentOS-7 infrastructure.


To configure a CentOS7 view from `sft.cern.ch` that includes the necessary paths for geant4, ROOT, etc. compilers:

```bash
source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_95 x86_64-centos7-gcc8-opt
```

Available `arch-os-complier` configurations for LCG_95 are:

* `x86_64-centos7-gcc7-dbg`
* `x86_64-centos7-gcc7-opt`
* `x86_64-centos7-gcc8-dbg`
* `x86_64-centos7-gcc8-opt`
* `x86_64-slc6-gcc62-opt`
* `x86_64-slc6-gcc7-dbg`
* `x86_64-slc6-gcc7-opt`
* `x86_64-slc6-gcc8-dbg`
* `x86_64-slc6-gcc8-opt`
* `x86_64-ubuntu1804-gcc7-opt`
* `x86_64-ubuntu1804-gcc8-dbg`
* `x86_64-ubuntu1804-gcc8-opt`

[1] A list of all RPMs installed via HEPOS_Libs for CentOS7 can be found at [https://gitlab.cern.ch/linuxsupport/rpms/HEP_OSlibs/blob/7.2.11-3.el7/dependencies/HEP_OSlibs.x86_64.dependencies-recursive-flat.txt](https://gitlab.cern.ch/linuxsupport/rpms/HEP_OSlibs/blob/7.2.11-3.el7/dependencies/HEP_OSlibs.x86_64.dependencies-recursive-flat.txt).


## Running in a Container

As of May 2020, we know of two main repositories for container images for high-energy physics software; they are distributed via CVMFS repositories.

**ATLAS:** Singularity image distributions are well documented at [https://twiki.cern.ch/twiki/bin/view/AtlasComputing/ADCContainersDeployment](https://twiki.cern.ch/twiki/bin/view/AtlasComputing/ADCContainersDeployment)

* **Packed images:** `/cvmfs/atlas.cern.ch/repo/containers/images/singularity/`
* **Unpacked images:** `/cvmfs/atlas.cern.ch/repo/containers/fs/singularity/`

**WLCG:** (Unpacked repository). This development project uses DUCC to automatically publish container images from a Docker registry to CVMFS. Images are published to CVMFS in a standard directory structure format used by Singularity as well as in the layered format used by Docker, allowing images to be instantiated directly from CVMFS with the Graph Driver plugin.  More documentation on this project can be found at [https://github.com/cvmfs/ducc](https://github.com/cvmfs/ducc). The automatically published image list includes the `atlas-grid-centos7` image. You can request to merge an additional image to this list.

Images are under `/cvmfs/unpacked.cern.ch/`.


## Invoking a Singularity Image

The Singularity executable is somewhat different depending on the Compute Canada site because it is started in a `setuid` environment by default and therefore installed elsewhere than with the usual Compute Canada CVMFS software. Several versions are available on each site and the defaults can be modified; it is therefore preferable to invoke the necessary version (currently, these are possibly 2.6.1, 3.2.0, and 3.3.0).

* **cedar:** `/opt/software/singularity-x.x.x`
* **graham:** `/opt/software/singularity-x.x.x`
* **niagara:** `module load singularity; /opt/singularity/2`

To invoke a container from a CVMFS repository, you can either do it directly since the image will be cached, or download it locally, which can improve performance depending on the system.
