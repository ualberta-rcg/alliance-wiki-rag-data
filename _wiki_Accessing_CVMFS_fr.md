# Accessing CVMFS

This page is a translated version of the page Accessing CVMFS and the translation is 100% complete.

Other languages: [English](link-to-english-page), [français](current-page-link)


## Introduction

The software and data directories we offer are accessible via CVMFS (CERN Virtual Machine File System).  Since CVMFS is pre-configured for you, you can use its directories directly. For more information on our software environment, see the wiki pages [Available Software](link-to-available-software-page), [Using Modules](link-to-using-modules-page), [Python](link-to-python-page), [R](link-to-r-page), and [Installing Software in your /home directory](link-to-home-install-page).

This document describes how to install and configure CVMFS on your own computer or cluster, giving you access to the same directories and software environments as our systems.  We use as an example the software environment presented at the PEARC 2019 conference, Practices and Experience in Advanced Research Computing.


## Avant de commencer (Before you begin)

If you are a member of our technical teams, please read the [internal documentation](link-to-internal-documentation).

**Important:** Please subscribe to the [announcement service](link-to-announcement-service) and fill out this [registration form](link-to-registration-form). If you use our software environment in your research, please acknowledge our contribution according to [these guidelines](link-to-guidelines). We also thank you for mentioning our [presentation](link-to-presentation).


### S'abonner au service d'annonces (Subscribing to the announcement service)

Changes may be made to CVMFS or the software and other content of the directories we provide; these changes affect users or require administrator intervention to ensure service continuity.

Subscribe to the cvmfs-announce@gw.alliancecan.ca mailing list to receive occasional important announcements. You can subscribe by writing to cvmfs-announce+subscribe@gw.alliancecan.ca and replying to the confirmation email you will receive.

Members of our technical teams can also [subscribe here](link-to-technical-team-subscription).


### Conditions d’utilisation et soutien technique (Terms of use and technical support)

The CVMFS client software is provided by CERN. Our CVMFS directories are provided without any warranty. Your access to the directories and the software environment may be limited or blocked if you violate the [terms of use](link-to-terms-of-use), or at our discretion.


### Exigences techniques (Technical requirements)

#### Pour un seul système (For a single system)

To install CVMFS on a personal computer, the requirements are:

* A compatible operating system (see [Basic Requirements](#exigences-de-lenvironnement-logiciel) below);
* The FUSE free software;
* Approximately 50GB of local storage space for the cache; a larger or smaller cache may be suitable depending on circumstances. For limited use on a personal computer, 5 to 10GB may be sufficient. For more information, see the [Cache Settings](link-to-cache-settings) paragraph.
* HTTP access to the internet, or HTTP access to one or more local proxy servers.

If these conditions are not met or you have other restrictions, consider this [other option](link-to-other-option).


#### Pour plusieurs systèmes (For multiple systems)

To deploy multiple CVMFS clients, for example on a cluster, in a laboratory, on a campus or otherwise, each system must meet the specific requirements listed above.  Also consider the following points:

* To improve performance, we recommend deploying HTTP proxy servers with external cache (forward caching) on your site, especially if you have multiple clients (see [Setting up a Local Squid Proxy](link-to-squid-proxy-setup)).
* Having only one proxy server is a single point of failure. As a general rule, you should have at least two local proxy servers and preferably one or more additional proxy servers nearby to take over in case of a problem.
* We recommend synchronizing the service account identity `cvmfs` of all client nodes with LDAP or otherwise. This will facilitate the use of an [external cache](link-to-external-cache) and should be done *before* CVMFS is installed. Even if the use of an external cache is not planned, it is easier to synchronize accounts from the start than to try to change them later.


### Exigences de l’environnement logiciel (Software environment requirements)

#### Exigences de base (Basic requirements)

* **Operating System:**
    * Linux: with kernel 2.6.32 or higher for 2016 and 2018 environments; kernel 3.2 or higher for the 2020 environment,
    * Windows: with version 2 of the Windows Subsystem for Linux (WSL) and a Linux distribution with kernel 2.6.32 or higher,
    * Mac OS: via virtual instance only;
* **CPU:** x86, for instruction sets SSE3, AVX, AVX2 or AVX512.

#### Pour une utilisation optimale (For optimal use)

* **Scheduler:** Slurm or Torque, for tight integration with OpenMPI applications;
* **Network Interconnect:** Ethernet, InfiniBand or OmniPath, for parallel applications;
* **GPU:** NVidia with CUDA drivers 7.5 or higher, for CUDA applications (see warning below);
* A minimum of Linux packages, to avoid the risk of conflicts.


## Installer CVMFS (Installing CVMFS)

If you want to use Ansible, there is a [CVMFS client role](link-to-ansible-role) for the basic configuration of a CVMFS client with an RPM system.  [Scripts](link-to-scripts) are available for easy installation of CVMFS on a cloud instance. Otherwise, follow the instructions below.


### Préinstallation (Pre-installation)

We recommend that the local CVMFS cache (located by default in `/var/lib/cvmfs` and configurable with the `CVMFS_CACHE_BASE` parameter) be located in a dedicated file system so that storage is not shared with that of other applications. You should therefore have this file system *before* installing CVMFS.


### Installation et configuration (Installation and configuration)

See the installation instructions in [Getting the Software](link-to-getting-software). To configure a client, see [Setting up the Software](link-to-setup-software) and [Client parameters](link-to-client-parameters). The `soft.computecanada.ca` repository is provided with the configuration and you can access it; you can include it in your client configuration `CVMFS_REPOSITORIES`.


### Test (Test)

First, make sure the directories to be tested are in `CVMFS_REPOSITORIES`. Validate the configuration:

```bash
[name@server ~]$ sudo cvmfs_config chksetup
```

Be sure to address any warnings or errors that may occur. Check the directories:

```bash
[name@server ~]$ cvmfs_config probe
```

If you have problems, this [debugging guide](link-to-debugging-guide) may be helpful.


## Activer notre environnement dans votre session (Activating our environment in your session)

Once the CVMFS repository is mounted, our environment is activated in your session using the script `/cvmfs/soft.computecanada.ca/config/profile/bash.sh`. This will load default modules. If you wish to have the default modules of a particular compute cluster, define the `CC_CLUSTER` variable by choosing one of the following values: `beluga`, `cedar`, or `graham`, before using the script. For example:

```bash
[name@server ~]$ export CC_CLUSTER=beluga
[name@server ~]$ source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
```

This command will do nothing if your user ID is below 1000. This is a security measure because you should not expect our software environment to give you operating privileges. If you still want to activate our environment, you can first define the variable `FORCE_CC_CVMFS=1` with the command:

```bash
[name@server ~]$ export FORCE_CC_CVMFS=1
```

Or, if you want our environment to be active permanently, you can create the file `$HOME/.force_cc_cvmfs` in your `/home` directory with:

```bash
[name@server ~]$ touch $HOME/.force_cc_cvmfs
```

If you want to avoid activating our environment, you can define `SKIP_CC_CVMFS=1` or create the file `$HOME/.skip_cc_cvmfs` to ensure that our environment is never activated in this particular environment.


### Personnaliser votre environnement (Customizing your environment)

By default, certain features of your system will be automatically detected by activating our environment and the required modules will be loaded. This default behavior can be modified by pre-defining the specific environment variables described below.


#### Variables d’environnement (Environment variables)

##### CC_CLUSTER

This variable identifies the cluster. It routes information to the system log and defines the behavior to adopt according to the software license. Its default value is `computecanada`. You could set its value so that the logs are identified by the name of your system.

##### RSNT_ARCH

This variable identifies the CPU instruction set for the system. By default, it is detected automatically from `/proc/cpuinfo`. However, you can use another instruction set by defining the variable before activating the environment. The possible sets are: `sse3`, `avx`, `avx2`, `avx512`.

##### RSNT_INTERCONNECT

This variable identifies the type of network interconnect of the system. It is automatically detected based on the presence of `/sys/module/opa_vnic` for OmniPath or `/sys/module/ib_core` for InfiniBand. The replacement value is `ethernet`. The possible values are: `omnipath`, `infiniband`, `ethernet`. The value of the variable triggers different transport protocol options for OpenMPI.

##### RSNT_CUDA_DRIVER_VERSION

This variable is used to hide or show versions of our CUDA modules depending on the required version for NVidia drivers, as documented [here](link-to-cuda-documentation). If the variable is not defined, the files in `/usr/lib64/nvidia` determine the versions to hide or show. If no libraries are found in `/usr/lib64/nvidia`, we assume that the driver versions are sufficient for CUDA 10.2. This is to ensure backward compatibility since this functionality became available with the release of CUDA 11.0. Defining the environment variable `RSNT_CUDA_DRIVER_VERSION=0.0` hides all CUDA versions.

##### RSNT_LOCAL_MODULEPATHS

This variable identifies where local module trees are located and integrates them into our central tree. First define:

```bash
[name@server ~]$ export RSNT_LOCAL_MODULEPATHS=/opt/software/easybuild/modules
```

Then install your EasyBuild recipe with:

```bash
[name@server ~]$ eb --installpath /opt/software/easybuild <your recipe>.eb
```

Our module naming convention will be used to install your recipe locally, which will be used in the module hierarchy. For example, if the recipe uses the compilation chain `iompi,2018.3`, the module will be available after the `intel/2018.3` and `openmpi/3.1.2` modules have been loaded.

##### LMOD_SYSTEM_DEFAULT_MODULES

This variable identifies the modules to load by default. If it is not defined, our environment loads the `StdEnv` module by default, which in turn loads a version of the Intel compiler and an OpenMPI version by default.

##### MODULERCFILE

This variable is used by Lmod to define the default version of modules and aliases. You can define your own `modulerc` file and add it to `MODULERCFILE`. This will take precedence over what is defined in our environment.


#### Chemin des fichiers (File paths)

Our software environment is designed to depend as little as possible on the host operating system; however, it must recognize certain paths to facilitate interactions with the tools installed there.

##### /opt/software/modulefiles

If it exists, this path is automatically added to the default `MODULEPATH`. This allows the use of our environment while keeping locally installed modules.

##### $HOME/modulefiles

If it exists, this path is automatically added to the default `MODULEPATH`. This allows the use of our environment by allowing the installation of modules in the `/home` directories.

##### /opt/software/slurm/bin, /opt/software/bin, /opt/slurm/bin

These paths are automatically added to the default `PATH`. It allows the addition of your executable in the search path.


### Installation locale de logiciels (Local software installation)

Since June 2020, it is possible to install additional modules on your compute cluster; these modules will then be recognized by our central hierarchy. For more information, see the [discussion and implementation on this topic](link-to-local-install-discussion).

To install additional modules, first identify a path where to install the software, for example `/opt/software/easybuild`. Make sure this directory exists. Then export the environment variable `RSNT_LOCAL_MODULEPATHS`:

```bash
[name@server ~]$ export RSNT_LOCAL_MODULEPATHS=/opt/software/easybuild/modules
```

If you want your users to be able to find this branch, we recommend that you define this environment variable in the common profile of the cluster. Then install the software packages you want with EasyBuild:

```bash
[name@server ~]$ eb --installpath /opt/software/easybuild <some easyconfig recipe>
```

The software will be installed locally according to our module naming hierarchy. They will be automatically presented to users when they load our compiler, MPI and CUDA.


## Mises en garde (Warnings)

### Utilisation de l’environnement logiciel par un administrateur (Use of the software environment by an administrator)

If you perform system operations with privileges or operations related to CVMFS, make sure your session does not depend on the software environment. For example, if you update CVMFS with YUM while your session uses a Python module loaded from CVMFS, YUM could be executed using this same module and lose access, blocking the update. Similarly, if your environment depends on CVMFS and you reconfigure CVMFS so that access to CVMFS is temporarily interrupted, your session could interfere with CVMFS operations or be suspended.  Considering this, updating or reconfiguring CVMFS can be done without service interruption in most cases, as the operation would succeed due to the absence of a circular dependency.


### Paquets logiciels non disponibles (Unavailable software packages)

We make several commercial software packages available to users, subject to the license of these products. These software packages are not available elsewhere than with our resources and you will not have access to them even if you follow the instructions to install and configure CVMFS. For example, Intel and Portland Group compilers: if the modules for these compilers are available, you only have access to the redistributable parts, usually the shared objects. You will be able to run compiled applications, but you will not be able to compile new applications.


### Localisation de CUDA (CUDA location)

In the case of CUDA packages, our software environment uses driver libraries installed in `/usr/lib64/nvidia`. However, with some platforms, recent NVidia drivers install the `/usr/lib64` libraries in `LD_LIBRARY_PATH` without borrowing all the system libraries, which could create incompatibilities with our software environment; we therefore recommend that you create symbolic links in `/usr/lib64/nvidia` to redirect to the NVidia libraries that are installed. The following script is used to install the drivers and create the symbolic links (replace the version number with the one you want).

```bash
# File: script.sh
NVIDIA_DRV_VER="410.48"
nv_pkg=("nvidia-driver" "nvidia-driver-libs" "nvidia-driver-cuda" "nvidia-driver-cuda-libs" "nvidia-driver-NVML" "nvidia-driver-NvFBCOpenGL" "nvidia-modprobe")
yum -y install ${nv_pkg[@]/%/}-${NVIDIA_DRV_VER}
for file in $(rpm -ql ${nv_pkg[@]}); do
  [ "${file%/*}" = '/usr/lib64' ] && [ ! -d "${file}" ] && \
  ln -snf "$file" "${file%/*}/nvidia/${file##*/}"
done
```


### LD_LIBRARY_PATH

Our environment is designed to use `RUNPATH`. It is not recommended to define `LD_LIBRARY_PATH` as the environment could cause problems.


### Bibliothèques introuvables (Missing libraries)

Since we do not define `LD_LIBRARY_PATH` and our libraries are not installed in default Linux locations, binary packages like Anaconda often have difficulty finding the libraries they need. Consult our [documentation on installing binary packages](link-to-binary-install-docs).


### dbus

For some applications, `dbus` must be installed locally, on the host operating system.


**(Retrieved from "https://docs.alliancecan.ca/mediawiki/index.php?title=Accessing_CVMFS/fr&oldid=172656")**
