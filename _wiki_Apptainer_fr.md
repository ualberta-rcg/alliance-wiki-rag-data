# Apptainer

This page is a translated version of the page Apptainer and the translation is 100% complete.

Other languages: English, français


## Contents

* [Introduction](#introduction)
    * [Official Apptainer Documentation](#official-apptainer-documentation)
    * [If you use Singularity](#if-you-use-singularity)
    * [Other Linux Container Technologies](#other-linux-container-technologies)
    * [Other Topics](#other-topics)
        * [Generalities](#generalities)
        * [sudo](#sudo)
        * [Building Images or Overlays](#building-images-or-overlays)
* [Loading the Apptainer Module](#loading-the-apptainer-module)
* [Running Programs in a Container](#running-programs-in-a-container)
    * [Important Command-Line Options](#important-command-line-options)
    * [Using GPUs](#using-gpus)
    * [Running MPI Programs](#running-mpi-programs)
    * [Help with `apptainer run-help`](#help-with-apptainer-run-help)
    * [Running Software with `apptainer run` or `apptainer exec`](#running-software-with-apptainer-run-or-apptainer-exec)
    * [Interactivity with `apptainer shell`](#interactivity-with-apptainer-shell)
    * [Using Daemons with `apptainer instance`](#using-daemons-with-apptainer-instance)
    * [Working with MPI Programs](#working-with-mpi-programs)
* [Bind Mounts and Persistent Overlays](#bind-mounts-and-persistent-overlays)
    * [Bind Mounts](#bind-mounts)
    * [Persistent Overlays](#persistent-overlays)
* [Building an Apptainer Image](#building-an-apptainer-image)
    * [Building a SIF Image](#building-a-sif-image)
    * [Building a Sandbox](#building-a-sandbox)
* [Use Cases](#use-cases)
    * [Working with Conda](#working-with-conda)
    * [Working with Spack](#working-with-spack)
    * [Working with NVIDIA GPUs](#working-with-nvidia-gpus)
    * [Working with MPI](#working-with-mpi)
    * [Creating an Apptainer Container from a Dockerfile](#creating-an-apptainer-container-from-a-dockerfile)
* [Miscellaneous Topics](#miscellaneous-topics)
    * [Clearing the Cache Directory](#clearing-the-cache-directory)
    * [Modifying Default Directories](#modifying-default-directories)


## Introduction

### Official Apptainer Documentation

This page does not describe all features and does not replace the [official Apptainer documentation](link-to-official-documentation). We describe basic usage here, address some aspects of usage on our systems, and present examples. We recommend reading the official documentation on the features you are using.

To install Apptainer on your computer, [consult this page](link-to-installation-page). If you are using a recent version of Windows, first install WSL, then install Apptainer in the subsystem. If you are using macOS, first install a Linux distribution in a virtual machine on your computer, then install Apptainer in the virtual machine.


### If you use Singularity

We recommend using Apptainer instead of Singularity. The Linux Foundation adopted SingularityCE (up to v3.9.5) and renamed it Apptainer, with the following changes:

* Added support for DMTCP (Distributed MultiThreaded Checkpointing).
* Dropped support for the `--nvccli` command-line option.
* Dropped support for `apptainer build --remote`.
* Replaced the SylabsCloud remote endpoint with a DefaultRemote endpoint, without server definition for `library://`.  If needed, you can [restore the SylabsCloud remote endpoint](link-to-restoring-sylabscloud).
* Replaced the term `singularity` with `apptainer` in all executable names, paths, etc.; e.g., the command `singularity` is changed to `apptainer`, e.g., the directory `~/.singularity` is changed to `~/.apptainer`.
* Replaced the term `SINGULARITY` with `APPTAINER` in all environment variables.

Apptainer version 1 being compatible with Singularity, you can use the same scripts.


### Other Linux Container Technologies

High-performance computing clusters usually use Apptainer. In response to several requests about other Linux container technologies, here are our comments on a few:

* **Podman:** Like Apptainer, supports the use of normal (rootless) containers, is available as a package for Linux distributions that support RPM, and for a few others; even though it is a Linux container technology, Podman can be installed on Windows and macOS computers; Podman version 4 supports Apptainer .SIF files.
* **Docker:** Docker cannot be used securely on multi-user systems. It is therefore not offered on our clusters; you can install Docker on your computer and create an Apptainer image which will then be uploaded to a high-performance computing cluster as described below.


### Other Topics

#### Generalities

You must first have an image of your container, i.e., a .sif file or a directory serving as a sandbox. If this is not the case, see Building an Apptainer Image below.

In addition to having installed Apptainer, you also need to install or build all the necessary software to work in the container. Several software packages are already installed on our clusters and you can use them without creating a container.


#### sudo

Websites and documentation often refer to `sudo` for obtaining superuser (root) permissions, but this is not possible on our clusters. If you need to use `sudo`, your options are:

1. Install Linux, Apptainer, and `sudo` in a virtual machine on a computer you control, which will give you `sudo` access. Build your image(s) in this machine and upload them to one of our clusters.
2. If necessary, request assistance from technical support to build your image. If it is not possible to do it for you with `sudo`, we may be able to offer you other solutions.

From version 1.1.x, support for implicit or explicit use of `--fakeroot` makes possible things that were not possible with previous versions or with Singularity, for example the possibility of building images from .def definition files or building images without using `sudo`. That being said, it must be remembered that not all images can be built without `sudo` or without root permissions.


#### Building Images or Overlays

To build your own images or overlays:

* Do not build a sandbox image with `--fakeroot` in a network file system; see the [official Apptainer documentation](link-to-official-documentation).
* Configure `APPTAINER_CACHEDIR` to indicate a location in a file system that is not networked; see the [official Apptainer documentation](link-to-official-documentation).
* Configure `APPTAINER_TMPDIR` to indicate a location in a file system that is not of Lustre/GPFS type; see the [official Apptainer documentation](link-to-official-documentation).
* Do not use file systems that are of Lustre/GPFS type because they do not offer the necessary features for container building (especially `--fakeroot`); see the [official Apptainer documentation](link-to-official-documentation).


## Loading the Apptainer Module

To use the default version, run:

```bash
$ module load apptainer
```

To see all available versions, run:

```bash
$ module spider apptainer
```


## Running Programs in a Container

### Important Command-Line Options

Software running in a container is in an environment that uses different libraries and tools than those installed on the host system. It is therefore important that programs running in a container do not use configuration parameters or software defined outside the container. However, Apptainer adopts the host interpreter's environment by default, which can cause problems when running some programs. The following options used with `apptainer run`, `apptainer shell`, `apptainer exec`, and/or `apptainer instance` avoid these problems.

**Command-line options:**

* `-C`: Isolates the active container from all file systems, the parent PID, IPC, and the environment. To access file systems outside the container, you must use bind mounts.
* `-c`: Isolates the active container from most file systems, using only a minimal `/dev` directory, an empty `/tmp` directory, and an empty `/home` directory. To access file systems outside the container, you must use bind mounts.
* `-e`: Removes certain variables from the interpreter's environment before launching commands and configures parameters for better OCI/Docker compatibility. This option implicitly adds `--containall`, `--no-init`, `--no-umask`, and `--writable-tmpfs`.

Another important option is `-W` or `--workdir`. On our clusters and with most Linux systems, file systems similar to `/tmp` use RAM and not disk space. Tasks running on our clusters usually have little RAM and are cancelled if they consume more memory than allocated. To get around this problem, Apptainer must use a physical disk for its workdir. To do this, use the `-W` option followed by the path to a disk where Apptainer can read and write temporary files. In the following example, the `myprogram` command in the `myimage.sif` container image specifies the `/path/to/a/workdir` workdir.

```bash
apptainer run -C -B /project -W /path/to/a/workdir myimage.sif myprogram
```

The `workdir` option can be removed if no active container uses it.

When Apptainer is used in a task launched with `salloc`, `sbatch`, or JupyterHub on our clusters, the work directory must be `${SLURM_TMPDIR}`, for example `-W ${SLURM_TMPDIR}`.

**Note:** No intensive program (including Apptainer) should be run on login nodes. Use `salloc` to start an interactive task instead.

Bind mounts do not work the same way on all our clusters; see Bind Mounts and Persistent Overlays below to learn how to access `/home`, `/project`, and `/scratch`.


### Using GPUs

Consider the following points when your software in a container requires the use of GPUs:

* Make sure to pass `--nv` (for NVIDIA hardware) and `--rocm` (for AMD hardware) to Apptainer commands. These options ensure that the appropriate entries in `/dev` are included in the bind mount inside the container.
* These options locate the libraries for the GPUs and attach them to the host, in addition to configuring the `LD_LIBRARY_PATH` environment variable so that the libraries work in the container.
* Make sure that the application using the GPU in the container has been correctly compiled to be able to use the GPU and its libraries.
* To use OpenCL in the container, use the previous options and add the bind mount `-B /etc/OpenCL`.

See the example under Working with NVIDIA GPUs below.


### Running MPI Programs

To run MPI programs in a container, some things need to be adjusted in the host environment. See an example in Working with MPI Programs below. You will find more information in the [official Apptainer documentation](link-to-official-documentation).


### Help with `apptainer run-help`

Apptainer containers built from definition files often have a `%help` feature called as follows:

```bash
apptainer run-help your-container-name.sif
```

where `your-container-name.sif` is the name of the container.

If your container also has applications, run:

```bash
apptainer run-help --app appname your-container-name.sif
```

where `appname` is the name of the application and `your-container-name.sif` is the name of the container.

To get the list of applications in the container, run:

```bash
apptainer inspect --list-apps your-container-name.sif
```

where `your-container-name.sif` is the name of the container.


### Running Software with `apptainer run` or `apptainer exec`

The `apptainer run` command launches the container, executes the `%runscript` script defined for that container (if any), then launches the specified command.  The `apptainer exec` command, on the other hand, will not execute the script, even if it is defined in the container. We recommend always using `apptainer run`.

Suppose you want to compile the C++ program `myprog.cpp` located in a container with `g++`, then run the program. You can use:

```bash
apptainer run your-container-name.sif g++ -O2 -march=broadwell ./myprog.cpp
apptainer run your-container-name.sif ./a.out
```

where `your-container-name.sif` is the name of the .SIF file and `g++ -O2 -march=broadwell ./myprog.cpp` is the command to execute in the container.

On our clusters, you will need to add options after `run`, but before `your-container-name.sif`, including `-C`, `-c`, `-e`, and `-W`, plus some bind mount options so that disk space is available for programs in the container, for example:

```bash
apptainer run -C -W $SLURM_TMPDIR -B /project -B /scratch your-container-name.sif g++ -O2 -march=broadwell ./myprog.cpp
apptainer run -C -W $SLURM_TMPDIR -B /project -B /scratch ./a.out
```

For more information, see Important Command-Line Options, Using GPUs, Bind Mounts and Persistent Overlays.  Also consult the [official documentation for Apptainer](link-to-official-documentation).


### Interactivity with `apptainer shell`

The `apptainer run`, `apptainer exec`, and `apptainer instance` commands immediately execute programs, which is perfect in BASH and Slurm job scripts. It may sometimes be necessary to work interactively in a container; to do so, use the `apptainer shell` command.

For example:

```bash
apptainer shell your-container-name.sif
```

where `your-container-name.sif` is the name of your SIF file.

When the container is ready, the `Apptainer>` prompt will appear (or `Singularity>` in the case of earlier versions). Then enter the commands for the interpreter, then enter `exit` and press the Enter/Return key to exit the container.

On our clusters, you will need to add options after `run`, but before `your-container-name.sif`, including `-C`, `-c`, `-e`, and `-W`, plus some bind mount options so that disk space is available for programs in the container, for example:

```bash
apptainer shell -C -W $SLURM_TMPDIR -B /home:/cluster_home -B /project -B /scratch your-container-name.sif
```

For more information, see Important Command-Line Options, Using GPUs, Bind Mounts and Persistent Overlays. Also consult the [official Apptainer documentation](link-to-official-documentation).

**IMPORTANT:** If you are using a persistent overlay image (in a SIF file or a separate file) and you want this image to reflect the modifications, you must, in addition to the options named above, pass the `-w` or `--writable` option to the container, otherwise the modifications made in the `apptainer shell` session will not be saved.


### Using Daemons with `apptainer instance`

Apptainer is designed to correctly run daemons for computing tasks on clusters, partly using the `apptainer instance` command. See details in [Running Services](link-to-running-services) of the official documentation.

**Note 1:** Do not manually run a daemon without using `apptainer instance` and other related commands. Apptainer works well with other tools like the Slurm scheduler used on our clusters. When a task crashes, is cancelled, or ends in any other way, daemons launched with `apptainer instance` will not be blocked and will not leave defunct processes. Also, the `apptainer instance` command allows you to control the daemons and programs that are running in the same container.

**Note 2:** Daemons are only executed when the task is running. If the scheduler cancels the task, all daemons attached to it will also be cancelled. If you need daemons that remain active beyond the execution time, you can instead run them in a virtual machine, in a cloud; then contact technical support.


### Working with MPI Programs

Running MPI programs on nodes in an Apptainer container requires special configuration. Communication between nodes is much more efficient with MPI because of its good use of interconnect hardware. This is usually done automatically and causes no problems, except when the program uses multiple nodes in a cluster.

**Note:** When all MPI processes are run in an Apptainer container on a single shared-memory node, the interconnect hardware is not solicited and no problems occur, for example with the `--nodes=1` option in an `sbatch` script. However, if the number of nodes is not explicitly defined as 1, the scheduler may choose to run the MPI program on multiple nodes and it is possible that the task cannot be executed.

(content in preparation)


## Bind Mounts and Persistent Overlays

Apptainer offers the following features: bind mounts, to access disk space outside the container; persistent overlays, to overlay a read/write file system on an immutable (read-only) container.


### Bind Mounts

Using the `-C` or `-c` options with a container prevents access to your disk space. To compensate, you must explicitly request the bind mount of this space. Suppose the `-C` option is used in an `apptainer run -C -W $SLURM_TMPDIR a-container.sif wc -l ./my_data_file.txt` task, where `./my_data_file.txt` is a file in the current directory of the host, i.e., the file is not located in the container. The `-C` option ensures that the `wc` program in the container will not have access to the file and an access error will occur. To avoid this, you must bind mount the current directory:

```bash
apptainer run -C -B . -W $SLURM_TMPDIR a-container.sif wc -l ./my_data_file.txt
```

where `-B .` bind mounts the current directory.

Even if it is possible to create multiple bind mounts, it is often simpler to bind mount the higher-level directory under which the directories are located. For example, on our clusters, you can use:

```bash
apptainer run -C -B /project -B /scratch -W $SLURM_TMPDIR a-container.sif wc -l ./my_data_file.txt
```

where `-B /project` bind mounts the `/project` file system and `-B /scratch` bind mounts the `/scratch` file system.

This is particularly useful for accessing files from other members of your team, for accessing files and directories some of which are symlinks to different locations and which might be inaccessible if the bind mount is not done for the entire file system.

If bind mounts do not work on the cluster you are using, run the following script to get the options that must be passed to Apptainer:

```bash
/home/preney/public/apptainer-scripts/get-apptainer-options.sh
```

The bind mount does not necessarily have to be in the same place in the container. You can bind mount a file or directory elsewhere, for example:

```bash
apptainer run -C -B ./my_data_file.txt:/special/input.dat -W $SLURM_TMPDIR a-container.sif wc -l /special/input.dat
```

where the bind mount `-B ./my_data_file.txt:/special/input.dat` associates the file `./my_data_file.txt` with the file `/special/input.dat` in the container, to be processed with the `wc` command. This is useful when programs or scripts in a container contain hard-coded paths to files or directories that are located elsewhere.

If you need to bind mount the `/home` file system in your container, use another destination directory such as `-B /home:/cluster_home`. This ensures that the configuration files and programs that are in your `/home` directory will not interfere with the software in your container. Conversely, if you use `-B /home`, programs in `$HOME/bin` and Python packages in `$HOME/.local/lib/python3.x` could be used instead of the corresponding files from the container.

Finally, avoid bind mounting CVMFS in your containers. Programs provided by CVMFS may be incompatible with your containers. The purpose of a container is to provide a complete environment that does not depend on external software. Programs running in a container should be entirely contained therein, and those that are not necessary should not be added.


### Persistent Overlays

See details in [Persistent Overlays](link-to-persistent-overlays) of the official documentation.


## Building an Apptainer Image

**WARNING:** First read the recommendations above in Building Images or Overlays above.

An Apptainer image can be a SIF file or a directory serving as a sandbox.

A SIF file can contain one or more compressed and read-only squashfs file systems. It is also possible for a SIF file to contain read-write files and/or overlay images, but we do not address these cases here; see the [official Apptainer documentation](link-to-official-documentation) instead. Unless more complex methods are used to create an image, the Apptainer `build` command produces a SIF file composed of a read-only squashfs file system. This is the best option, as the read-only image will remain as is and will be more compact; it should be remembered that read operations from this image are very fast.

A sandbox directory is an ordinary directory that is empty at the beginning and to which Apptainer adds files and directories as the image is built. Access to the directory and its update must be done only via Apptainer. A sandbox is useful when it is necessary to access the image in read-write mode to modify it. However, if modifications are infrequent, it is preferable to use a SIF file. It is possible to build an image, make modifications, then build a new SIF file for the modified image, for example:

```bash
$ cd $HOME
$ mkdir mynewimage.dir
$ apptainer build mynewimage.dir myimage.sif
$ apptainer shell --writable mynewimage.dir
Apptainer> # Run commands to update mynewimage.dir here.
Apptainer> exit
$ apptainer build newimage.sif mynewimage.dir
$ rm -rf mynewimage.dir
```

The use of a SIF file is recommended because the performance from the container image is faster than when each file is stored separately in the file systems of our clusters, which are optimized to handle large files and parallel read and write operations. Also, unlike a SIF image, a sandbox will have a significant impact on the number of files you store, while this number is limited by a quota. (Some images can contain thousands of files and directories.)

Root permissions are required for using the package managers of Linux distributions; a simple user cannot therefore build images on our computing clusters with Apptainer 1.0.x and previous Singularity versions. If necessary, write to technical support for assistance in creating your image or use a computer where Apptainer is installed and where you have root permissions.

The `--fakeroot` option of Apptainer is used to create and manipulate images. With versions prior to 1.1, it is necessary to contact technical support and request that an administrator grant permission to use `--fakeroot` on the cluster used, which is not always possible. With Apptainer version 1.1, `--fakeroot` can be used without additional permission.

Some containers cannot be created if you do not have root permissions. Such containers cannot be built on our clusters.

If all you need is a Docker image as is, you will often be able to build and run it easily without root permissions and without using `--fakeroot`. If you later need to modify the image, you may need root permissions, for example to use a package manager. For this reason, the following examples assume the use of a Docker image as is.


### Building a SIF Image

**WARNING:** Please take into account the recommendations made in Building Images or Overlays above.

To build the image of a SIF file with the latest busybox image from Docker, run:

```bash
$ apptainer build bb.sif docker://busybox
```

For more advanced functions, see the [official Apptainer documentation](link-to-official-documentation).


### Building a Sandbox

**WARNING:** Please take into account the recommendations made in Building Images or Overlays above.

To build a sandbox instead of a SIF file, replace the SIF file name with `--sandbox DIR_NAME` or `-s DIR_NAME`, where `DIR_NAME` is the name of the directory to create for the sandbox. For example, to create a SIF file with `apptainer build`, the command is:

```bash
$ apptainer build bb.sif docker://busybox
```

Replace `bb.sif` with a directory name, for example `bb.dir`, with the `--sandbox` option:

```bash
$ apptainer build --sandbox bb.dir docker://busybox
```

Let us recall the differences between a SIF file and a sandbox:

* The container image is contained in a single compressed, read-only SIF file.
* The individual files forming the container image are placed in a directory serving as a sandbox. These files are not compressed, can be numerous (several thousand), and are accessible in read-write mode.

Using a sandbox significantly consumes your disk space and file number quotas. If you do not need frequent read and write access to the container image, it is recommended to use a SIF file. The latter also offers faster access.


## Use Cases

### Working with Conda

Before starting this tutorial, take note of some important points:

* Use Conda as a last resort, even in a container. Prioritize the modules from our software stack and wheels among those available instead. These modules and wheels are optimized for our systems and we can provide better support if needed. To have a module or package added, contact technical support.
* In this tutorial, we use the micromamba package manager instead of Conda. If you want to use Conda, you must take into account the Anaconda terms of use and may need to hold a commercial license.
* In this tutorial, we create a read-only image, i.e., a .sif file that contains a Conda environment with everything you need to use your application. It is strongly recommended not to interactively install software in a container with Conda and no information will be given in this regard.

Creating an Apptainer image and installing software in a container with Conda is a three-step process.

First, you need to create a .yml file that describes the Conda environment to be created in the container; in the following example, it is `environment.yml`. This file contains the name of the environment to create, the list of packages to install, and how to find them (channel).

**File: `environment.yml`**

```yaml
name: base
channels:
  - conda-forge
  - bioconda
  - defaults
dependencies:
  - python
  - pip
  - star
  - bwa
  - multiqc
```

Then you need to create an image definition file (named here `image.def`) that describes the steps to create the image with Apptainer.

Download a Docker image from DockerHub that contains the pre-installed micromamba package manager.
Create a copy of the `environment.yml` definition file in the container.
Run micromamba to configure the `environment.yml` environment.

**File: `image.def`**

```yaml
Bootstrap: docker
From: mambaorg/micromamba:latest
%files
environment.yml /environment.yml
%post
micromamba install -n base --file environment.yml && \
micromamba clean --all --yes
```

The last step is to build the Apptainer image using the definition file above:

```bash
module load apptainer
APPTAINER_BIND=' ' apptainer build image.sif image.def
```

You can now test if `multiqc` is available with, for example, the command:

```bash
[name@server ~]$ apptainer run image.sif multiqc --help
```


### Working with Spack

(content in preparation)


### Working with NVIDIA GPUs

(content in preparation)


### Working with MPI

(content in preparation)


### Creating an Apptainer Container from a Dockerfile

Note: You must first install Docker and Apptainer on a computer where you have the necessary permissions. The commands presented here do not work on our clusters.

Unfortunately, some software projects offer a Dockerfile instruction file but no container image. It is then necessary to create an image from the Dockerfile. However, Docker is not installed on our clusters. That said, if you can work with a computer where Docker and Apptainer are installed and where you have sufficient permissions (root or sudo access, or membership in the docker group and --fakeroot permission), the following commands will allow you to use Docker then Apptainer to build an Apptainer image on that computer.

Note: Docker may crash if you are not part of the docker group. It may also be impossible to create certain containers without root, sudo, or --fakeroot permissions. Verify that you have the necessary permissions.

If you only have a Dockerfile and want to create an Apptainer image, run the following command on a computer where Docker and Apptainer are installed and where you have the necessary permissions:

```bash
docker build -f Dockerfile -t your-tag-name
docker save your-tag-name -o your-tarball-name.tar
docker image rm your-tag-name
apptainer build --fakeroot your-sif-name.sif docker-archive://your-tarball-name.tar
rm your-tarball-name.tar
```

where `your-tag-name` is the name to give to the Docker container, `your-tarball-name.tar` is the name to give to the file in which Docker will save the generated content for the container, `--fakeroot` can be omitted if it is optional; to use `sudo` instead, omit `--fakeroot` and add `sudo` as a prefix to the line, and `your-sif-name.sif` is the name of the SIF file for the Apptainer container.

The resulting SIF file is an Apptainer container corresponding to the Dockerfile instructions. Copy the SIF file to the cluster(s) you want to use.

Note: It is possible that the Dockerfile adds additional layers; you only have to delete them with `docker images` then `docker image rm ID` (where ID is the identifier of the image obtained by the `docker images` command). This frees up the disk space occupied by the additional layers.


## Miscellaneous Topics

### Clearing the Cache Directory

To find the files in the cache directory, run:

```bash
apptainer cache list
```

Delete the files with:

```bash
apptainer cache clean
```


### Modifying Default Directories

Before launching Apptainer, configure the following environment variables to use temporary and cache directories other than the default ones.

* `APPTAINER_CACHEDIR`: directory where files are downloaded and cached by Apptainer
* `APPTAINER_TMPDIR`: directory where Apptainer saves temporary files, including for the creation of squashfs images

For example, to have Apptainer use your `/scratch` space for cache and temporary files (which is probably the best place), use:

```bash
$ mkdir -p /scratch/$USER/apptainer/{cache,tmp}
$ export APPTAINER_CACHEDIR="/scratch/$USER/apptainer/cache"
$ export APPTAINER_TMPDIR="/scratch/$USER/apptainer/tmp"
```

before launching Apptainer.


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Apptainer/fr&oldid=164639](https://docs.alliancecan.ca/mediawiki/index.php?title=Apptainer/fr&oldid=164639)"
