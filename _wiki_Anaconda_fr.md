# Anaconda

Other languages: English, français

Anaconda is a Python distribution. We ask our users not to install Anaconda on our clusters. We recommend using other methods such as a virtual environment or an Apptainer container for more complex cases.


## Do Not Install Anaconda on Our Clusters

We are aware that Anaconda is widely used in several fields studied by our users (data science, AI, bioinformatics, etc.). Anaconda is an interesting solution to simplify the management of Python and libraries on a personal computer. However, on a cluster like those maintained by the Alliance, library management must be done by our staff to ensure maximum compatibility and performance. Here is a list of reasons:

* Anaconda very often installs software (compilers, scientific libraries, etc.) that already exist on the Alliance clusters as modules, with a non-optimal configuration.
* Installs binaries that are not optimized for the processors of our clusters.
* Makes wrong assumptions about the location of libraries.
* Installs in `$HOME` by default, where it places a huge amount of files. The installation of Anaconda alone can take up nearly half of your file quota in your personal space.
* Is slower to install packages.
* Modifies `$HOME/.bashrc`, which can cause conflicts.


## Transitioning from Conda to virtualenv

`virtualenv` offers all the features you need to use Python on our clusters. Here's how to switch to `virtualenv` if you're using Anaconda on your personal computer:

1. List the dependencies (requirements) of the application you want to use. To do this, you can:
    * Run `pip show <package_name>` from your virtual environment (if the package exists on PyPI).
    * Or, check if there is a `requirements.txt` file in the Git repository.
    * Or, check the `install_requires` variable of the `setup.py` file which lists the requirements.

2. Find out which dependencies are Python packages and which are libraries provided by Anaconda. For example, CUDA and CuDNN are libraries available on the Anaconda Cloud, but you should not install them yourself on our clusters. They are already installed.

3. Remove from the dependency list anything that is not a Python package (for example, remove `cudatoolkit` and `cudnn`).

4. Use a `virtualenv`, in which you will install these dependencies.

5. Your application should work. If not, feel free to contact our technical support.


## Using Apptainer

In some situations, the complexity of a software's dependencies requires a solution where the environment can be fully controlled. For these situations, we recommend the `Apptainer` tool: note that a Docker image can be converted into an Apptainer image. The only drawback of Apptainer is that the images consume a lot of disk space, so if your research group plans to use several images, it would be wise to group them together in a single directory of the group's project space to avoid duplicates.


## Specific Examples Where Anaconda Does Not Work

**R:** A conda recipe forces the installation of R. This installation does not perform as well as the R available through the modules (which uses Intel MKL). This same R malfunctions and tasks die, wasting resources and your time.
