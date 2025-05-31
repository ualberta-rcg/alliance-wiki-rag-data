# Advanced Jupyter Configuration

This page is a translated version of the page [Advanced Jupyter configuration](https://docs.alliancecan.ca/mediawiki/index.php?title=Advanced_Jupyter_configuration&oldid=156875) and the translation is 100% complete.

Other languages: [English](https://docs.alliancecan.ca/mediawiki/index.php?title=Advanced_Jupyter_configuration&oldid=156875), français

## Introduction

Running Jupyter notebooks is suitable for short interactive tasks for quick testing, debugging, or data visualization (a few minutes). For longer analysis, running longer analysis must be done in a non-interactive task with `sbatch`.  See also [Running notebooks as Python scripts](#running-notebooks-as-python-scripts) below.

[Project Jupyter](https://jupyter.org/about.html) is a non-profit, open-source project stemming from the IPython Project in 2014 to enable interactive data science and scientific computing across all programming languages.<sup>[1](#references)</sup>

[JupyterLab](https://jupyter.org/) is an interactive web-based development environment for notebooks, code, and data. Its flexible interface allows for the configuration and use of workflows in data science, scientific computing, computational journalism, and machine learning. Its modular design allows for the addition of extensions that enrich its functionalities.<sup>[2](#references)</sup>

A JupyterLab server should always be located on a compute node or cloud instance. Login nodes are not a good choice because they impose limits that can interrupt an application that would consume too much CPU time or RAM. To obtain a compute node, you can reserve resources by submitting a job that requests a predetermined number of CPUs or GPUs, a certain amount of memory, and an execution time limit.

We describe here how to configure and submit a JupyterLab job on our national clusters. If you are looking for a pre-configured Jupyter environment, see the [Jupyter](https://docs.alliancecan.ca/mediawiki/index.php?title=Jupyter&oldid=156875) page.


## Installing JupyterLab

These instructions install JupyterLab using the `pip` command in a Python virtual environment.

If you don't already have a Python virtual environment, create one and activate it. Load the default Python module (as shown below) or load a specific version (see available versions with `module avail python`).

```bash
[name@server ~]$ module load python
```

If you intend to use RStudio Server, load `rstudio-server` first with:

```bash
[name@server ~]$ module load rstudio-server python
```

Create a new Python virtual environment.

```bash
[name@server ~]$ virtualenv --no-download $HOME/jupyter_py3
```

Activate the new virtual environment.

```bash
[name@server ~]$ source $HOME/jupyter_py3/bin/activate
```

Install JupyterLab in your new virtual environment (this will take a few minutes).

```bash
(jupyter_py3) [name@server ~]$ pip install --no-index --upgrade pip
(jupyter_py3) [name@server ~]$ pip install --no-index jupyterlab
```

In the virtual environment, create a wrapper script for automatic launching of JupyterLab.

```bash
(jupyter_py3) [name@server ~]$ echo -e '#!/bin/bash\nunset XDG_RUNTIME_DIR\njupyter lab --ip $(hostname -f) --no-browser' > $VIRTUAL_ENV/bin/jupyterlab.sh
```

Finally, make this script executable.

```bash
(jupyter_py3) [name@server ~]$ chmod u+x $VIRTUAL_ENV/bin/jupyterlab.sh
```


## Installing Extensions

Extensions add functionalities and can modify the JupyterLab user interface.


### Jupyter Lmod

[Jupyter Lmod](https://github.com/jupyterlab/jupyterlab-lmod) is an extension that allows interaction with environment modules before launching kernels. It uses the Python interface of Lmod to perform module-related tasks such as loading, unloading, saving collections, etc.

The following commands will install and enable the Jupyter Lmod extension in your environment (the third command will take a few minutes).

```bash
(jupyter_py3) [name@server ~]$ module load nodejs
(jupyter_py3) [name@server ~]$ pip install jupyterlmod
(jupyter_py3) [name@server ~]$ jupyter labextension install jupyterlab-lmod
```

See the [JupyterHub](https://docs.alliancecan.ca/mediawiki/index.php?title=JupyterHub&oldid=156875) page for instructions on managing loaded modules in the JupyterLab interface.


### RStudio Server

RStudio Server allows you to develop R code in an RStudio environment, in a tab of your browser. There are a few differences with the [JupyterLab installation procedure](#installing-jupyterlab).

Before loading the `python` module and before creating a new virtual environment, load the `rstudio-server` module.

```bash
[name@server ~]$ module load rstudio-server python
```

Once JupyterLab is installed in the new virtual environment, install the Jupyter RSession proxy.

```bash
(jupyter_py3) [name@server ~]$ pip install --no-index jupyter-rsession-proxy
```

All other configuration and usage steps are the same. You should see an RStudio application under the Launcher tab.


## Using Your Installation


### Activating the Environment

Make sure the Python virtual environment where you installed JupyterLab is activated. For example, when you connect to the cluster, you must activate it again with:

```bash
[name@server ~]$ source $HOME/jupyter_py3/bin/activate
```

To verify that your environment is ready, you can get a list of installed `jupyter*` packages with the command:

```bash
(jupyter_py3) [name@server ~]$ pip freeze | grep jupyter
```


### Launching JupyterLab

To start a JupyterLab server, submit an interactive job with `salloc`. Adjust the parameters according to your needs. For more information, see [Running jobs](https://docs.alliancecan.ca/mediawiki/index.php?title=Running_jobs&oldid=156875).

```bash
(jupyter_py3) [name@server ~]$ salloc --time=1:0:0 --ntasks=1 --cpus-per-task=2 --mem-per-cpu=1024M --account=def-yourpi srun $VIRTUAL_ENV/bin/jupyterlab.sh
...
[I 2021-12-06 10:37:14.262 ServerApp] jupyterlab | extension was successfully linked.
...
[I 2021-12-06 10:37:39.259 ServerApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 2021-12-06 10:37:39.356 ServerApp]
To access the server, open this file in a browser:
file:///home/name/.local/share/jupyter/runtime/jpserver-198146-open.html
Or copy and paste one of these URLs:
http://node_name.int.cluster.computecanada.ca:8888/lab?token=101c3688298e78ab554ef86d93a196deaf5bcd2728fad4eb
or http://127.0.0.1:8888/lab?token=101c3688298e78ab554ef86d93a196deaf5bcd2728fad4eb
```


### Connecting to JupyterLab

To access the JupyterLab server on a compute node from your web browser, you must create an SSH tunnel from your computer to the cluster since compute nodes are not directly accessible from the internet.


#### On Linux or macOS

We recommend using the Python package `sshuttle`.

On your computer, open a new terminal window and create the SSH tunnel with the `sshuttle` command where you will replace `<username>` with the username for your account with the Alliance and `<cluster>` with the cluster on which you launched JupyterLab.

```bash
[name@local ~]$ sshuttle --dns -Nr <username>@<cluster>.computecanada.ca
```

Copy and paste the first HTTP address into your web browser; in the `salloc` example above, this would be `http://node_name.int.cluster.computecanada.ca:8888/lab?token=101c3688298e78ab554ef86d93a196deaf5bcd2728fad4eb`.


#### On Windows

To create an SSH tunnel from Windows, use MobaXTerm or any terminal that supports the `ssh` command.

Once JupyterLab is launched on a compute node (see [Launching JupyterLab](#launching-jupyterlab)), you can extract the `hostname:port` and `token` from the first HTTP address provided, for example `http://node_name.int.cluster.computecanada.ca:8888/lab?token=101c368829...2728fad4eb`

```
       └────────────────────┬────────────────────┘           └──────────┬──────────┘
                      hostname:port                                   token
```

Open a new Terminal tab in MobaXTerm. In the following command, replace `<hostname:port>` with the corresponding value (see image above); replace `<username>` with the username for your account with the Alliance; replace `<cluster>` with the cluster on which you launched JupyterLab.

```bash
[name@local ~]$ ssh -L 8888:<hostname:port> <username>@<cluster>.computecanada.ca
```

Open your web browser and go to the following address, where `<token>` should be replaced with the alphanumeric value from the address shown above.

```
http://localhost:8888/?token=<token>
```


### Closing JupyterLab

To stop the JupyterLab server before the end of the execution time, press `CTRL-C` twice in the terminal where the interactive job was launched.

If you used MobaXterm to create an SSH tunnel, press `Ctrl-D` to close the tunnel.


## Adding Kernels

It is possible to add kernels for other programming languages, for a different version of Python, or for a persistent virtual environment that has all the necessary packages and libraries for your project. For more information, see [Making kernels for Jupyter](https://jupyter.org/documentation).

Installing a new kernel is done in two steps:

1. Installing the packages that allow the language interpreter to communicate with the Jupyter interface.
2. Creating a file that tells JupyterLab how to start a communication channel with the language interpreter. This kernel spec file is saved in a subdirectory of `~/.local/share/jupyter/kernels`.

The next sections present examples of kernel installation procedures.


### Julia Kernel

**Prerequisites:**

The configuration of a Julia kernel depends on a Python virtual environment and a `kernels` directory. If you don't have these dependencies, make sure to follow the first few instructions in the [Python Kernel](#python-kernel) section below (a Python kernel is not required).

Since installing Julia packages requires internet access, configuring a Julia kernel must be done at the command prompt on a login node.

Once the Python virtual environment is available and activated, you can configure the Julia kernel. Load the Julia module.

```bash
(jupyter_py3) [name@server ~]$ module load julia
```

Install IJulia.

```bash
(jupyter_py3) [name@server ~]$ echo -e 'using Pkg\nPkg.add("IJulia")' | julia
```

**Important:** Before using the Julia kernel, start or restart a new JupyterLab session.

For more information, see the [IJulia documentation](https://github.com/JuliaLang/IJulia.jl).


### Installing Other Julia Packages

As with the installation procedure above, Julia packages must be installed from a login node, but the Python virtual environment can remain deactivated. Make sure the same Julia module is loaded.

```bash
[name@server ~]$ module load julia
```

Install the necessary packages, for example `Glob`.

```bash
[name@server ~]$ echo -e 'using Pkg\nPkg.add("Glob")' | julia
```

The newly installed Julia packages should be used in a notebook run by the Julia kernel.


### Python Kernel

In a terminal with an active session on a remote server, you can configure a Python virtual environment with all the necessary Python modules and a Python kernel adapted to JupyterLab.

The simplest configuration of Jupyter in a new Python virtual environment is as follows:

If you don't already have a Python virtual environment, create one and activate it. Start from a clean Bash environment (this is only necessary if you are using the Jupyter Terminal via JupyterHub to create and configure the Python kernel).

```bash
[name@server ~]$ env -i HOME=$HOME bash -l
```

Load a Python module.

```bash
[name@server ~]$ module load python
```

Create a new Python virtual environment.

```bash
[name@server ~]$ virtualenv --no-download $HOME/jupyter_py3
```

Activate the new virtual environment.

```bash
[name@server ~]$ source $HOME/jupyter_py3/bin/activate
```

Create the common `kernels` directory which is used by all kernels you want to install.

```bash
(jupyter_py3) [name@server ~]$ mkdir -p ~/.local/share/jupyter/kernels
```

Finally, install the Python kernel. Install the `ipykernel` library.

```bash
(jupyter_py3) [name@server ~]$ pip install --no-index ipykernel
```

Generate the kernel spec file. Replace `<unique_name>` with a name specific to your kernel.

```bash
(jupyter_py3) [name@server ~]$ python -m ipykernel install --user --name <unique_name> --display-name "Python 3.x Kernel"
```

**Important:** Before using the Python kernel, start or restart a new JupyterLab session.

For more information, see the [IPython kernel documentation](https://ipython.readthedocs.io/en/stable/install/kernel_install.html).


### Installing Other Python Libraries

Depending on the Python virtual environment configured in the previous section: Jupyter Terminal via JupyterHub, make sure the Python virtual environment is activated and is in a clean Bash environment. See the section above for details. Install a library that would be required, for example `numpy`.

```bash
(jupyter_py3) [name@server ~]$ pip install --no-index numpy
```

You can now import Python libraries in a notebook run by the Python 3.x Kernel.


### R Kernel

**Prerequisites:**

The configuration of an R kernel depends on a Python virtual environment and a `kernels` directory. If you don't have these dependencies, make sure to follow the first few instructions in the [Python Kernel](#python-kernel) section above (a Python kernel is not required).

Since installing R packages requires access to CRAN, configuring an R kernel must be done at the command prompt on a login node.

Once the Python virtual environment is available and activated, you can configure the R kernel. Load an R module.

```bash
(jupyter_py3) [name@server ~]$ module load r/4.1
```

Install the R kernel dependencies, namely `crayon`, `pbdZMQ`, and `devtools`; this could take up to 10 minutes and the packages should be installed in a local directory such as `~/R/x86_64-pc-linux-gnu-library/4.1`.

```bash
(jupyter_py3) [name@server ~]$ R --no-save
> install.packages(c('crayon', 'pbdZMQ', 'devtools'), repos='http://cran.us.r-project.org')
> devtools::install_github(paste0('IRkernel/', c('repr', 'IRdisplay', 'IRkernel')))
> IRkernel::installspec()
```

**Important:** Before using the R kernel, start or restart a new JupyterLab session.

For more information, see the [IRkernel documentation](https://irkernel.github.io/).


### Installing Other R Packages

Installing R packages cannot be done from notebooks because there is no access to CRAN. As in the installation procedure above, R packages must be installed on a login node, but the Python virtual environment can remain deactivated. Make sure the same R module is loaded.

```bash
[name@server ~]$ module load r/4.1
```

Start the R interpreter and install the required packages. Here is an example with `doParallel`:

```bash
[name@server ~]$ R --no-save
> install.packages('doParallel', repos='http://cran.us.r-project.org')
```

The newly installed R packages should already be usable in a notebook run by the R kernel.


## Running Notebooks as Python Scripts

For longer tasks or analyses, submit an interactive job.  You will need to convert the notebook to a Python script, create the script, and submit it.

1. On a login node, create and activate a virtual environment, then install `nbconvert` if it's not already installed.

```bash
(venv) [name@server ~]$ pip install --no-index nbconvert
```

2. Convert the notebook(s) to Python scripts with:

```bash
(venv) [name@server ~]$ jupyter nbconvert --to python mynotebook.ipynb
```

3. Create the script and submit the job. In the submission script, run the converted notebook with `python mynotebook.py`. Submit your non-interactive job with:

```bash
[name@server ~]$ sbatch my-submit.sh
```


## References

1.  See [https://jupyter.org/about.html](https://jupyter.org/about.html).
2.  See [https://jupyter.org/](https://jupyter.org/).


<a name="references"></a>
[1]: https://jupyter.org/about.html
[2]: https://jupyter.org/

