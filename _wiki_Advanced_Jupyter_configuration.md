# Advanced Jupyter Configuration

## Introduction

JupyterLab and notebooks are intended for short, interactive tasks such as testing, debugging, or quickly visualizing data (a few minutes).  Longer analyses should be run as non-interactive jobs (using `sbatch`).

See also how to run notebooks as Python scripts below.


**Project Jupyter:** "a non-profit, open-source project, born out of the IPython Project in 2014 as it evolved to support interactive data science and scientific computing across all programming languages." [1]

**JupyterLab:** "a web-based interactive development environment for notebooks, code, and data. Its flexible interface allows users to configure and arrange workflows in data science, scientific computing, computational journalism, and machine learning. A modular design allows for extensions that expand and enrich functionality." [2]

A JupyterLab server should only run on a compute node or a cloud instance; cluster login nodes are unsuitable due to various limits that can halt applications consuming excessive CPU time or memory. When using a compute node, reserve compute resources by submitting a job requesting a specific number of CPUs (and optionally GPUs), memory amount, and runtime.

This page details configuring and submitting a JupyterLab job on any national cluster. For a preconfigured Jupyter environment, see the [Jupyter](link-to-jupyter-page) page.


## Installing JupyterLab

These instructions install JupyterLab using the `pip` command within a Python virtual environment:

1.  If you don't have a Python virtual environment, create one. Then, activate it:

    Load a Python module, either the default one (as shown below) or a specific version (see available versions with `module avail python`):

    ```bash
    [name@server ~]$ module load python
    ```

    If you intend to use RStudio Server, load `rstudio-server` first:

    ```bash
    [name@server ~]$ module load rstudio-server python
    ```

2. Create a new Python virtual environment:

    ```bash
    [name@server ~]$ virtualenv --no-download $HOME/jupyter_py3
    ```

3. Activate your newly created Python virtual environment:

    ```bash
    [name@server ~]$ source $HOME/jupyter_py3/bin/activate
    ```

4. Install JupyterLab in your new virtual environment (note: this takes a few minutes):

    ```bash
    (jupyter_py3) [name@server ~]$ pip install --no-index --upgrade pip
    (jupyter_py3) [name@server ~]$ pip install --no-index jupyterlab
    ```

5. In the virtual environment, create a wrapper script that launches JupyterLab:

    ```bash
    (jupyter_py3) [name@server ~]$ echo -e '#!/bin/bash\nunset XDG_RUNTIME_DIR\njupyter lab --ip $(hostname -f) --no-browser' > $VIRTUAL_ENV/bin/jupyterlab.sh
    ```

6. Make the script executable:

    ```bash
    (jupyter_py3) [name@server ~]$ chmod u+x $VIRTUAL_ENV/bin/jupyterlab.sh
    ```


## Installing extensions

Extensions add functionalities and modify JupyterLab's user interface.

### Jupyter Lmod

Jupyter Lmod is an extension allowing interaction with environment modules before launching kernels. It uses Lmod's Python interface for module-related tasks (loading, unloading, saving collections, etc.).

These commands install and enable the Jupyter Lmod extension (note: the third command takes a few minutes):

```bash
(jupyter_py3) [name@server ~]$ module load nodejs
(jupyter_py3) [name@server ~]$ pip install jupyterlmod
(jupyter_py3) [name@server ~]$ jupyter labextension install jupyterlab-lmod
```

Instructions for managing loaded software modules in the JupyterLab interface are on the [JupyterHub](link-to-jupyterhub-page) page.


### RStudio Server

RStudio Server lets you develop R code in an RStudio environment within your web browser's separate tab.  Based on the above JupyterLab installation, there are a few differences:

1. Load the `rstudio-server` module *before* the `python` module and before creating a virtual environment:

   ```bash
   [name@server ~]$ module load rstudio-server python
   ```

2. Once JupyterLab is installed in the new virtual environment, install the Jupyter RSession proxy:

   ```bash
   (jupyter_py3) [name@server ~]$ pip install --no-index jupyter-rsession-proxy
   ```

All other configuration and usage steps remain the same. In JupyterLab, you should see an RStudio application in the Launcher tab.


## Using your installation

### Activating the environment

Ensure the Python virtual environment with JupyterLab is activated.  Upon logging onto the cluster, reactivate it:

```bash
[name@server ~]$ source $HOME/jupyter_py3/bin/activate
```

Verify your environment by listing installed `jupyter*` packages:

```bash
(jupyter_py3) [name@server ~]$ pip freeze | grep jupyter
```


### Starting JupyterLab

Start a JupyterLab server by submitting an interactive job with `salloc`. Adjust parameters as needed. See [Running jobs](link-to-running-jobs-page) for more information.

```bash
(jupyter_py3) [name@server ~]$ salloc --time=1:0:0 --ntasks=1 --cpus-per-task=2 --mem-per-cpu=1024M --account=def-yourpi srun $VIRTUAL_ENV/bin/jupyterlab.sh
```

(Example output will follow, including URLs to access the JupyterLab server)


### Connecting to JupyterLab

To access JupyterLab running on a compute node from your web browser, create an SSH tunnel from your computer through the cluster, as compute nodes aren't directly internet-accessible.

#### From Linux or macOS

Use the `sshuttle` Python package. Open a new terminal window and create the SSH tunnel:

```bash
[name@local ~]$ sshuttle --dns -Nr <username>@<cluster>.computecanada.ca
```

Replace `<username>` with your Alliance account username and `<cluster>` with the cluster where you launched JupyterLab. Then, paste the first provided HTTP address into your web browser.


#### From Windows

Create an SSH tunnel from Windows using MobaXTerm (or any terminal supporting the `ssh` command).

After launching JupyterLab on a compute node (see Starting JupyterLab), extract the `hostname:port` and `token` from the first HTTP address:

```
http://node_name.int.cluster.computecanada.ca:8888/lab?token=101c368829...2728fad4eb
       └────────────────────┬────────────────────┘           └──────────┬──────────┘
                      hostname:port                                   token
```

Open a new MobaXTerm terminal tab and run this command, replacing `<hostname:port>`, `<username>`, and `<cluster>` appropriately:

```bash
[name@local ~]$ ssh -L 8888:<hostname:port> <username>@<cluster>.computecanada.ca
```

Access JupyterLab in your web browser at:

```
http://localhost:8888/?token=<token>
```


### Shutting down JupyterLab

Press `Ctrl-C` twice in the terminal that launched the interactive job to shut down the JupyterLab server before the walltime limit. If you used MobaXTerm for the SSH tunnel, press `Ctrl-D` to close the tunnel.


## Adding kernels

Add kernels for other programming languages, different Python versions, or persistent virtual environments with necessary packages and libraries for your project. See [Making kernels for Jupyter](link-to-making-kernels-page) for details.

Installing a new kernel involves two steps:

1.  Installing packages enabling the language interpreter to communicate with the Jupyter interface.
2.  Creating a kernel spec file (saved in a subfolder of `~/.local/share/jupyter/kernels`) telling JupyterLab how to initiate communication with the language interpreter.


The following sections provide kernel installation examples.


### Julia Kernel

**Prerequisites:** A Python virtual environment and a `kernels` folder. If missing, follow the initial instructions in the Python kernel section (note: no Python kernel is required).

Because installing Julia packages requires internet access, configure the Julia kernel in a remote shell session on a login node.

With a Python virtual environment available and activated:

1. Load the Julia module:

   ```bash
   (jupyter_py3) [name@server ~]$ module load julia
   ```

2. Install IJulia:

   ```bash
   (jupyter_py3) [name@server ~]$ echo -e 'using Pkg\nPkg.add("IJulia")' | julia
   ```

**Important:** Start or restart a new JupyterLab session before using the Julia kernel. See the [IJulia documentation](link-to-ijulia-docs) for more information.


#### Installing more Julia packages

Install Julia packages from a login node (the Python virtual environment can be deactivated):

1. Load the Julia module:

   ```bash
   [name@server ~]$ module load julia
   ```

2. Install any required package (e.g., Glob):

   ```bash
   [name@server ~]$ echo -e 'using Pkg\nPkg.add("Glob")' | julia
   ```

The newly installed Julia packages should be usable in a notebook run by the Julia kernel.


### Python kernel

Configure a Python virtual environment with required Python modules and a custom Python kernel for JupyterLab.  These are the initial steps for the simplest Jupyter configuration in a new Python virtual environment:

1. If you don't have a Python virtual environment, create one and activate it.

2. Start from a clean Bash environment (only needed if using the Jupyter Terminal via JupyterHub for kernel creation and configuration):

   ```bash
   [name@server ~]$ env -i HOME=$HOME bash -l
   ```

3. Load a Python module:

   ```bash
   [name@server ~]$ module load python
   ```

4. Create a new Python virtual environment:

   ```bash
   [name@server ~]$ virtualenv --no-download $HOME/jupyter_py3
   ```

5. Activate your newly created Python virtual environment:

   ```bash
   [name@server ~]$ source $HOME/jupyter_py3/bin/activate
   ```

6. Create the `kernels` folder (used by all kernels):

   ```bash
   (jupyter_py3) [name@server ~]$ mkdir -p ~/.local/share/jupyter/kernels
   ```

7. Install the Python kernel:

   Install the `ipykernel` library:

   ```bash
   (jupyter_py3) [name@server ~]$ pip install --no-index ipykernel
   ```

   Generate the kernel spec file. Replace `<unique_name>` with a unique kernel identifier:

   ```bash
   (jupyter_py3) [name@server ~]$ python -m ipykernel install --user --name <unique_name> --display-name "Python 3.x Kernel"
   ```

**Important:** Start or restart a new JupyterLab session before using the Python kernel. See the [ipykernel documentation](link-to-ipykernel-docs) for more information.


#### Installing more Python libraries

Based on the Python virtual environment configured above:

If using the Jupyter Terminal via JupyterHub, ensure the activated Python virtual environment runs in a clean Bash environment (see above for details).

Install any required library (e.g., numpy):

```bash
(jupyter_py3) [name@server ~]$ pip install --no-index numpy
```

The newly installed Python libraries can be imported in any notebook using the "Python 3.x Kernel".


### R Kernel

**Prerequisites:** A Python virtual environment and a `kernels` folder. If missing, follow the initial instructions in the Python kernel section (note: no Python kernel is required).

Since installing R packages requires access to CRAN, configure the R kernel in a remote shell session on a login node.

With a Python virtual environment available and activated:

1. Load an R module:

   ```bash
   (jupyter_py3) [name@server ~]$ module load r/4.1
   ```

2. Install the R kernel dependencies (`crayon`, `pbdZMQ`, `devtools`) – this will take up to 10 minutes, and packages should be installed in a local directory like `~/R/x86_64-pc-linux-gnu-library/4.1`:

   ```r
   (jupyter_py3) [name@server ~]$ R --no-save
   > install.packages(c('crayon', 'pbdZMQ', 'devtools'), repos='http://cran.us.r-project.org')
   > devtools::install_github(paste0('IRkernel/', c('repr', 'IRdisplay', 'IRkernel')))
   > IRkernel::installspec()
   ```

**Important:** Start or restart a new JupyterLab session before using the R kernel. See the [IRkernel documentation](link-to-irkernel-docs) for more information.


#### Installing more R packages

R package installation cannot be done from notebooks due to lack of CRAN access.

Install R packages from a login node (the Python virtual environment can be deactivated):

1. Load the R module:

   ```bash
   [name@server ~]$ module load r/4.1
   ```

2. Start the R shell and install any required package (e.g., `doParallel`):

   ```r
   [name@server ~]$ R --no-save
   > install.packages('doParallel', repos='http://cran.us.r-project.org')
   ```

The newly installed R packages should be usable in a notebook run by the R kernel.


## Running notebooks as Python scripts

For longer runs or analyses, submit a non-interactive job. Convert your notebook to a Python script, create a submission script, and submit it.

1.  From the login node, create and activate a virtual environment, then install `nbconvert` if not already available:

    ```bash
    (venv) [name@server ~]$ pip install --no-index nbconvert
    ```

2.  Convert the notebook (or all notebooks) to Python scripts:

    ```bash
    (venv) [name@server ~]$ jupyter nbconvert --to python mynotebook.ipynb
    ```

3.  Create your submission script and submit your job. In your submission script, run your converted notebook with:

    ```bash
    python mynotebook.py
    ```

    Then submit your non-interactive job:

    ```bash
    [name@server ~]$ sbatch my-submit.sh
    ```


## References

[1] https://jupyter.org/about.html

[2] https://jupyter.org/


**(Remember to replace placeholder links like `link-to-jupyter-page`, etc., with actual links.)**
