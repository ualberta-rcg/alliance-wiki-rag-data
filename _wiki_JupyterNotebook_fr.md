# Jupyter Notebook

This page is a translated version of the page JupyterNotebook and the translation is 100% complete.

Other languages: English, français

## Advanced material

This page is for advanced users. Please see [JupyterHub](JupyterHub) instead.


## Contents

1. Introduction
2. Installation
3. Installer des modules d'extension
    * Jupyter Lmod
    * Services web mandataires (proxy)
        * Exemple
    * RStudio Launcher
4. Activer l'environnement
    * RStudio Server (optionnel)
5. Lancer Jupyter Notebook
6. Se connecter à Jupyter Notebook
    * Sous Linux ou macOS X
    * Sous Windows
7. Fermer Jupyter Notebook
8. Ajouter des noyaux (kernels)
    * Julia
    * Python
    * R
9. Références


## Introduction

Project Jupyter is a non-profit, open-source project whose mission is to serve interactive scientific computing and data science. Initiated in 2014 as part of the IPython Project, the scope of Project Jupyter extends to several other programming languages. [1]

The Jupyter Notebook web application makes it possible to create and share documents containing code, equations, visualizations, and text. [2]

Jupyter Notebook works on a compute node or a login node (not recommended). In the case of the login node, various limits are imposed on both the user and the processes, and applications are sometimes terminated when they use too much CPU time or memory. In the case of the compute node, the task is submitted with the specification of the number of CPUs or GPUs to use, the amount of memory, and the execution time. The following instructions concern the submission of a Jupyter Notebook task.

Other information: Jupyter Notebook is not the latest Jupyter interface; we suggest installing JupyterLab instead. To use a pre-configured Jupyter environment, see the Jupyter page.


## Installation

These instructions allow you to install Jupyter Notebook with the `pip` command in a Python virtual environment in your home directory. The instructions are valid for Python version 3.6, but you can install the application for other versions by loading the appropriate Python module.

Load the Python module.

```bash
[name@server ~]$ module load python/3.7
```

Create a new Python virtual environment.

```bash
[name@server ~]$ virtualenv $HOME/jupyter_py3
```

Activate your new Python virtual environment.

```bash
[name@server ~]$ source $HOME/jupyter_py3/bin/activate
```

Install Jupyter Notebook in your new Python virtual environment.

```bash
(jupyter_py3)_[name@server ~]$ pip install --no-index --upgrade pip
(jupyter_py3)_[name@server ~]$ pip install --no-index jupyter
```

In your new virtual environment, create a script (wrapper) to launch Jupyter Notebook.

```bash
(jupyter_py3)_[name@server ~]$ echo -e '#!/bin/bash\nexport JUPYTER_RUNTIME_DIR=$SLURM_TMPDIR/jupyter\njupyter notebook --ip $(hostname -f) --no-browser' > $VIRTUAL_ENV/bin/notebook.sh
```

Finally, make the script executable.

```bash
(jupyter_py3)_[name@server ~]$ chmod u+x $VIRTUAL_ENV/bin/notebook.sh
```


## Installer des modules d'extension

Extensions add functionality and can modify the application's user interface.


### Jupyter Lmod

Jupyter Lmod is an extension that allows you to interact with environment modules before launching kernels. It uses the Lmod Python interface to perform module-related tasks such as loading, unloading, saving collections, etc.

```bash
(jupyter_py3)_[name@server ~]$ pip install jupyterlmod
(jupyter_py3)_[name@server ~]$ jupyter nbextension install --py jupyterlmod --sys-prefix
(jupyter_py3)_[name@server ~]$ jupyter nbextension enable --py jupyterlmod --sys-prefix
(jupyter_py3)_[name@server ~]$ jupyter serverextension enable --py jupyterlmod --sys-prefix
```


### Services web mandataires (proxy)

`nbserverproxy` allows access to proxy web services started in a Jupyter server. This is useful in the case of web services that only listen on a local server port, for example, TensorBoard.

```bash
(jupyter_py3)_[name@server ~]$ pip install nbserverproxy
(jupyter_py3)_[name@server ~]$ jupyter serverextension enable --py nbserverproxy --sys-prefix
```

#### Exemple

With Jupyter, a web service is started via Terminal in the New dropdown list.

```bash
[name@server ~]$ tensorboard --port=8008
```

The service is available via `/proxy/` at `https://address.of.notebook.server/user/theuser/proxy/8008`.


### RStudio Launcher

Jupyter Notebook can start an RStudio session that uses the Jupyter Notebook token authentication system. RStudio Launcher creates the RStudio Session option in the New dropdown list of Jupyter Notebook.

**Note:** The following procedure only works with StdEnv/2016.4 and StdEnv/2018.3 software environments.

```bash
(jupyter_py3)_[name@server ~]$ pip install nbserverproxy
(jupyter_py3)_[name@server ~]$ pip install https://github.com/jupyterhub/nbrsessionproxy/archive/v0.8.0.zip
(jupyter_py3)_[name@server ~]$ jupyter serverextension enable --py nbserverproxy --sys-prefix
(jupyter_py3)_[name@server ~]$ jupyter nbextension install --py nbrsessionproxy --sys-prefix
(jupyter_py3)_[name@server ~]$ jupyter nbextension enable --py nbrsessionproxy --sys-prefix
(jupyter_py3)_[name@server ~]$ jupyter serverextension enable --py nbrsessionproxy --sys-prefix
```


## Activer l'environnement

Once Jupyter Notebook is installed, you only need to reload the Python module associated with your environment when you connect to the cluster.

```bash
[name@server ~]$ module load python/3.7
```

Then activate the virtual environment in which Jupyter Notebook is installed.

```bash
[name@server ~]$ source $HOME/jupyter_py3/bin/activate
```

### RStudio Server (optionnel)

To use RStudio Launcher, load the RStudio Server module.

```bash
(jupyter_py3)_[name@server ~]$ module load rstudio-server
```


## Lancer Jupyter Notebook

To launch the application, submit an interactive task. Adjust the parameters as needed. For more information, see Exécuter des tâches.

```bash
(jupyter_py3)_[name@server ~]$ salloc --time=1:0:0 --ntasks=1 --cpus-per-task=2 --mem-per-cpu=1024M --account=def-yourpi srun $VIRTUAL_ENV/bin/notebook.sh
```

```
salloc: Granted job allocation 1422754
salloc: Waiting for resource configuration
salloc: Nodes cdr544 are ready for job
[I 14:07:08.661 NotebookApp] Serving notebooks from local directory: /home/fafor10
[I 14:07:08.662 NotebookApp] 0 active kernels
[I 14:07:08.662 NotebookApp] The Jupyter Notebook is running at:
[I 14:07:08.663 NotebookApp] http://cdr544.int.cedar.computecanada.ca:8888/?token=7ed7059fad64446f837567e32af8d20efa72e72476eb72ca
[I 14:07:08.663 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 14:07:08.669 NotebookApp]
Copy/paste this URL into your browser when you connect for the first time,
to login with a token:
http://cdr544.int.cedar.computecanada.ca:8888/?token=7ed7059fad64446f837567e3
```


## Se connecter à Jupyter Notebook

Since compute nodes are not directly accessible via the Internet, you must create an SSH tunnel between the cluster and your workstation so that your web browser can access Jupyter Notebook running on a compute node.


### Sous Linux ou macOS X

We recommend the Python package `sshuttle`.

On your workstation, open a new terminal window and run the `sshuttle` command to create the tunnel.

```bash
[name@my_computer ~]$ sshuttle --dns -Nr <username>@<cluster>.computecanada.ca
```

In the previous command, replace `<username>` with your username and `<cluster>` with the cluster to which you connected to launch Jupyter Notebook.

Then copy-paste the URL address into your browser. With the previous example, the result would be `http://cdr544.int.cedar.computecanada.ca:8888/?token=7ed7059fad64446f837567e3`.


### Sous Windows

To create an SSH tunnel, use MobaXTerm as follows, which also works with Unix (macOS, Linux, etc.).

In MobaXTerm, open a first Terminal tab (session 1) and connect to a cluster. Then follow the instructions in the Launching Jupyter Notebook section above. The following URL should appear:

`http://cdr544.int.cedar.computecanada.ca:8888/?token= 7ed7059fad64446f837567e3`

In MobaXTerm, open a second Terminal tab (session 2). In the following command, replace `<nom du serveur:port>` with the corresponding value in the URL obtained in session 1 (see the previous image); replace `<username>` with your username; and replace `<cluster>` with the cluster to which you connected in session 1. Run the command.

```bash
[name@my_computer ~]$ ssh -L 8888:<nom du serveur:port> <username>@<cluster>.computecanada.ca
```

Through your browser, go to `http://localhost:8888/?token=<jeton>`. Replace `<jeton>` with the value obtained in session 1.


## Fermer Jupyter Notebook

To close the Jupyter Notebook server before the end of the execution time, press CTRL-C twice in the terminal where the interactive task was launched.

If the tunnel was created with MobaXTerm, press CTRL-D in session 2 to close the tunnel.


## Ajouter des noyaux (kernels)

It is possible to add kernels for other programming languages or for versions of Python different from the one in which Jupyter Notebook is running. For more information, see Making kernels for Jupyter.

The installation is done in two steps:

1. Installation of the packages allowing the interpreter to communicate with Jupyter Notebook.
2. Creation of the file so that Jupyter Notebook can create a communication channel with the interpreter: this is the kernel configuration file.

Each of the kernel configuration files must be created in its own subdirectory in a directory of your home directory via the path `~/.local/share/jupyter/kernels`. Jupyter Notebook does not create this file; in all cases, the first step is to create it with the command:

```bash
[name@server ~]$ mkdir -p ~/.local/share/jupyter/kernels
```

The next sections present examples of kernel installation procedures.


### Julia

Load the Julia module.

```bash
[name@server ~]$ module load julia
```

Activate the Jupyter Notebook virtual environment.

```bash
[name@server ~]$ source $HOME/jupyter_py3/bin/activate
```

Install IJulia.

```bash
(jupyter_py3)_[name@server ~]$ echo 'Pkg.add("IJulia")' | julia
```

For more information, see the IJulia documentation.


### Python

Load the Python module.

```bash
[name@server ~]$ module load python/3.5
```

Create a new Python environment.

```bash
[name@server ~]$ virtualenv $HOME/jupyter_py3.5
```

Activate the new Python environment.

```bash
[name@server ~]$ source $HOME/jupyter_py3.5/bin/activate
```

Install the `ipykernel` library.

```bash
(jupyter_py3.5)_[name@server ~]$ pip install ipykernel
```

Generate the kernel configuration file. Replace `<unique_name>` with a unique name for your kernel.

```bash
(jupyter_py3.5)_[name@server ~]$ python -m ipykernel install --user --name <unique_name> --display-name "Python 3.5 Kernel"
```

Deactivate the virtual environment.

```bash
(jupyter_py3.5)_[name@server ~]$ deactivate
```

For more information, see the ipykernel documentation.


### R

Load the R module.

```bash
[name@server ~]$ module load r
```

Activate the Jupyter Notebook virtual environment.

```bash
[name@server ~]$ source $HOME/jupyter_py3/bin/activate
```

Install the kernel dependencies.

```bash
(jupyter_py3)_[name@server ~]$ R -e "install.packages(c('crayon', 'pbdZMQ', 'devtools'), repos='http://cran.us.r-project.org')"
```

Install the R kernel.

```bash
(jupyter_py3)_[name@server ~]$ R -e "devtools::install_github(paste0('IRkernel/', c('repr', 'IRdisplay', 'IRkernel')))"
```

Install the R kernel configuration file.

```bash
(jupyter_py3)_[name@server ~]$ R -e "IRkernel::installspec()"
```

For more information, see the IRKernel documentation.


## Références

[1] http://jupyter.org/about.html
[2] http://www.jupyter.org/

Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=JupyterNotebook/fr&oldid=164559](https://docs.alliancecan.ca/mediawiki/index.php?title=JupyterNotebook/fr&oldid=164559)"
