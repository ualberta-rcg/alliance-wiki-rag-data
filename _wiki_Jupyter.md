# The Jupyter Vocabulary and Related Wiki Pages

## JupyterLab

A web portal with a modern interface for managing and running applications, as well as rendering notebook files of various kernels.  For more details:

* **JupyterLab via JupyterHub:** A pre-installed JupyterLab environment, with a default Python kernel and access to software modules.
* **JupyterLab from a virtual environment:** A self-made environment to be launched by a Slurm job.


## Jupyter Notebook

An older web portal for managing and running applications, as well as rendering notebook files of various kernels. For more details:

* **Jupyter Notebook via JupyterHub:** A pre-installed Jupyter Notebook environment, with a default Python kernel and access to software modules.
* **Jupyter Notebook from a virtual environment:** A self-made environment to be launched by a Slurm job.


## Kernel

The active service behind the web interface.  There are:

* Notebook kernels (e.g., Python, R, Julia)
* Application kernels (e.g., RStudio, VSCode)


## Notebook

A page of executable cells of code and formatted text:

* **IPython notebooks:** A notebook executed by a Python kernel, and has some IPython interactive special commands that are not supported by a regular Python shell.


## Jupyter

Jupyter: An implementation of web applications and notebook rendering.  Google Colab would be another implementation of the same kind of environment.


## Jupyter Application

Like a regular application, but is displayed in a separate web browser tab. The application has access to the data stored remotely on the server, and the heavy computations are also handled by the remote server.


## JupyterHub

A web server hosting Jupyter portals and kernels.

