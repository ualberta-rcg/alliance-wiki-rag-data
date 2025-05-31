# JupyterHub

Other languages: English, français

JupyterHub is the best way to serve Jupyter Notebook for multiple users. It can be used in a class of students, a corporate data science group, or a scientific research group.<sup>[1]</sup> JupyterHub provides a preconfigured version of JupyterLab and/or Jupyter Notebook; for more configuration options, please check the [Jupyter](link-to-jupyter-page) page.

## Running notebooks

JupyterLab and notebooks are meant for *short* interactive tasks such as testing, debugging, or quickly visualizing data (few minutes). Running longer analyses must be done in a *non-interactive job (sbatch)*.

See also how to run notebooks as Python scripts below.


## Contents

1. Alliance initiatives
    * JupyterHub on clusters
    * JupyterHub for universities and schools
2. Server options
    * Compute resources
    * User interface
3. JupyterLab
    * The JupyterLab interface
        * Menu bar on top
        * Tool selector on left
        * Applications area on right
        * Status bar at the bottom
    * Prebuilt applications
        * Command line interpreters
            * Julia console
            * Python console
            * Terminal
        * Available notebook kernels
            * Julia notebook
            * Python notebook
        * Other applications
            * OpenRefine
            * RStudio
            * VS Code
            * Desktop
    * Running notebooks as Python scripts
4. Possible error messages
5. References


## Alliance initiatives

Some regional initiatives offer access to computing resources through JupyterHub.

### JupyterHub on clusters

On the following clusters<sup>‡</sup>, use your Alliance username and password to connect to JupyterHub:

| JupyterHub | Comments |
|---|---|
| Béluga | Provides access to JupyterLab servers spawned through jobs on the Béluga cluster. |
| Cedar | Provides access to JupyterLab servers spawned through jobs on the Cedar cluster. |
| Narval | Provides access to JupyterLab servers spawned through jobs on the Narval cluster. |
| Niagara | Provides access to JupyterLab as one of the applications of the SciNet Open OnDemand portal. To learn more, see the [wiki page](link-to-wiki-page). |
| Graham | Provides access to JupyterLab servers spawned through jobs on the Graham cluster. |

<sup>‡</sup> Note that the compute nodes running the Jupyter kernels do not have internet access. This means that you can only transfer files from/to your own computer; you cannot download code or data from the internet (e.g., cannot do "git clone", cannot do "pip install" if the wheel is absent from our [wheelhouse](link-to-wheelhouse)). You may also have problems if your code performs downloads or uploads (e.g., in machine learning where downloading data from the code is often done).


### JupyterHub for universities and schools

The [Pacific Institute for the Mathematical Sciences](link-to-pacific-institute) in collaboration with the Alliance and [Cybera](link-to-cybera) offer cloud-based hubs to universities and schools. Each institution can have its own hub where users authenticate with their credentials from that institution. The hubs are hosted on Alliance [clouds](link-to-alliance-clouds) and are essentially for training purposes. Institutions interested in obtaining their own hub can visit [Syzygy](link-to-syzygy).


## Server options

### Compute resources

For example, *Server Options* available on Béluga's JupyterHub are:

*   **Account** to be used: any `def-*`, `rrg-*`, `rpp-*` or `ctb-*` account a user has access to
*   **Time (hours)** required for the session
*   **Number of (CPU) cores** that will be reserved on a single node
*   **Memory (MB)** limit for the entire session
*   **(Optional) GPU configuration**: at least one GPU


### User interface

While JupyterHub allows each user to use one Jupyter server at a time on each hub, there can be multiple options under *User interface*:

*   **Jupyter Notebook (classic interface):** Even though it offers many functionalities, the community is moving towards JupyterLab, which is a better platform that offers many more features.
*   **JupyterLab (modern interface):** This is the most recommended Jupyter user interface for interactive prototyping and data visualization.
*   **Terminal (for a single terminal only):** It gives access to a terminal connected to a remote account, which is comparable to connecting to a server through an SSH connection.

Note: JupyterHub could also have been configured to force a specific user interface. This is usually done for special events.


## JupyterLab

JupyterLab is the recommended general-purpose user interface to use on a JupyterHub. From a JupyterLab server, you can manage your remote files and folders, and you can launch Jupyter applications like a terminal, (Python 3) notebooks, RStudio, and a Linux desktop. You can add your own "kernels," which appear as application tiles described below. To configure such kernels, please see [Adding kernels](link-to-adding-kernels).


### The JupyterLab interface

When JupyterLab is ready to be used, the interface has multiple panels.

#### Menu bar on top

In the *File* menu:

*   **Hub Control Panel**: if you want to manually stop the JupyterLab server and the corresponding job on the cluster. This is useful when you want to start a new JupyterLab server with more or less resources.
*   **Log Out**: the JupyterHub session will end, which will also stop the JupyterLab server and the corresponding job on the cluster.

Most other menu items are related to notebooks and Jupyter applications.


#### Tool selector on left

*   **File Browser** (folder icon): This is where you can browse in your home, project, and scratch spaces. It is also possible to upload files.
*   **Running Terminals and Kernels** (stop icon): To stop kernel sessions and terminal sessions.
*   **Commands**
*   **Property Inspector**
*   **Open Tabs**: To navigate between application tabs. To close application tabs (the corresponding kernels remain active).
*   **Loaded modules and available modules**
*   **Software** (blue diamond sign): Alliance modules can be loaded and unloaded in the JupyterLab session. Depending on the modules loaded, icons directing to the corresponding *Jupyter applications* will appear in the *Launcher* tab. The search box can search for any *available module* and show the result in the *Available Modules* subpanel. Note: Some modules are hidden until their dependency is loaded: we recommend that you first look for a specific module with `module spider module_name` from a terminal. The next subpanel is the list of *Loaded Modules* in the whole JupyterLab session. Note: While `python` and `ipython-kernel` modules are loaded by default, additional modules must be loaded before launching some other applications or notebooks. For example: `scipy-stack`. The last subpanel is the list of *Available modules*, similar to the output of `module avail`. By clicking on a module's name, detailed information about the module is displayed. By clicking on the *Load* link, the module will be loaded and added to the *Loaded Modules* list.


#### Applications area on right

The *Launcher* tab is open by default. It contains all available *Jupyter applications and notebooks*, depending on which modules are loaded.


#### Status bar at the bottom

By clicking on the icons, this brings you to the *Running Terminals and Kernels* tool.


### Prebuilt applications

JupyterLab offers access to a terminal, an IDE (Desktop), a Python console, and different options to create text and markdown files. This section presents only the main supported Jupyter applications that work with our software stack.


#### Command line interpreters

*   **Julia console launcher button**
*   **Python console launcher button**
*   **Terminal launcher button**


##### Julia console

To enable the *Julia 1.x* console launcher, an `ijulia-kernel` module needs to be loaded. When launched, a Julia interpreter is presented in a new JupyterLab tab.


##### Python console

The *Python 3.x* console launcher is available by default in a new JupyterLab session. When launched, a Python 3 interpreter is presented in a new JupyterLab tab.


##### Terminal

This application launcher will open a terminal in a new JupyterLab tab:

*   The terminal runs a (Bash) shell on the remote compute node without the need of an SSH connection.
*   Gives access to the remote filesystems (`/home`, `/project`, `/scratch`).
*   Allows running compute tasks.

The terminal allows copy-and-paste operations of text:

*   Copy operation: select the text, then press Ctrl+C. Note: Usually, Ctrl+C is used to send a SIGINT signal to a running process, or to cancel the current command. To get this behaviour in JupyterLab's terminal, click on the terminal to deselect any text before pressing Ctrl+C.
*   Paste operation: press Ctrl+V.


#### Available notebook kernels


##### Julia notebook

To enable the *Julia 1.x* notebook launcher, an `ijulia-kernel` module needs to be loaded. When launched, a Julia notebook is presented in a new JupyterLab tab.


##### Python notebook

Searching for `scipy-stack` modules: If any of the following scientific Python packages is required by your notebook, before you open this notebook, you must load the `scipy-stack` module from the JupyterLab *Softwares* tool: `ipython`, `ipython_genutils`, `ipykernel`, `ipyparallel`, `matplotlib`, `numpy`, `pandas`, `scipy`. Other notable packages are `Cycler`, `futures`, `jupyter_client`, `jupyter_core`, `mpmath`, `pathlib2`, `pexpect`, `pickleshare`, `ptyprocess`, `pyzmq`, `simplegeneric`, `sympy`, `tornado`, `traitlets`. And many more (click on the `scipy-stack` module to see all *Included extensions*).

Note: You may also install needed packages by running, for example, the following command inside a cell: `!pip install --no-index numpy`. For some packages (like `plotly`, for example), you may need to restart the notebook's kernel before importing the package. The installation of packages in the default Python kernel environment is temporary to the lifetime of the JupyterLab session; you will have to reinstall these packages the next time you start a new JupyterLab session. For a persistent Python environment, you must configure a *custom Python kernel*.

To open an existing Python notebook:

1.  Go back to the *File Browser*.
2.  Browse to the location of the *.ipynb file.
3.  Double-click on the *.ipynb file. This will open the Python notebook in a new JupyterLab tab. An IPython kernel will start running in the background for this notebook.

To open a new Python notebook in the current *File Browser* directory:

1.  Click on the *Python 3.x* launcher under the *Notebook* section.
2.  This will open a new Python 3 notebook in a new JupyterLab tab. A new IPython kernel will start running in the background for this notebook.


#### Other applications

*   **OpenRefine launcher button**
*   **RStudio launcher button**
*   **VS Code launcher button**
*   **Desktop launcher button**


##### OpenRefine

To enable the *OpenRefine* application launcher, an `openrefine` module needs to be loaded. Depending on the software environment version, the latest version of OpenRefine should be loaded:

*   With `StdEnv/2023`, no OpenRefine module is available as of August 2024; please load `StdEnv/2020` first.
*   With `StdEnv/2020`, load module: `openrefine/3.4.1`.

This *OpenRefine* launcher will open or reopen an OpenRefine interface in a new web browser tab. It is possible to reopen an active OpenRefine session after the web browser tab was closed. The OpenRefine session will end when the JupyterLab session ends.


##### RStudio

To enable the *RStudio* application launcher, load the module: `rstudio-server/4.3`. This *RStudio* launcher will open or reopen an RStudio interface in a new web browser tab. It is possible to reopen an active RStudio session after the web browser tab was closed. The RStudio session will end when the JupyterLab session ends. Note that simply quitting RStudio or closing the RStudio and JupyterHub tabs in your browser will not release the resources (CPU, memory, GPU) nor end the underlying Slurm job. Please end your session with the menu item *File \> Log Out* on the JupyterLab browser tab.


##### VS Code

To enable the *VS Code* (Visual Studio Code) application launcher, a `code-server` module needs to be loaded. Depending on the software environment version, the latest version of VS Code should be loaded:

*   With `StdEnv/2023`, load module: `code-server/4.92.2`.
*   With `StdEnv/2020`, load module: `code-server/3.12.0`.

This *VS Code* launcher will open or reopen the VS Code interface in a new web browser tab. For a new session, the *VS Code* session can take up to 3 minutes to complete its startup. It is possible to reopen an active VS Code session after the web browser tab was closed. The VS Code session will end when the JupyterLab session ends.


##### Desktop

This *Desktop* launcher will open or reopen a remote Linux desktop interface in a new web browser tab: This is equivalent to running a *VNC server on a compute node*, then creating an *SSH tunnel* and finally using a *VNC client*, but you need nothing of all this with JupyterLab! It is possible to reopen an active desktop session after the web browser tab was closed. The desktop session will end when the JupyterLab session ends.


### Running notebooks as Python scripts

1.  From the console, or in a new notebook cell, install `nbconvert`: `!pip install --no-index nbconvert`
2.  Convert your notebooks to Python scripts: `!jupyter nbconvert --to python my-current-notebook.ipynb`
3.  Create your *non-interactive submission script*, and submit it. In your submission script, run your converted notebook with: `python my-current-notebook.py` And submit your non-interactive job: `[name@server ~] $ sbatch my-submit.sh`


## Possible error messages

Most JupyterHub errors are caused by the underlying job scheduler which is either unresponsive or not able to find appropriate resources for your session. For example: *Spawn failed: Timeout*. When starting a new session, JupyterHub automatically submits on your behalf a new *interactive job* to the cluster. If the job does not start within five minutes, a "Timeout" error message is raised and the session is cancelled. Just like any interactive job on any cluster, a longer requested time can cause a longer wait time in the queue. Requesting a GPU or too many CPU cores can also cause a longer wait time. Make sure to request only the resources you need for your session. If you already have another interactive job on the same cluster, your Jupyter session will be waiting along with other regular batch jobs in the queue. If possible, stop or cancel any other interactive job before using JupyterHub. There may be just no resource available at the moment. Check the [status page](link-to-status-page) for any issue and try again later.


## References

[1] http://jupyterhub.readthedocs.io/en/latest/index.html


**(Remember to replace the bracketed placeholders like `link-to-jupyter-page` with the actual links.)**
