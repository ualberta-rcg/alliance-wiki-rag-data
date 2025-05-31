# Python

This page is a translated version of the page Python and the translation is 100% complete.

Other languages: [English](link-to-english-page), [français](current-page-url)


## Description

Python is an interpreted programming language whose design philosophy emphasizes code readability. Its syntax is simple and expressive, and its standard module library is very extensive.

The capabilities of the Python language can be extended using third-party packages. In general, we do not install third-party packages in the available software directory to simplify operations as much as possible; it is therefore your responsibility to install them. However, we provide several versions of the Python interpreter and the necessary tools for you to easily install the packages you need.

The following sections present the Python interpreter and explain how to install and use packages.


## Loading the Interpreter

### Default Version

A version is available when you connect to our clusters, but you will often need a different version, especially if you want to install packages. Find the Python version you need and load the appropriate module. If in doubt, you can use the latest available version.  `mpi4py` as a dependency of another package.


### Loading a Python Module

To find out which versions are available, use:

```bash
[name@server ~]$ module avail python
```

You can then load the version of your choice with the `module load` command. For example, to load Python 3.10:

```bash
[name@server ~]$ module load python/3.10
```


### Supported Versions

Generally, in the Python ecosystem, the transition to more modern versions is accelerating, and many packages only support the last few versions of Python 3.x. In our case, we only offer pre-built packages (Python wheels) for the three most recent versions available on our systems. Dependency problems will arise when you try to install these packages with older versions of Python. See the Troubleshooting section.

The following table shows the dates on which we stopped building wheels for Python versions.

| Version | Date       |
|---------|------------|
| 3.10    | 2022-02    |
| 3.9     |            |
| 3.8     |            |
| 3.7     |            |
| 3.6     | 2021-02    |
| 3.5     | 2020-02    |
| 2.7     | 2020-01    |


### SciPy Software Stack

In addition to the basic Python module, the SciPy package is also available as an environment module. The `scipy-stack` module includes:

* NumPy
* SciPy
* Matplotlib
* dateutil
* pytz
* IPython
* pyzmq
* tornado
* pandas
* Sympy
* nose

To use one of these packages, load a Python version, then:

```bash
module load scipy-stack
```

To list and see the version numbers of the packages contained in `scipy-stack`, run:

```bash
module spider scipy-stack/2020a
```

(replacing `2020a` with the version you want).


## Creating and Using a Virtual Environment

With each version of Python comes the `virtualenv` tool, which allows you to create virtual environments within which you can easily install your Python packages. These environments allow, for example, to install several versions of the same package, or to compartmentalize installations according to needs or experiments to be carried out. You would usually create your Python virtual environments in your `/home` directory or in one of your `/project` directories. For a third option, see the section "Creating a virtual environment in your tasks" below.


### Where to Create a Virtual Environment

Do not create your virtual environment in `$SCRATCH` because of the risk that it will be partially destroyed. See "Creating a virtual environment in your tasks" below.


To create a virtual environment, first select a Python version with:

```bash
module load python/X.Y.Z
```

as indicated above in "Loading a Python Module". If you want to use the packages listed in "SciPy Software Stack", also run:

```bash
module load scipy-stack/X.Y.Z
```

Then enter the next command, where `ENV` is the name of the directory for your new environment:

```bash
[name@server ~]$ virtualenv --no-download ~/ENV
```

Once the virtual environment is created, all you have to do is activate it with:

```bash
[name@server ~]$ source ~/ENV/bin/activate
```

You should also update `pip` in the environment:

```bash
[name@server ~]$ pip install --no-index --upgrade pip
```

To exit the virtual environment, simply enter the command:

```bash
(ENV) [name@server ~] deactivate
```

To reuse the virtual environment:

1. Load the same environment modules that you loaded when the virtual environment was created, i.e., `module load python scipy-stack`.
2. Activate the environment with `source ENV/bin/activate`.


### Installing Packages

Once you have loaded a virtual environment, you can run the `pip` command. This command supports the compilation and installation of most Python packages and their dependencies. Consult the [complete index of Python packages](link-to-package-index).

The available commands are explained in the [pip user manual](link-to-pip-manual). We mention here the most important commands by presenting an example of installing the NumPy package.

First, let's load the Python interpreter with:

```bash
[name@server ~]$ module load python/3.10
```

Then, let's activate the virtual environment created previously with the `virtualenv` command:

```bash
[name@server ~]$ source ~/ENV/bin/activate
```

Finally, we can install the latest stable version of NumPy with:

```bash
(ENV) [name@server ~] pip install numpy --no-index
```

The `pip` command can install packages from several sources, including PyPI and pre-built distribution packages called Python wheels. We provide Python wheels for several packages. In the example above, the `--no-index` option tells `pip` to not install from PyPI, but rather to only install from local source packages, i.e., our wheels.

If one of our wheels is available for a package you want, we strongly recommend using it with the `--no-index` option. Unlike PyPI packages, the wheels compiled by our staff avoid problems with missing or conflicting dependencies and are also optimized for our clusters and libraries. See "Available Wheels".

If you omit the `--no-index` option, `pip` will search for PyPI packages and local packages and use the most recent version. If this is from PyPI, it will be installed instead of ours and you may have problems. If you prefer to download a PyPI package rather than using a wheel, use the `--no-binary` option, which tells `pip` not to consider any pre-built packages; thus, the wheels distributed via PyPI will not be considered and the package will always be compiled from source.

To find out where the Python package installed by `pip` comes from, add the `-vvv` option. When installing multiple Python packages, it is best to install them in a single step, as `pip` can then resolve cross-dependencies.


### Creating a Virtual Environment in Your Tasks

Parallel file systems like those installed on our clusters are very efficient when it comes to reading or writing large amounts of data, but not for intensive use of small files. For this reason, launching software and loading libraries can be slow, which happens when launching Python and loading a virtual environment.

To counter this kind of slowdown, especially for single-node Python tasks, you can create your virtual environment within your task using the local disk of the compute node. It may seem unreasonable to recreate your environment for each of your tasks, but it is often faster and more efficient than using the parallel file system. You must create a virtualenv locally on each of the nodes used by the task since access to virtualenv is done per node. The following script is an example:


**File:** `submit_venv.sh`

```bash
#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --mem-per-cpu=1.5G      # increase as needed
#SBATCH --time=1:00:00
module load python/3.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r requirements.txt
python ...
```

where the `requirements.txt` file will have been created in a test environment. For example, to create an environment for TensorFlow, use the following commands on a login node:

```bash
[name@server ~]$ module load python/3.10
[name@server ~]$ ENVDIR=/tmp/$RANDOM
[name@server ~]$ virtualenv --no-download $ENVDIR
[name@server ~]$ source $ENVDIR/bin/activate
[name@server ~]$ pip install --no-index --upgrade pip
[name@server ~]$ pip install --no-index tensorflow
[name@server ~]$ pip freeze --local > requirements.txt
[name@server ~]$ deactivate
[name@server ~]$ rm -rf $ENVDIR
```

This produces the `requirements.txt` file whose content looks like this:


**File:** `requirements.txt`

```
absl_py==1.2.0+computecanada
astunparse==1.6.3+computecanada
cachetools==5.2.0+computecanada
certifi==2022.6.15+computecanada
charset_normalizer==2.1.0+computecanada
flatbuffers==1.12+computecanada
gast==0.4.0+computecanada
google-pasta==0.2.0+computecanada
google_auth==2.9.1+computecanada
google_auth_oauthlib==0.4.6+computecanada
grpcio==1.47.0+computecanada
h5py==3.6.0+computecanada
idna==3.3+computecanada
keras==2.9.0+computecanada
Keras-Preprocessing==1.1.2+computecanada
libclang==14.0.1+computecanada
Markdown==3.4.1+computecanada
numpy==1.23.0+computecanada
oauthlib==3.2.0+computecanada
opt-einsum==3.3.0+computecanada
packaging==21.3+computecanada
protobuf==3.19.4+computecanada
pyasn1==0.4.8+computecanada
pyasn1-modules==0.2.8+computecanada
pyparsing==3.0.9+computecanada
requests==2.28.1+computecanada
requests_oauthlib==1.3.1+computecanada
rsa==4.8+computecanada
six==1.16.0+computecanada
tensorboard==2.9.1+computecanada
tensorboard-data-server==0.6.1+computecanada
tensorboard_plugin_wit==1.8.1+computecanada
tensorflow==2.9.0+computecanada
tensorflow_estimator==2.9.0+computecanada
tensorflow_io_gcs_filesystem==0.23.1+computecanada
termcolor==1.1.0+computecanada
typing_extensions==4.3.0+computecanada
urllib3==1.26.11+computecanada
Werkzeug==2.1.2+computecanada
wrapt==1.13.3+computecanada
```

This file ensures that your environment can be reproduced for other tasks.

Note that the above directives require that all the packages you need are available in the Python wheels we provide. If this is not the case, you can pre-download them (see "Pre-downloading packages" below). If you believe that wheels should be provided, please contact [technical support](link-to-support).


### Creating a Virtual Environment in Your Tasks (Multiple Nodes)

For your scripts to use multiple nodes, each must have its own activated environment.

1. In your job submission script, create the virtual environment for each of the allocated nodes.

```bash
srun --ntasks $SLURM_NNODES --tasks-per-node=1 bash << EOF
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
EOF
```

2. Activate the virtual environment of the main node.

```bash
source $SLURM_TMPDIR/env/bin/activate;
```

3. Run the script with:

```bash
srun python myscript.py;
```


### Example (Multiple Nodes)

**File:** `submit-nnodes-venv.sh`

```bash
#!/bin/bash
#SBATCH --account=<your account>
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=2000M
module load StdEnv/2023 python/3.11 mpi4py
# create the virtual environment for each of the nodes.
srun --ntasks $SLURM_NNODES --tasks-per-node=1 bash << EOF
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
EOF
# activate only on the main node
source $SLURM_TMPDIR/env/bin/activate;
# srun exports the current environment which contains $VIRTUAL_ENV and $PATH variables
srun python myscript-mpi.py;
```


### Available Wheels

The currently available wheels are listed on the [Python Wheels](link-to-wheels-page) page. You can also use the `avail_wheels` command on the cluster.

By default, this command only shows:

1. The most recent version of a particular package, unless a particular version has been specified;
2. The versions compatible with the loaded Python module or the activated virtual environment; otherwise, all versions are displayed;
3. The versions compatible with the CPU architecture and the software environment (StdEnv) you are currently using.


#### Names

To list the wheels that contain `cdf` (case-insensitive) in their name, run:

```bash
[name@server ~]$ avail_wheels "*cdf*"
```

```
name        version python  arch
--------    --------- -------- -------
h5netcdf    0.7.4     py2,py3  generic
netCDF4     1.5.8     cp39    avx2
netCDF4     1.5.8     cp38    avx2
netCDF4     1.5.8     cp310   avx2
```

Or use the exact name, for example:

```bash
[name@server ~]$ avail_wheels numpy
```


#### Version

To list a particular version, you can use the same format as with `pip`:

```bash
[name@server ~]$ avail_wheels numpy==1.23
```

```
name    version python  arch
------  --------- -------- -------
numpy   1.23.0   cp39    generic
numpy   1.23.0   cp38    generic
numpy   1.23.0   cp310   generic
```

Or use the longer version, such as:

```bash
[name@server ~]$ avail_wheels numpy --version 1.23
```

With the `pip` format, you can use different operators: `==`, `<`, `>`, `~=`, `<=`, `>=`, `!=`. For example, to list previous versions:

```bash
[name@server ~]$ avail_wheels 'numpy<1.23'
```

```
name    version python  arch
------  --------- -------- -------
numpy   1.22.2   cp39    generic
numpy   1.22.2   cp38    generic
numpy   1.22.2   cp310   generic
```

And to list all available versions:

```bash
[name@server ~]$ avail_wheels "*cdf*" --all-version
```


#### Python

To list a particular Python version, run:

```bash
[name@server ~]$ avail_wheels 'numpy<1.23' --python 3.9
```

```
name    version python  arch
------  --------- -------- -------
numpy   1.22.2   cp39    generic
```

The `python` column shows the Python version for which the wheel is available, where `cp39` is used for `cpython 3.9`.


#### Requirements File

To find out if the available wheels include those indicated in the `requirements.txt` file, run:

```bash
[name@server ~]$ avail_wheels -r requirements.txt
```

```
name        version python  arch
---------    --------- -------- -------
packaging    21.3      py3     generic
tabulate     0.8.10    py3     generic
```

To list those that are not available, the command is:

```bash
[name@server ~]$ avail_wheels -r requirements.txt --not-available
```


### Pre-downloading Packages

The following procedure pre-downloads the `tensorboardX` package on a login node and installs it on a compute node:

1. Run `pip download --no-deps tensorboardX` to download the `tensorboardX-1.9-py2.py3-none-any.whl` (or similar) package to the working directory. The syntax for `pip download` is the same as that for `pip install`.

2. If the file name does not end with `none-any`, but with `linux_x86_64` or `manylinux*_x86_64`, it is possible that the wheel will not work correctly. Contact [technical support](link-to-support) to have us compile the wheel and make it available on our supercomputers.

3. At installation, use the file path: `pip install tensorboardX-1.9-py2.py3-none-any.whl`.


## Parallel Programming with the `multiprocessing` Module

Parallel programming with Python is an easy way to get results faster, which is usually accomplished using the `multiprocessing` module. The `Pool` class of this module is particularly interesting because it allows you to control the number of processes launched in parallel to execute the same calculation with multiple data. Suppose we want to calculate the cube of a list of numbers; the serial code would be similar to:


### With a Loop

**File:** `cubes_sequential.py`

```python
def cube(x):
    return x**3
data = [1, 2, 3, 4, 5, 6]
cubes = [cube(x) for x in data]
print(cubes)
```

### Using `map`

**File:** `cubes_sequential.py`

```python
def cube(x):
    return x**3
data = [1, 2, 3, 4, 5, 6]
cubes = list(map(cube,data))
print(cubes)
```

With the `Pool` class, the parallel code becomes:


### With a Loop

**File:** `cubes_parallel.py`

```python
import multiprocessing as mp

def cube(x):
    return x**3
pool = mp.Pool(processes=4)
data = [1, 2, 3, 4, 5, 6]
results = [pool.apply_async(cube, args=(x,)) for x in data]
cubes = [p.get() for p in results]
print(cubes)
```

### Using `map`

**File:** `cubes_parallel.py`

```python
import multiprocessing as mp

def cube(x):
    return x**3
pool = mp.Pool(processes=4)
data = [1, 2, 3, 4, 5, 6]
cubes = pool.map(cube, data)
print(cubes)
```

In the previous examples, however, we are limited to four processes. With a cluster, it is very important to use the cores that are allocated to the task. If the number of processes executed exceeds the number of cores requested for the task, the calculations will be slower and the compute node may be overloaded. If the number of processes executed is less than the number of cores requested, some cores will remain idle and resources will not be used optimally. Your code should call as many cores as the amount of resources requested from the scheduler. For example, to perform the same calculation on dozens of data or more, it would be sensible to use all the cores of a node. In this case, the job submission script would have the following header:


**File:** `submit.sh`

```bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
python cubes_parallel.py
```

The code would then be:


### With a Loop

**File:** `cubes_parallel.py`

```python
import multiprocessing as mp
import os

def cube(x):
    return x**3
ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK',default=1))
pool = mp.Pool(processes=ncpus)
data = [1, 2, 3, 4, 5, 6]
results = [pool.apply_async(cube, args=(x,)) for x in data]
cubes = [p.get() for p in results]
print(cubes)
```

### Using `map`

**File:** `cubes_parallel.py`

```python
import multiprocessing as mp
import os

def cube(x):
    return x**3
ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK',default=1))
pool = mp.Pool(processes=ncpus)
data = [1, 2, 3, 4, 5, 6]
cubes = pool.map(cube, data)
print(cubes)
```

Note that in this example, the `cube` function is itself sequential. It is possible that a function called from an external library like `numpy` is itself parallel. To distribute processes with the previous technique, first check if the called functions are themselves parallel and if so, you will need to control the number of threads they will use. If, as in the example, the functions use all the available cores (here 32) and you launch 32 processes, your code will be slower and the node may be overloaded.

Since the `multiprocessing` module can only use a single compute node, the performance gain is usually limited to the number of CPU cores on the node. If you want to exceed this limit and use multiple nodes, consider `mpi4py` or `PySpark`. There are [other parallelization methods](link-to-parallelization-methods), but they cannot all be used with our clusters. Remember, however, that quality code will always provide the best performance; so before parallelizing it, make sure your code is optimal. If you doubt the efficiency of your code, contact [technical support](link-to-support).


## Anaconda

See the page on [Anaconda](link-to-anaconda-page).


## Jupyter

See the page on [Jupyter Notebook](link-to-jupyter-page).


## Troubleshooting

### Frozen Script

With the `faulthandler` module, you can modify your script so that a trace of the origin of the problem is provided after a certain duration; see the information on the `faulthandler.dump_traceback_later(timeout, repeat=False, file=sys.stderr, exit=False)` command.

You can also inspect a Python process during the execution of a task without having to modify it beforehand with `py-spy`:

1. Install `py-spy` in a virtual environment in your `/home` directory.
2. Connect to a running task with `srun --pty --jobid JOBID bash`.
3. Find the process ID of the Python script with `htop -u $USER`.
4. Activate the virtual environment where `py-spy` is installed.
5. Run `py-spy top --pid PID` to view live where the code is using a lot of time.
6. Run `py-spy dump --pid PID` to get a trace of the state of your code.


### Message: `Package 'X' requires a different Python: X.Y.Z not in '>=X.Y'`

When installing a package, you may get an error like:

```
ERROR: Package 'X' requires a different Python: 3.6.10 not in '>=3.7'
```

In this case, the loaded Python 3.6.10 module is not supported by the package. You can use a more recent version of Python, such as the latest available module, or install an older version of package X.


### Message: `Package has requirement X, but you'll have Y which is incompatible`

When installing a package, you may get an error like:

```
ERROR: Package has requirement X, but you'll have Y which is incompatible.
```

To use the new dependency resolver, install the latest version of `pip` or a version greater than [21.3].

```bash
(ENV) [name@server ~] pip install --no-index --upgrade pip
```

Then run the installation command again.


### Message: `No matching distribution found for X`

When installing a package, you may get a message similar to:

```bash
(ENV) [name@server ~] pip install X
ERROR: Could not find a version that satisfies the requirement X (from versions: none)
ERROR: No matching distribution found for X
```

`pip` did not find any package to install that meets the requirements (name, version, or tags).

Make sure the name and version are correct. Also know that `manylinux_x_y` wheels are ignored.

You can also check if the package is available with the `avail_wheels` command or by consulting the "Available Wheels" page.


### Installing Multiple Packages

Whenever possible, it is best to install multiple packages with a single command:

```bash
(ENV) [name@server ~] pip install --upgrade pip
(ENV) [name@server ~] pip install package1 package2 package3 package4
```

This way, `pip` can more easily resolve dependency issues.


### My virtual environment worked yesterday, but not today

Frequent package updates mean that a virtual environment often cannot be reproduced.

It is also possible that a virtual environment created in `$SCRATCH` is partially destroyed during the automatic purging of this file system, which would prevent the virtual environment from working properly.

To counter this, freeze the packages and their versions with:

```bash
(ENV) [name@server ~] pip install --upgrade pip
(ENV) [name@server ~] pip install --no-index 'package1==X.Y' 'package2==X.Y.Z' 'package3<X.Y' 'package4>X.Y'
```

and then create a [requirements file](link-to-requirements-file-info) that will be used to install these packages in your task.


### Message: `The wheel X is not supported on this platform`

When installing a package, you may get an error like:

```
ERROR: package-3.8.1-cp311-cp311-manylinux_2_28_x86_64.whl is not a supported wheel on this platform.
```

Some packages may be incompatible or not supported by our systems. Two common cases are:

1. Installing a `manylinux` package.
2. A Python package built for a different Python version (e.g., installing a package built for Python 3.11 when you have Python 3.9).

Some `manylinux` packages can be found among our [Python wheels](link-to-wheels-page).


### Message: `AttributeError: module ‘numpy’ has no attribute ‘X’`

When installing a wheel, the latest version of Numpy is installed if no specific version is requested. Several attributes were deprecated in Numpy v1.20 and are no longer available in v1.24.

Depending on the attribute, an error like `AttributeError: module ‘numpy’ has no attribute ‘bool’` might occur. This is solved by installing a previous version of Numpy with:

```bash
pip install --no-index 'numpy<1.24'
```


### Message: `ModuleNotFoundError: No module named 'X'`

It is possible that a Python module you want to import is not found. There are several explanations for this, but the most frequent are that:

1. The package is not installed or is not visible to the Python interpreter.
2. The module name does not match the actual name.
3. The virtual environment is faulty.

To counter this, avoid:

1. Modifying the `PYTHONPATH` environment variable.
2. Modifying the `PATH` environment variable.
3. Loading a module while a virtual environment is activated; load all modules first before activating the virtual environment.

If you have this problem:

1. With `pip list`, check if the package is installed.
2. Check again if the name you enter exactly matches the module name (uppercase, lowercase, underscores, etc.).
3. Check if the module is imported at the right level when it comes from its source directory.

If in doubt, start again with a new environment.


### Message: `ImportError: numpy.core.multiarray failed to import`

This message may appear when you try to import a Python module that depends on Numpy. This happens when an incompatible version of Numpy is installed or used; you must install a compatible version. The typical case is Numpy version 2.0, which breaks the ABI. In the case of a wheel built with version 1.x but installed with version 2.x, you must install a previous version with:

```bash
pip install --no-index 'numpy<2.0'
```

Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Python/fr&oldid=162728](https://docs.alliancecan.ca/mediawiki/index.php?title=Python/fr&oldid=162728)"
