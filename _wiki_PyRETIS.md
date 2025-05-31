# PyRETIS

PyRETIS is a Python library for rare event molecular simulations, focusing on methods based on transition interface sampling (TIS) and replica exchange transition interface sampling (RETIS).

## Installing PyRETIS

Pre-compiled Python Wheels for PyRETIS are available in our [Wheelhouse](link-to-wheelhouse-here).  These are compatible with various Python versions and can be installed within a virtual Python environment.

As of July 2020, PyRETIS 2.5.0 is compatible with Python versions 3.6 and 3.7.  According to the [PyRETIS installation instructions](link-to-instructions-here), the dependency `MDTraj` must be installed *after* PyRETIS.

A Python virtualenv with PyRETIS can be created using the following commands:

```bash
# load the Python module (e.g., python/3.7)
$ module load python/3.7

# create a virtualenv
$ virtualenv --no-download ~/env_PyRETIS

# activate the virtualenv
$ source ~/env_PyRETIS/bin/activate

# install PyRETIS and then mdtraj
(env_PyRETIS) $ pip install --no-index pyretis
(env_PyRETIS) $ pip install --no-index mdtraj

# run PyRETIS
(env_PyRETIS) $ pyretisrun --help
```

To use `pyretisrun` (e.g., in job scripts), activate the module again:

```bash
source ~/env_PyRETIS/bin/activate
pyretisrun --input INPUT.rst  --log_file LOG_FILE.log
```

PyRETIS also includes an analysis tool, PyVisA. Its GUI requires PyQt5, which is typically part of the Qt modules. To ensure the virtualenv's Python version finds PyQt5, load the Python and Qt modules before activating the PyRETIS virtualenv:

```bash
$ module load python/3.7 qt/5.11.3
$ source ~/env_PyRETIS/bin/activate
(env_PyRETIS) $ pyretisanalyse  -pyvisa  ...
```

## Using PyRETIS

Documentation on using PyRETIS is available on the [PyRETIS website](http://www.pyretis.org/) and in these publications:

* Lervik A, Riccardi E, van Erp TS. PyRETIS: A well-done, medium-sized python library for rare events. J Comput Chem. 2017;38: 2439–2451. doi:10.1002/jcc.24900
* Riccardi E, Lervik A, Roet S, Aarøen O, Erp TS. PyRETIS 2: An improbability drive for rare events. J Comput Chem. 2020;41: 370–377. doi:10.1002/jcc.26112


