# PyRETIS

PyRETIS is a Python library for molecular simulations of rare events using methods based on Transition Interface Sampling (TIS) and Replica Exchange Transition Interface Sampling (RETIS).

## Installation

Calcul Canada offers pre-compiled Python wheels for PyRETIS (see available wheels) that are compatible with certain Python versions and can be installed in a Python virtual environment.

As of July 2020, PyRETIS 2.5.0 is compatible with Python versions 3.6 and 3.7.  According to the installation guidelines, the dependency MDTraj must be installed *after* PyRETIS.

To create a Python virtualenv, run the following commands, where lines starting with `#` are comments, those starting with `$` are prompts, and those starting with `(env_PyRETIS) $` are prompts with the virtualenv activated.

```bash
# load the Python module we want to use, e.g. python/3.7:
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

To use `pyretisrun` (in scripts for example) we only need to activate the module again with `source ~/env_PyRETIS/bin/activate`

```bash
pyretisrun --input INPUT.rst  --log_file LOG_FILE.log
```

PyRETIS also offers the PyVisA analysis tool whose user interface requires PyQt5 to be run. PyQt5 is included in the Qt module.  For the Python virtualenv to find PyQt5, it is important to first load the modules for Python and Qt before activating the Python virtualenv like so:

```bash
$ module load python/3.7 qt/5.11.3
$ source ~/env_PyRETIS/bin/activate
(env_PyRETIS) $ pyretisanalyse  -pyvisa  ...
```

## Usage

Consult the documentation on the website and in the following articles:

* Lervik A, Riccardi E, van Erp TS. PyRETIS: A well-done, medium-sized python library for rare events. J Comput Chem. 2017;38: 2439–2451. doi:10.1002/jcc.24900
* Riccardi E, Lervik A, Roet S, Aarøen O, Erp TS. PyRETIS 2: An improbability drive for rare events. J Comput Chem. 2020;41: 370–377. doi:10.1002/jcc.26112

