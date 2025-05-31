# MPI for Python

MPI for Python provides a Python interface to the Message Passing Interface (MPI) standard, enabling Python applications to leverage multiple processors on workstations, clusters, and supercomputers.

## Available Versions

In our environment, `mpi4py` is a module and not a pre-compiled package (wheel) like most Python packages. To find the available versions, use:

```bash
[name@server ~]$ module spider mpi4py
```

For more information on a specific version, use:

```bash
[name@server ~]$ module spider mpi4py/X.Y.Z
```

where `X.Y.Z` is the version number, for example `4.0.0`.


## Hello World Example

1. Start a short interactive task:

```bash
[name@server ~]$ salloc --account=<your account> --ntasks=5
```

2. Load the module:

```bash
[name@server ~]$ module load mpi4py/4.0.0 python/3.12
```

3. Run a Hello World test:

```bash
[name@server ~]$ srun python -m mpi4py.bench helloworld
```

This will output something similar to:

```
Hello, World! I am process 0 of 5 on node1.
Hello, World! I am process 1 of 5 on node1.
Hello, World! I am process 2 of 5 on node3.
Hello, World! I am process 3 of 5 on node3.
Hello, World! I am process 4 of 5 on node3.
```

In this example, two nodes (node1 and node3) were allocated, and the tasks were distributed across the available resources.


## mpi4py as a Dependency of Another Package

When another package depends on `mpi4py`:

1. Deactivate any Python virtual environment:

```bash
[name@server ~]$ test $VIRTUAL_ENV && deactivate
```

**Note:** If a virtual environment is active, it's crucial to deactivate it before loading the module.  After loading the module, reactivate your virtual environment.

2. Load the module:

```bash
[name@server ~]$ module load mpi4py/4.0.0 python/3.12
```

3. Verify that the module is visible to `pip`:

```bash
[name@server ~]$ pip list | grep mpi4py
mpi4py          4.0.0
```

and that the Python module you loaded has access to it:

```bash
[name@server ~]$ python -c 'import mpi4py'
```

If no error occurs, everything is fine.

4. Create a virtual environment and install the packages.


## Running Tasks

MPI tasks can be distributed across multiple cores or multiple nodes. For more information, see [MPI job](link_to_mpi_job_doc) and [Advanced MPI scheduling](link_to_advanced_mpi_scheduling_doc).


### On CPU

1. Prepare the following Python code to distribute a NumPy array:

**File: "mpi4py-np-bc.py"**

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = np.arange(100, dtype='i')
else:
    data = np.empty(100, dtype='i')

comm.Bcast(data, root=0)

for i in range(100):
    assert data[i] == i
```

This example is based on the [mpi4py tutorial](link_to_mpi4py_tutorial).

2. Prepare the job submission script:

**Distributed Nodes:**

**File: submit-mpi4py-distributed.sh**

```bash
#!/bin/bash
#SBATCH --account=def-someprof    # adjust this to match the accounting group you are using to submit jobs
#SBATCH --time=08:00:00           # adjust this to match the walltime of your job
#SBATCH --ntasks=4                # adjust this to match the number of tasks/processes to run
#SBATCH --mem-per-cpu=4G          # adjust this according to the memory you need per process
# Run on cores across the system : https://docs.alliancecan.ca/wiki/Advanced_MPI_scheduling#Few_cores,_any_number_of_nodes
# Load modules dependencies.
module load StdEnv/2023 gcc mpi4py/4.0.0 python/3.12
# create the virtual environment on each allocated node:
srun --ntasks $SLURM_NNODES --tasks-per-node=1 bash << EOF
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index numpy==2.1.1
EOF
# activate only on main node
source $SLURM_TMPDIR/env/bin/activate;
# srun exports the current env, which contains $VIRTUAL_ENV and $PATH variables
srun python mpi4py-np-bc.py;
```

**Whole Nodes:**

**File: submit-mpi4py-whole-nodes.sh**

```bash
#!/bin/bash
#SBATCH --account=def-someprof    # adjust this to match the accounting group you are using to submit jobs
#SBATCH --time=01:00:00           # adjust this to match the walltime of your job
#SBATCH --nodes=2                 # adjust this to match the number of whole node
#SBATCH --ntasks-per-node=40      # adjust this to match the number of tasks/processes to run per node
#SBATCH --mem-per-cpu=1G          # adjust this according to the memory you need per process
# Run on N whole nodes : https://docs.alliancecan.ca/wiki/Advanced_MPI_scheduling#Whole_nodes
# Load modules dependencies.
module load StdEnv/2023 gcc openmpi mpi4py/4.0.0 python/3.12
# create the virtual environment on each allocated node:
srun --ntasks $SLURM_NNODES --tasks-per-node=1 bash << EOF
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index numpy==2.1.1
EOF
# activate only on main node
source $SLURM_TMPDIR/env/bin/activate;
# srun exports the current env, which contains $VIRTUAL_ENV and $PATH variables
srun python mpi4py-np-bc.py;
```

3. Test your script. Before submitting the job, it's important to test the script for possible errors. Do a quick test with an interactive task.

4. Submit your job:

```bash
[name@server ~]$ sbatch submit-mpi4py-distributed.sh
```


### On GPU

1. On a login node, download the example from the demos:

```bash
[name@server ~]$ wget https://raw.githubusercontent.com/mpi4py/mpi4py/refs/heads/master/demo/cuda-aware-mpi/use_cupy.py
```

2. Prepare your submission script:

**File: submit-mpi4py-gpu.sh**

```bash
#!/bin/bash
#SBATCH --account=def-someprof    # adjust this to match the accounting group you are using to submit jobs
#SBATCH --time=08:00:00           # adjust this to match the walltime of your job
#SBATCH --ntasks=2                # adjust this to match the number of tasks/processes to run
#SBATCH --mem-per-cpu=2G          # adjust this according to the memory you need per process
#SBATCH --gpus=1
# Load modules dependencies.
module load StdEnv/2023 gcc cuda/12 mpi4py/4.0.0 python/3.11
# create the virtual environment on each allocated node:
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install --no-index cupy numba

srun python use_cupy.py;
```

3. Test your script. Before submitting the job, it's important to test the script for possible errors. Do a quick test with an interactive task.

4. Submit your job:

```bash
[name@server ~]$ sbatch submit-mpi4py-gpu.sh
```


## Troubleshooting

### Message: `ModuleNotFoundError: No module named 'mpi4py'`

This message can occur during import when `mpi4py` is not accessible.

**`ModuleNotFoundError: No module named 'mpi4py'`**

**Suggested solutions:**

* Using `module spider mpi4py/X.Y.Z`, check which Python versions are compatible with the `mpi4py` module you loaded. When a compatible version is loaded, check if `python -c 'import mpi4py'` works;
* Load the module before activating your virtual environment (see `mpi4py as a dependency of another package` above).

See also [Message ModuleNotFoundError: No module named 'X'](link_to_module_not_found_doc).


**(Remember to replace placeholder links like `link_to_mpi4py_tutorial` with actual links when creating the final GitHub documentation.)**
