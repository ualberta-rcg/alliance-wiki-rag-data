# Job Arrays

This page is a translated version of the page [Job arrays](https://docs.alliancecan.ca/mediawiki/index.php?title=Job_arrays) and the translation is 100% complete.

Other languages: [English](https://docs.alliancecan.ca/mediawiki/index.php?title=Job_arrays), [français](https://docs.alliancecan.ca/mediawiki/index.php?title=Job_arrays/fr)

This page is a child page of [Running tasks](https://docs.alliancecan.ca/mediawiki/index.php?title=Running_tasks)


If you have multiple tasks that differ by only one parameter, you can use a job array (job array, array job, task array). The environment variable `$SLURM_ARRAY_TASK_ID` differentiates each task, and the scheduler assigns a different value to each. The values are defined by the `--array` parameter.

For more information, see [SchedMD documentation](https://slurm.schedmd.com/documentation.html).


## Examples of the `--array` parameter

```bash
sbatch --array=0-7       # $SLURM_ARRAY_TASK_ID uses values from 0 to 7 inclusive
sbatch --array=1,3,5,7   # $SLURM_ARRAY_TASK_ID uses values from the list
sbatch --array=1-7:2     # the step is equal to 2 as in the previous example
sbatch --array=1-100%10  # limits to 10 the number of tasks executed simultaneously
```


## Simple Example

**File:** `simple_array.sh`

```bash
#!/bin/bash
#SBATCH --array=1-10
#SBATCH --time=3:00:00
program_x <input.$SLURM_ARRAY_TASK_ID
program_y $SLURM_ARRAY_TASK_ID some_arg another_arg
```

This script creates 10 independent tasks. Each has a maximum duration of 3 hours and each can start at different times on different nodes.

The script uses `$SLURM_ARRAY_TASK_ID` to indicate the file for input data (in our example `program_x`) or to use as a command line argument (with for example `program_y`).

Using a job array instead of multiple sequential tasks is advantageous for yourself and other users. A pending job array produces only one line in `squeue`, allowing you to check its result more easily.  In addition, the scheduler is not required to analyze the needs of each task separately, resulting in a performance gain.


Excluding the use of `sbatch` as an initial step, the scheduler experiences the same load with a job array as with an equivalent number of tasks submitted separately. It is not recommended to use an array to submit tasks that have a duration of much less than an hour. Tasks lasting only a few minutes should be grouped with `META`, `GLOST`, `GNU Parallel`, or in an interpreter loop inside a task.


## Example with Multiple Directories

Suppose you want to run the same script in multiple directories with an identical structure; if the directory names can be sequential numbers, it would be easy to adapt the example presented above; otherwise, create a file with the directory names as follows:

```bash
$ cat case_list
pacific2016
pacific2017
atlantic2016
atlantic2017
```

There are several ways to select a particular line in a file; in the next example, we use `sed`.

**File:** `directories_array.sh`

```bash
#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --array=1-4
echo "Starting task $SLURM_ARRAY_TASK_ID"
DIR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" case_list)
cd $DIR
# enter the code here
pwd
ls
```

**WARNING:** The number of tasks you request must be equal to the number of lines in the file. The `case_list` file must not be modified until all tasks in the array have been executed since the file will be read at the beginning of each new task.


## Example with Multiple Parameters

Suppose you have a Python script that performs calculations with certain parameters defined in a Python list or a NumPy array such as:

**File:** `my_script.py`

```python
import time
import numpy as np

def calculation(x, beta):
    time.sleep(2) #simulates a long execution
    return beta * np.linalg.norm(x**2)

if __name__ == "__main__":
    x = np.random.rand(100)
    betas = np.linspace(10,36.5,100) #subdivides the interval [10,36.5] into 100 values
    for i in range(len(betas)): #iteration on the values of the parameter beta
        res = calculation(x,betas[i])
        print(res) #displays the results
# to be executed with python my_script.py
```

The processing of this task can be done with a job array so that each value of the parameter `beta` is processed in parallel.  It is necessary to pass `$SLURM_ARRAY_TASK_ID` to the Python script and obtain the parameter `beta` according to its value.

The Python script is now:

**File:** `my_script_parallel.py`

```python
import time
import numpy as np
import sys

def calculation(x, beta):
    time.sleep(2) #simulates a long execution
    return beta * np.linalg.norm(x**2)

if __name__ == "__main__":
    x = np.random.rand(100)
    betas = np.linspace(10,36.5,100) #subdivides the interval [10,36.5] into 100 values
    i = int(sys.argv[1]) #gets the value of $SLURM_ARRAY_TASK_ID
    res = calculation(x,betas[i])
    print(res) #displays the results
# to be executed with python my_script_parallel.py $SLURM_ARRAY_TASK_ID
```

The script to submit the task is as follows (note that the parameters range from 0 to 99 just like the indices of the NumPy array).

**File:** `data_parallel_python.sh`

```bash
#!/bin/bash
#SBATCH --array=0-99
#SBATCH --time=1:00:00
module load scipy-stack
python my_script_parallel.py $SLURM_ARRAY_TASK_ID
```

Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Job_arrays/fr&oldid=150091](https://docs.alliancecan.ca/mediawiki/index.php?title=Job_arrays/fr&oldid=150091)"
