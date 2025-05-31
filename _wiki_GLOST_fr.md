# GLOST (Greedy Launcher Of Small Tasks)

This page is a translated version of the page GLOST and the translation is 100% complete.

Other languages: [English](link-to-english-page), [français](current-page-url)


## Introduction

GLOST (for Greedy Launcher Of Small Tasks) is a tool for running a large number of short or variable-duration sequential tasks, or with parameter sweeps.  Its operation is similar to that of GNU parallel or a task vector, but with simplified syntax.

GLOST uses the wrapper `glost_launch` and the MPI commands `srun`, `mpiexec`, and `mpirun`. A text file named `list_glost_tasks.txt` groups the tasks and is used as an argument for the `glost_launch` wrapper.

GLOST is particularly useful in the following cases:

* Multiple sequential tasks of comparable duration
* Multiple short sequential tasks
* Sequential tasks with variable parameters (parameter sweep)

The principle is to group several sequential tasks and run them in an MPI task that can use multiple cores (one or more nodes). With fewer tasks in the queue, the scheduler will be less solicited.

You might consider using the META software package developed by one of our teams instead, which has significant advantages over GLOST. With META, the total waiting time can be much shorter; the overhead imposed is less (fewer wasted CPU cycles); a practical mechanism allows resubmitting calculations that have failed or have never been executed; and META can handle both sequential and multithreaded, MPI, GPU, and hybrid tasks.

**NOTE:** Read this entire page to determine if this tool can be used in your work. If so, you can request assistance from the technical team to modify your processes.


## Advantages

Depending on their duration and number, several sequential tasks are grouped into one or more MPI tasks.

Submitting multiple sequential tasks at the same time can slow down the scheduler and cause long response times and frequent interruptions in the execution of `sbatch` or `squeue`. GLOST's solution is to group all sequential tasks into a single file named `list_glost_tasks.txt` and submit an MPI task with the `glost_launch` wrapper. This greatly reduces the number of tasks in the queue and therefore produces fewer requests to be processed by the scheduler than if the tasks were submitted separately. To submit multiple sequential tasks without delay, GLOST alleviates the burden on Slurm.

With GLOST, the user submits and processes a few MPI tasks rather than hundreds or thousands of sequential tasks.


## Modules

GLOST uses OpenMPI to group sequential tasks into an MPI task. You must load OpenMPI and the corresponding GLOST module. For more information, see [Using Modules](link-to-modules-page). To see the available GLOST modules, use the command `module spider glost`. Before submitting a task, make sure you can load GLOST and the other modules required to run your application.

```bash
$ module spider glost/0.3.1

--------------------------------------------------------------------------------------------------------------------------------------
glost:
glost/0.3.1
--------------------------------------------------------------------------------------------------------------------------------------
Description:
This is GLOST, the Greedy Launcher Of Small Tasks.
Properties:
Tools for development / Outils de développement
You will need to load all module(s) on any one of the lines below before the "glost/0.3.1" module is available to load.
StdEnv/2023 gcc/12.3 openmpi/4.1.5
StdEnv/2023 intel/2023.2.1 openmpi/4.1.5
Help:
Description
===========
This is GLOST, the Greedy Launcher Of Small Tasks.
More information
================
- Homepage: https://github.com/cea-hpc/glost
```

If an OpenMPI module is already in your environment, which is the case for the default environment, adding `module load glost` to the list of modules you need is sufficient to activate GLOST. To ensure that GLOST and the other modules are present, run the command `module list`.


## Usage

### Syntax

The following forms are possible:

```bash
srun glost_launch list_glost_tasks.txt

mpiexec glost_launch list_glost_tasks.txt
mpirun glost_launch list_glost_tasks.txt
```

### Number of Cores and Number of Tasks

Sequential tasks are assigned to available cores by cyclic distribution. The GLOST wrapper starts with the first task (or line in the list) and assigns it a processor. This is repeated until the end of the list or until the task duration is reached. The number of cores does not necessarily correspond to the number of tasks listed. However, to optimize resources, ensure that the tasks have a similar execution time and that they can be distributed evenly across the requested number of cores. Let's examine the following cases:

* With a large number of very short sequential tasks (e.g., hundreds or thousands of tasks of a few minutes each), submit one or more GLOST tasks to run them using a limited number of cores. You can submit the tasks with a short duration and per node to take advantage of backfilling and the scheduler.
* With tens to hundreds of relatively short tasks (approximately one hour), you can group them into one or more GLOST tasks.
* With several long-duration tasks with similar execution times, you can also group them into a GLOST task.


### Execution Time Estimation

Before launching a task, try to estimate its execution time; this can be used to estimate the execution time of the GLOST task.

Suppose your GLOST task includes a number `Njobs` of similar tasks where each uses a time `t0` on one (1) processor. The total duration will then be `t0*Njobs`.

To now use a number of cores `Ncores`, the duration will be `wt = t0*Njobs/Ncores`.

**Note:** An MPI task is often designed so that processors can exchange information between them, which often uses a large part of the time for communication rather than performing calculations. A large number of small dependent communications can decrease the performance of the code, but GLOST uses MPI to launch sequential tasks only and therefore, the communication overhead is relatively rare. You can achieve the same result by using MPI directly, but GLOST is almost as efficient while saving you from writing MPI code.


### Memory Requirements

GLOST runs sequential tasks with MPI, and the memory per core should be the same as the memory used by the tasks run separately. In the Slurm script, use `--mem-per-cpu` rather than `--mem`.


### Creating the Task List

Before submitting a task, create a text file named `list_glost_tasks.txt` with one task per line and the commands for each task. Choosing tasks with similar execution times optimizes the resources used. Tasks can be located in one or more directories. If the tasks are all in the same directory, it is necessary to avoid the results using the same temporary files or the same output files; to do this, the results can be redirected to a file with a variable indicating the option or argument used in the execution of the task. In the case where the tasks use the same temporary files or the same output files, you may need to create a directory for each task (a directory for each option or argument corresponding to a particular task).

**Note:** A task can contain one or more commands executed one after the other. Commands must be separated by `&&`.


The following `list_glost_example.txt` file contains eight tasks.

**File: `run_glost_test.sh`**

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=00-02:00
#SBATCH --mem-per-cpu=4000M
# load the GLOST module
module load intel/2023.2.1 openmpi/4.1.5 glost/0.3.1
echo "Starting run at: `date`"
# launch GLOST with the argument list_glost_example.txt
srun glost_launch list_glost_example.txt
echo "Program glost_launch finished with exit code $? at: `date`"
```

**File: `list_glost_example.txt`**

```
job01 and/or other commands related to job01
job02 and/or other commands related to job02
job03 and/or other commands related to job03
job04 and/or other commands related to job04
job05 and/or other commands related to job05
job06 and/or other commands related to job06
job07 and/or other commands related to job07
job08 and/or other commands related to job08
```

**Note:** This example script does not contain any commands and cannot be executed. It only shows the basic syntax for the task list `list_glost_tasks.txt` that will serve as an argument for `glost_launch`; a typical script for submitting tasks. The task list and script must be adapted to your context.


### Task List Located in the Same Directory

GLOST can be used to execute a set or list of sequential tasks in a directory. It is necessary to avoid the results using the same temporary files or the same output files by adding arguments to differentiate the tasks. The next example contains 10 tasks, each containing one or more commands that will be executed one after the other.

The first command defines `nargument`, which can be a variable or parameter that can, for example, be passed to the program; the second command executes the program; for testing purposes, we use the command `sleep 360` which you will replace with the command line for your application, for example `./my_first_prog < first_input_file.txt > first_output_file.txt`; the third command and the following ones are optional; for testing purposes, we use `echo ${nargument}.`hostname` > log_${nargument}.txt` which prints the argument and hostname to the file `log_${nargument}.txt`. As is the case for the second command, this line will be replaced according to your application, for example by `./my_second_prog < second_input_file.txt > second_output_file.txt`.


**File: `run_glost_test.sh`**

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=00-02:00
#SBATCH --mem-per-cpu=4000M
# load the GLOST module with the other modules needed to launch your application
module load intel/2023.2.1 openmpi/4.1.5 glost/0.3.1
echo "Starting run at: `date`"
# launch GLOST with the argument list_glost_tasks.txt
srun glost_launch list_glost_tasks.txt
echo "Program glost_launch finished with exit code $? at: `date`"
```

**File: `list_glost_tasks.txt`**

```
nargument=20 && sleep 360 && echo ${nargument}.`hostname` > log_${nargument}.txt
nargument=21 && sleep 360 && echo ${nargument}.`hostname` > log_${nargument}.txt
nargument=22 && sleep 360 && echo ${nargument}.`hostname` > log_${nargument}.txt
nargument=23 && sleep 360 && echo ${nargument}.`hostname` > log_${nargument}.txt
nargument=24 && sleep 360 && echo ${nargument}.`hostname` > log_${nargument}.txt
nargument=25 && sleep 360 && echo ${nargument}.`hostname` > log_${nargument}.txt
nargument=26 && sleep 360 && echo ${nargument}.`hostname` > log_${nargument}.txt
nargument=27 && sleep 360 && echo ${nargument}.`hostname` > log_${nargument}.txt
nargument=28 && sleep 360 && echo ${nargument}.`hostname` > log_${nargument}.txt
nargument=29 && sleep 360 && echo ${nargument}.`hostname` > log_${nargument}.txt
```

**Note:** In this example, we use 2 cores and a list of 10 tasks. The first two tasks (corresponding to the first two lines) will be assigned by GLOST to the available processors. When the processor(s) have finished processing the first two tasks, they will move on to the next task and so on until the end of the list.


### Task List Located in Different Directories

In this case, several sequential tasks are executed in separate directories, which can be useful to avoid tasks ending abnormally or results overlapping when a program uses temporary files or input/output files with identical names. It is necessary to ensure that each task has its input files and its directory. It is also possible to use commands as in the following example:


**File: `run_glost_test.sh`**

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --time=00-03:00
#SBATCH --mem-per-cpu=4000M
# load the GLOST module with the other modules needed to launch your application
module load intel/2023.2.1 openmpi/4.1.5 glost/0.3.1
echo "Starting run at: `date`"
# launch GLOST with the argument list_glost_tasks.txt
srun glost_launch list_glost_tasks.txt
echo "Program glost_launch finished with exit code $? at: `date`"
```

**File: `list_glost_tasks.txt`**

```
nargument=20 && mkdir -p RUN_${nargument} && cd RUN_${nargument} && sleep 360 && echo ${nargument}.`hostname` > log_run.txt
nargument=21 && mkdir -p RUN_${nargument} && cd RUN_${nargument} && sleep 360 && echo ${nargument}.`hostname` > log_run.txt
nargument=22 && mkdir -p RUN_${nargument} && cd RUN_${nargument} && sleep 360 && echo ${nargument}.`hostname` > log_run.txt
nargument=23 && mkdir -p RUN_${nargument} && cd RUN_${nargument} && sleep 360 && echo ${nargument}.`hostname` > log_run.txt
nargument=24 && mkdir -p RUN_${nargument} && cd RUN_${nargument} && sleep 360 && echo ${nargument}.`hostname` > log_run.txt
nargument=25 && mkdir -p RUN_${nargument} && cd RUN_${nargument} && sleep 360 && echo ${nargument}.`hostname` > log_run.txt
nargument=26 && mkdir -p RUN_${nargument} && cd RUN_${nargument} && sleep 360 && echo ${nargument}.`hostname` > log_run.txt
nargument=27 && mkdir -p RUN_${nargument} && cd RUN_${nargument} && sleep 360 && echo ${nargument}.`hostname` > log_run.txt
nargument=28 && mkdir -p RUN_${nargument} && cd RUN_${nargument} && sleep 360 && echo ${nargument}.`hostname` > log_run.txt
nargument=29 && mkdir -p RUN_${nargument} && cd RUN_${nargument} && sleep 360 && echo ${nargument}.`hostname` > log_run.txt
nargument=30 && mkdir -p RUN_${nargument} && cd RUN_${nargument} && sleep 360 && echo ${nargument}.`hostname` > log_run.txt
nargument=31 && mkdir -p RUN_${nargument} && cd RUN_${nargument} && sleep 360 && echo ${nargument}.`hostname` > log_run.txt
```


### Restarting a GLOST Task

If you have misjudged the execution time of your GLOST task, it may need to be restarted to process all tasks. First, identify the tasks that have been executed and delete the corresponding lines in the list or create a new list with the unexecuted tasks. Resubmit the script with the new list as an argument to `glost_launch`.


## Other Examples

If you are used to preparing scripts, use the following examples and modify them according to your context.

After loading the GLOST module, copy the examples into your directory with the command `cp -r $EBROOTGLOST/examples Glost_Examples`. The copied examples will be saved in the `Glost_Examples` directory.


## References

* [META-Farm](link-to-meta-farm-page)
* [GNU parallel](link-to-gnu-parallel-page)
* [Task Vectors](link-to-task-vectors-page)
* [MPI](link-to-mpi-page)
* [Running Tasks](link-to-running-tasks-page)

**(Remember to replace bracketed placeholders like `[link-to-english-page]` with actual links.)**
