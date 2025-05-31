# GNU Parallel

This page is a translated version of the page GNU Parallel and the translation is 100% complete.

Other languages: English, français

## Contents

* [Introduction](#introduction)
* [Basic Commands](#basic-commands)
* [Specifying Multiple Arguments](#specifying-multiple-arguments)
* [Using the Content of a File as a List of Arguments](#using-the-content-of-a-file-as-a-list-of-arguments)
* [Using the Content of a File as a List of Commands](#using-the-content-of-a-file-as-a-list-of-commands)
* [Using Multiple Nodes](#using-multiple-nodes)
* [Tracking Executed or Failed Commands; Restart Features](#tracking-executed-or-failed-commands-restart-features)
* [Working with Large Files](#working-with-large-files)
* [Running Hundreds or Thousands of Simulations](#running-hundreds-or-thousands-of-simulations)
    * [Argument List](#argument-list)
    * [Command List](#command-list)
    * [Multiple Arguments](#multiple-arguments)
* [See Also](#see-also)


## Introduction

`parallel` is a GNU tool that allows you to execute multiple sequential tasks in parallel on one or more nodes. It is a particularly useful tool for executing a large number of short or variable-duration sequential tasks on a set of parameters (parameter exploration). This documentation only covers the basics; for more information, see the [product documentation](link_to_product_documentation).

By default, `parallel` maximizes resource utilization by running as many tasks as the number of cores allocated by the scheduler. You can change this with the `--jobs` option, followed by the number of tasks that GNU Parallel should execute simultaneously. When a task is finished, `parallel` automatically starts the next one; thus, the maximum number of tasks is always running.


## Basic Commands

The curly braces `{}` indicate the parameters passed as arguments to the command to be executed.  Thus, to execute the `gzip` command on all text files in a directory, you will use `gzip` as follows:

```bash
[name@server ~]$ ls *.txt | parallel gzip {}
```

You can also use `:::`, as in the following example:

```bash
[name@server ~]$ parallel echo {} ::: $(seq 1 3)
1
2
3
```

GNU Parallel commands are called `jobs`.  These `jobs` should not be confused with tasks (also `jobs`) which are scripts executed by the scheduler; in this context, the tasks executed with `parallel` are `subtasks`.


## Specifying Multiple Arguments

You can also use multiple arguments by numbering them as follows:

```bash
[name@server ~]$ parallel echo {1} {2} ::: $(seq 1 3) ::: $(seq 2 3)
1 2
1 3
2 2
2 3
3 2
3 3
```


## Using the Content of a File as a List of Arguments

The `::::` syntax allows you to use the content of a file as argument values.  Thus, if your parameter list is in the file `maliste.txt`, you can display its content as follows:

```bash
[name@server ~]$ parallel echo {1} :::: maliste.txt
```


## Using the Content of a File as a List of Commands

The lines in a file can represent subtasks to be executed in parallel; in this case, each subtask must be on a separate line.  Thus, if your list of subtasks is in the file `my_commands.txt`, you can execute it as follows:

```bash
[name@server ~]$ parallel < my_commands.txt
```

Note that no command or argument is passed to `parallel`. This usage mode is particularly useful if the subtasks contain specific GNU Parallel symbols or if the subcommands contain several commands, for example `cd dir1 && ./executable`.

The following example shows how to execute a task by Slurm with GNU Parallel. The list of commands in `my_commands.txt` will be executed sequentially with 4 CPUs. As soon as a command is finished, a new command is launched to always have 4 commands running at the same time, until the end of the list.

**Script**

**Task List**

**File:** `run_gnuparallel_test.sh`

```bash
#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --cpus-per-task=4
#SBATCH --time=00-02:00
#SBATCH --mem=4000M     # Total memory for all tasks
parallel --joblog parallel.log < ./my_commands.txt
```

**File:** `my_commands.txt`

```
command1
command2
command3
command4
command5
command6
command7
command8
command9
```


## Using Multiple Nodes

**NOT RECOMMENDED**

Even though GNU parallel can be used with multiple nodes of a cluster, it is not recommended to do so, as problems may arise, especially in the case of short tasks. This is because an SSH session must be launched on a remote node, which is often a multi-second operation and the session is likely to be blocked. If you choose to do so, be sure to use the `--sshdelay 30` option to have a delay of at least 30 seconds between each task.

You can also distribute your work across multiple nodes of a cluster, as in the following example:

```bash
[name@server ~]$ scontrol show hostname > ./node_list_${SLURM_JOB_ID}
[name@server ~]$ parallel --jobs $SLURM_NTASKS_PER_NODE --sshloginfile ./node_list_${SLURM_JOB_ID} --env MY_VARIABLE --workdir $PWD --sshdelay 30 ./my_program
```

Here we create a file containing the list of nodes to indicate to GNU parallel which ones to use to distribute the tasks. The `--env` option allows us to transfer a particular environment variable to all nodes and the `--workdir` option ensures that the GNU parallel tasks will be launched in the same directory as the main node.

For example, when multiple OpenMP tasks are submitted together with `--nodes=N`, `--ntasks-per-node=5` and `--cpus-per-task=8`, the following command will manage all the processes to be started on all the reserved nodes, as well as the number of OpenMP threads per process.

```bash
[name@server ~]$ export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
[name@server ~]$ parallel --jobs $SLURM_NTASKS_PER_NODE --sshloginfile ./node_list_${SLURM_JOB_ID} --workdir $PWD --env OMP_NUM_THREADS --sshdelay 30 ./my_program
```

In this case, `5*N` OpenMP processes work simultaneously and CPU usage can go up to 800%.


## Tracking Executed or Failed Commands; Restart Features

The argument `--joblog JOBLOGFILE` produces a log of the executed commands. The JOBLOGFILE file then contains the list of completed commands with the start time, duration, compute node name and exit code, for example:

```bash
[name@server ~]$ ls *.txt | parallel --joblog gzip.log gzip {}
```

This function offers several options for restarting. If the `parallel` command was interrupted (i.e., the task took longer than specified), it could resume using the `--resume` option, for example:

```bash
[name@server ~]$ ls *.txt | parallel --resume --joblog gzip.log gzip {}
```

New tasks will be added to the end of the same log.

If some of the subcommands failed (i.e., the exit code is different from zero) and you think the error is resolved, these subcommands can be executed again with `--resume-failed`, for example:

```bash
[name@server ~]$ ls *.txt | parallel --resume-failed --joblog gzip.log gzip {}
```

This also executes subtasks that were not previously considered.


## Working with Large Files

If for example we want to count in parallel the number of characters in the large FASTA file named `database.fa` using an 8-core task, we must use the `--pipepart` and `--block` arguments to efficiently manage large portions of the file.

```bash
[name@server ~]$ parallel --jobs $SLURM_CPUS_PER_TASK --keep-order --block -1 --recstart '>' --pipepart wc :::: database.fa
```

By varying the size of `block` we have:

| Number of Cores | Database Size | Block Size | Number of Parallel Tasks | Cores Used | Task Duration |
|---|---|---|---|---|---|
| 1 | 827MB | 10MB | 83 | 8 | 0m2.633s |
| 2 | 827MB | 100MB | 9 | 8 | 0m2.042s |
| 3 | 827MB | 827MB | 1 | 1 | 0m10.877s |
| 4 | 827MB | -1 | 8 | 8 | 0m1.734s |

We see that using the right block size has a real impact on efficiency and the number of cores used. On the first line, the block size is too small and several tasks are distributed across the available cores. On the second line, the block size is more appropriate since the number of tasks is closer to the number of cores available. On the third line, the block size is too large and only one core is used out of 8, which is not efficient. On the last line, we note that letting GNU Parallel adapt and decide the block size itself is often faster.


## Running Hundreds or Thousands of Simulations

Start by determining the amount of resources needed for one simulation; you can then determine the total amount of resources required by the task.

In the following examples, the submission scripts are for 1 serial simulation with 2GB of memory, 1 core and 5 minutes, and 1000 simulations. With 1 core, the duration would be 83.3 hours. With 1 node of 32 cores, the duration would be 6 hours. It would also be possible to use more than one node (see #Using Multiple Nodes).


### Argument List

As mentioned in the section #Using the Content of a File as a List of Arguments, you can use a file that contains all the parameters. In this case, the parameters are separated by a tab character (`\t`) and each line corresponds to a simulation.

**Parameters**

**Script**

**File:** `my_parameters.txt`

```
1	1
1	2
1	3
...
```

**File:** `sim_submit.sh`

```bash
#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --mem-per-cpu=2G
# Read the parameters, placing each column to their respective argument
parallel -j $SLURM_CPUS_PER_TASK --colsep '\t' my_simulator --alpha {1} --beta {2} :::: ./my_parameters.txt
```


### Command List

As mentioned in the section #Using the Content of a File as a List of Commands, you can use a file that contains all the commands and their parameters.

**Commands**

**Script**

**File:** `my_commands.txt`

```
my_simulator --alpha 1 --beta 1
my_simulator --alpha 1 --beta 2
my_simulator --alpha 1 --beta 3
...
```

**File:** `sim_submit.sh`

```bash
#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --mem-per-cpu=2G
parallel -j $SLURM_CPUS_PER_TASK < ./my_commands.txt
```


### Multiple Arguments

You can use GNU Parallel to generate the parameters and associate them with the commands.

**File:** `sim_submit.sh`

```bash
#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --mem-per-cpu=2G
# Generates 1000 simulations where the alpha argument ranges from 1-10, and beta from 1-100
# placing each source to their respective argument
parallel -j $SLURM_CPUS_PER_TASK my_simulator --alpha {1} --beta {2} ::: {1..10} ::: {1..100}
```


## See Also

META
GLOST
Task Vectors

Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=GNU_Parallel/fr&oldid=175630](https://docs.alliancecan.ca/mediawiki/index.php?title=GNU_Parallel/fr&oldid=175630)"
