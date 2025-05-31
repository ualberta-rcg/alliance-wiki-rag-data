# Job Scheduling Policies

A lot of work can be accomplished on our clusters by submitting jobs that only specify the number of cores required and the maximum execution time.  However, if you want to submit multiple jobs or jobs that require a large amount of resources, you will likely gain productivity with a good understanding of our scheduling policy.

## Fair Share Prioritization

Jobs are processed in order of priority determined by the Fair Tree algorithm.<sup>[1]</sup> Each job is charged to a resource allocation project. The project is defined by the `--account` argument passed to `sbatch`.

For a project that has been allocated CPU or GPU time as part of the resource allocation competition, the account code usually starts with `rrg-` or `rpp-`. For a project using the rapid access service, the account name usually starts with `def-`. To find out which code to use, see the Accounts and Projects section of the Running Jobs page.

Each project is assigned a target usage level. Projects from the resource allocation competition have a target usage that depends on the amount of CPU-years or GPU-years allocated. Other types of projects have an equal target usage.

As an example, let's determine the usage and sharing information for a fictitious group with the account code `def-prof1`. The usernames for the members of this group are `prof1`, `grad2`, and `postdoc3`. We can see the usage and sharing information with the `sshare` command as shown below. Note that you must add `_cpu` or `_gpu` to the end of the account code since both are counted individually.

```bash
[prof1@gra-login4 ~]$ sshare -l -A def-prof1_cpu -u prof1,grad2,postdoc3
       Account       User  RawShares  NormShares  RawUsage  ... EffectvUsage  ...    LevelFS  ...
-------------- ---------- ---------- -----------  --------  ... ------------  ... ----------  ...
def-prof1_cpu                 434086    0.001607   1512054  ...     0.000043  ...  37.357207  ...
def-prof1_cpu      prof1          1    0.100000         0  ...     0.000000  ...        inf  ...   
 def-prof1_cpu      grad2          1    0.100000     54618  ...     0.036122  ...   2.768390  ...
 def-prof1_cpu   postdoc3          1    0.100000    855517  ...     0.565798  ...   0.176741  ...
```

We have removed several fields from the example that are not relevant to our discussion. The first line in red is the most important for scheduling; it describes the status of the project relative to other projects using the cluster. In our example, the value for sharing is 0.1607% and the use of cluster resources is at 0.0043%. At 37, the `LevelFS` value is quite high because the group has only used a small portion of the resources allocated to it.  It can be expected that the tasks submitted by this group will have a rather high priority.

The lines of the table show the values for each user relative to the other users *in the same project*. On the 3rd line, we see that `grad2` has 1 share, which represents 10% of the resources allocated to the group; its usage accounts for only 3.6122% of the recent usage by the group and the `LevelFS` value for this user is the highest. The tasks submitted by `grad2` should therefore have a slightly higher priority than those for `postdoc3`, but lower than those for `prof1`. The priority level for the tasks of the `def-prof1` group relative to those for other research groups is determined only by the `LevelFS` value for the group and not by that of the users who make up the group.

Slurm documentation calls the project itself or the user within a project an *association*. The `Account` column contains the project name with the suffix `_cpu` or `_gpu`. In the `User` column, the first line has no username.

The content of the `RawShares` column is proportional to the number of CPU-years of the cluster allocated to the project as part of the resource allocation competition. Accounts that do not have resources allocated by competition have a small equal number of shares. For numerical reasons, inactive accounts (those with no running or pending tasks) receive a single share. Activity is tracked periodically; if you submit a task with an inactive account, there may be a delay of up to 15 minutes before the `RawShares` and `LevelFS` values are updated.

The content of the `NormShares` column shows the number of shares assigned to the user or account, divided by the total number of shares assigned for that level. On the first line, the value 0.001607 is the fraction of shares held by the project relative to all projects. On the other lines, the value 0.10000 is the fraction of shares held by each of the project members, relative to the other members; there are ten members, but we only requested information for three of them.

The content of the `RawUsage` column represents a weighting of the total number of resource-seconds (i.e., CPU time, GPU time, and memory) charged to the account. Past usage is reduced by a half-life of one week (see half-life); usage dating back more than a few weeks will therefore have only a minimal effect on prioritization.

The `EffectvUsage` column shows the association's usage relative to its parent, i.e., the project's usage relative to other projects and each user's usage relative to all users. In this example, the usage of `postdoc3` is 56.6% and that of `grad2` is 3.6%.

The `LevelFS` column shows the fair share (FS) value expressed by `NormShares / EffectvUsage`. A result between 0 and 1 indicates an association that receives more resources than deserved from the scheduler; a result greater than 1 indicates an association that receives fewer resources than deserved from the scheduler. For an inactive account (as described under `RawShares`), the value is an infinitesimal number close to 0.0001.

A project for which the target is used regularly will see its `LevelFS` value close to 1.0. If the target is exceeded, `LevelFS` will be below 1.0 and new tasks for the project will also receive a low priority. If usage is below the target, `LevelFS` will be greater than 1.0 and new tasks will benefit from a high priority.

**See also:** [Allocation and scheduling of computing tasks](link-to-allocation-page).


## Whole Nodes or Cores

Parallel computations that can efficiently use 32 cores or more might be better served with whole nodes. A portion of each of the clusters is reserved for tasks requiring one or more whole nodes. For more information and example scripts, see the Whole Nodes section of the Controlling Scheduling with MPI page.

Note that requesting an inefficient number of processors simply to take advantage of any scheduling advantage for a whole node will be interpreted as unjustified resource abuse. For a program with a similar execution time on 16 cores and 32 cores, the request should be for `--ntasks=16` and not for `--nodes=1 --ntasks-per-node=32`. However, `--ntasks=16` is correct if you want all tasks to be on the same node.  Furthermore, since whole nodes reserve a specific amount of memory, submitting whole-node tasks that would misuse memory capacity would also be considered inappropriate.

If you have a large number of serial tasks and you can make good use of GNU Parallel, GLOST, or other techniques to gather these tasks for a single node, we encourage you to do so.


## Maximum Duration

Niagara can accommodate tasks with execution times up to 24 hours. The maximum duration with Béluga, Graham, and Narval is 7 days and with Cedar is 28 days.

With general-purpose clusters, long-duration tasks can only use a portion of the cluster by partitioning. There are partitions for tasks with execution times of 3 hours or less, 12 hours or less, 24 hours or less, 72 hours or less, 7 days or less, and 28 days or less.

Since a 3-hour execution time is shorter than 12 hours or more, shorter tasks can still be executed in partitions with longer maximum durations. A shorter task will therefore be likely to be scheduled faster than a longer task with identical other characteristics.

With Béluga, tasks must be longer than one hour.


## Backfilling

The scheduler optimizes resource utilization with backfilling. Without backfilling, each partition is scheduled strictly by priority, which generally minimizes resource utilization and response time. Backfilling ensures that low-priority tasks are launched provided that higher-priority tasks are not delayed. Since the expected time for launching pending tasks depends on the completion of current tasks, the proper functioning of the backfilling technique requires reasonably accurate maximum execution times.

Backfilling benefits tasks with shorter maximum durations, i.e., less than 3 hours.


## Available Node Percentages

Here is a description of the partitioning of the Cedar and Graham general-purpose clusters.

First, there are four categories of nodes:

*   `base` nodes (4 to 8 GB of memory per core)
*   `large` nodes (16 to 96 GB of memory per core)
*   `GPU base` nodes
*   `GPU large` nodes (on Cedar only)

Submitted tasks are directed to the appropriate category according to the required resources.

Then, among the nodes of the same category, some are reserved for tasks that can use whole nodes, i.e., those that use all the resources available on the allocated nodes. If a task uses few cores or even a single core on the same node, only a subset of the nodes will be allocated to it. These partitions are called "by-node" and "by-core".

Finally, the actual execution time also plays a role. Shorter tasks have access to more resources. For example, a task requiring an actual execution time of less than 3 hours can be found on any node that allows real times of 12 hours, but some nodes that accept 3-hour tasks *will not* accept 12-hour tasks.

The `partition-stats` utility shows:

*   In each partition, how many tasks are waiting in an execution queue,
*   How many tasks are running,
*   How many nodes are idle,
*   How many nodes are assigned to each of the partitions.

Here is an example:

```bash
[user@gra-login3 ~]$ partition-stats

Node type |                     Max walltime
          |   3 hr   |  12 hr  |  24 hr  |  72 hr  |  168 hr |  672 hr |
----------|-------------------------------------------------------------
       Number of Queued Jobs by partition Type (by node:by core)
----------|-------------------------------------------------------------
Regular   |   12:170 |  69:7066|  70:7335| 386:961 |  59:509 |   5:165 |
Large Mem |    0:0   |   0:0   |   0:0   |   0:15  |   0:1   |   0:4   |
GPU       |    5:14  |   3:8   |  21:1   | 177:110 |   1:5   |   1:1   |
----------|-------------------------------------------------------------
      Number of Running Jobs by partition Type (by node:by core)
----------|-------------------------------------------------------------
Regular   |    8:32  |  10:854 |  84:10  |  15:65  |   0:674 |   1:26  |
Large Mem |    0:0   |   0:0   |   0:0   |   0:1   |   0:0   |   0:0   |
GPU       |    5:0   |   2:13  |  47:20  |  19:18  |   0:3   |   0:0   |
----------|-------------------------------------------------------------
        Number of Idle nodes by partition Type (by node:by core)
----------|-------------------------------------------------------------
Regular   |   16:9   |  15:8   |  15:8   |   7:0   |   2:0   |   0:0   |
Large Mem |    3:1   |   3:1   |   0:0   |   0:0   |   0:0   |   0:0   |
GPU       |    0:0   |   0:0   |   0:0   |   0:0   |   0:0   |   0:0   |
----------|-------------------------------------------------------------
       Total Number of nodes by partition Type (by node:by core)
----------|-------------------------------------------------------------
Regular   |  871:431 | 851:411 | 821:391 | 636:276 | 281:164 |  90:50  |
Large Mem |   27:12  |  27:12  |  24:11  |  20:3   |   4:3   |   3:2   |
GPU       |  156:78  | 156:78  | 144:72  | 104:52  |  13:12  |  13:12  |
----------|-------------------------------------------------------------
```

At the top of the table, the values `12:170`, `0:0`, and `5:14` mean that:

*   12 tasks are waiting; these tasks requested whole nodes, less than 8 GB of memory per core, and an execution time of 3 hours or less.
*   170 tasks are waiting; these tasks requested less than whole nodes and are therefore waiting for individual cores, less than 8 GB of memory per core, and an execution time of 3 hours or less.
*   5 tasks are waiting; these tasks requested a whole node with GPU and an execution time of 3 hours or less.
*   14 tasks are waiting; these tasks requested single GPUs and an execution time of 3 hours or less.

No waiting or running tasks request a `large` node and 3 hours of execution time.

At the bottom of the table is the distribution of resources by policy; this does not take into account the tasks in progress.  Therefore, there are 871 `base` nodes called `regular` here, i.e., nodes with 4 to 8 GB per core that can receive whole-node tasks with a duration of less than 3 hours. Of these 871:

*   431 can also receive `by-core` tasks of less than 3 hours
*   851 can receive whole-node tasks with a duration of less than 12 hours
*   and so on.

The partitions are organized a bit like Russian nesting dolls. The partition for 3 hours contains a subset of nodes for the partition for 12 hours; the partition for 12 hours contains a subset of nodes for the partition for 24 hours; and so on.

The `partition-stats` utility does not provide any information on:

*   The number of cores used by running or pending tasks;
*   The number of free cores in the `by-core` partitions of partially assigned nodes; and
*   The available memory associated with the free cores in the `by-core` partitions.

Running `partition-stats` requires a lot from the scheduler. Therefore, avoid making automatic calls repeatedly in your scripts. If you believe it would be advantageous to use `partition-stats`, contact technical support to find out how to proceed.


## Number of Tasks

A limit may be imposed on the number of tasks running at the same time. For Graham and Béluga, a normal account can have no more than 1000 tasks running or pending at the same time. In a task vector, each counts as one task. The Slurm parameter `MaxSubmit` sets this limit.


<sup>[1]</sup> Read a detailed description of the algorithm in [https://slurm.schedmd.com/SC14/BYU_Fair_Tree.pdf](https://slurm.schedmd.com/SC14/BYU_Fair_Tree.pdf) which presents an example with the Beatles and Elvis Presley.

