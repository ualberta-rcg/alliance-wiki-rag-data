# Gaussian

This page is a translated version of the page Gaussian and the translation is 100% complete.

Other languages: [English](link-to-english-page), [français](current-page-url)

See also the page on [Gaussian error messages](link-to-error-messages-page).


Gaussian is a computational chemistry application produced by Gaussian, Inc.


## Limits

Gaussian is currently available only on Graham and Cedar.

Our national systems do not support cluster/network parallel execution (Linda parallelism), but only shared-memory multiprocessing parallel execution.  Therefore, a Gaussian task cannot use more than one compute node.


## License

To use the application, you must agree to certain conditions. Copy the following statements into an email and send it to technical support.

* I am not part of a research group that is developing a competing application.
* I will not copy Gaussian or make the application available to a third party.
* I will acknowledge the Alliance's collaboration in any publication.
* I will inform the Alliance of any changes to the previous conditions.

If you are a user sponsored by a principal investigator, they must also have sent us a copy of the same statements. We can then give you access to Gaussian.


## Using Gaussian on Graham and Cedar

The `gaussian` module is installed on Graham and Cedar. To see the available versions, use the `module spider` command as follows:

```bash
[name@server $] module spider gaussian
```

For commands that apply to modules, see [Using Modules](link-to-modules-page).


### Submitting Tasks

The national clusters use the Slurm scheduler; for information on submitting a task, see [Running Tasks](link-to-running-tasks-page).

Since only the shared-memory multiprocessing version of Gaussian is supported, your tasks can only use a single node and up to 48 cores per node on Cedar and 32 cores per node on Graham. If your tasks require more memory than you can get on a single node, note that each cluster offers a few nodes with more memory. To find out the number of nodes on a cluster and their capacity, see [Cedar](link-to-cedar-page) and [Graham](link-to-graham-page).

In addition to the input file `name.com`, you must prepare a script describing the computational resources for the task; this script must be in the same directory as the input file.

There are two options for Gaussian tasks on Graham and Cedar, depending on the location of the default execution files and the size of the task:


#### Option 1: G16 (G09, G03)

With this option, the default execution files (unnamed .rwf, .inp, .d2e, .int, .skr) are saved in `/scratch/username/jobid/` and remain in this directory if the task is not completed or if it fails. The .rwf file can be retrieved from there to restart the task later.

Here is an example of a G16 script. Note that for consistency, the files have the same name with different extensions (name.sh, name.com, name.log).

**File: `mysub.sh`**

```bash
#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --mem=16G             # memory, roughly 2 times %mem defined in the input name.com file
#SBATCH --time=02-00:00       # expect run time (DD-HH:MM)
#SBATCH --cpus-per-task=16    # No. of cpus for the job as defined by %nprocs in the name.com file
module load gaussian/g16.c01
G16 name.com
# G16 command, input: name.com, output: name.log
```

To use Gaussian 09 or Gaussian 03, replace `gaussian/g16.b01` with `gaussian/g09.e01` or `gaussian/g03.d01` and replace `G16` with `G09` or `G03`. Modify `--mem`, `--time`, `--cpus-per-task` according to your computational resource needs.


#### Option 2: g16 (g09, g03)

With this option, the default execution files (unnamed .rwf, .inp, .d2e, .int, .skr) are temporarily saved in `$SLURM_TMPDIR` (`/localscratch/username.jobid.0/`) on the compute node where the task was to be executed. The scheduler deletes the files when the task is finished successfully or unsuccessfully. You can use this option if you will not need the .rwf file to restart the task later.

`/localscratch` is approximately 800GB, shared by all tasks running on the same node. If the size of your files is similar or larger, use the G16 (G09, G03) option instead.

Here is an example of a g16 script.

**File: `mysub.sh`**

```bash
#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --mem=16G             # memory, roughly 2 times %mem defined in the input name.com file
#SBATCH --time=02-00:00       # expect run time (DD-HH:MM)
#SBATCH --cpus-per-task=16    # No. of cpus for the job as defined by %nprocs in the name.com file
module load gaussian/g16.c01
g16 < name.com
# g16 command, input: name.com, output: slurm-<jobid>.out by default
```


#### Submitting the Task

```bash
sbatch mysub.sh
```


### Interactive Tasks

It is possible to run an interactive Gaussian task on Graham and Cedar for testing purposes. However, it is not recommended to run an interactive Gaussian task on a login node. Instead, open an interactive session on a compute node with `salloc` for a duration of one hour, with 8 CPUs and 10GB of memory.

```bash
[name@server ~] $ salloc --time=1:0:0 --cpus-per-task=8 --mem=10g
```

Then use:

```bash
[name@server ~] $ module load gaussian/g16.c01
[name@server ~] $ G16 g16_test2.com # G16 saves runtime file (.rwf etc.) to /scratch/yourid/93288/
```

or

```bash
[name@server ~] $ module load gaussian/g16.c01
[name@server ~] $ g16 < g16_test2.com > & g16_test2.log & # g16 saves runtime file to /localscratch/yourid/
```


### Restarting a Task

A Gaussian task can be restarted from the previous `rwf` file.

As usual, geometric optimization can be restarted from the `chk` file.

With the `rwf` file, you can restart calculations that are done in one step, such as analytical frequency calculations including properties such as ROA and VCD with ONIOM; CCSD and EOM-CCSD calculations; NMR; Polar=OptRot; and CID, CISD, CCD, QCISD and BD energies.

To restart a task from the `rwf` file, you must know the location of this `rwf` file from the previous task.  Simply specify the path `%rwf` to the previous `rwf` file first and modify the keyword line to read `#p restart`, then leave a blank line at the end.

Here is an example:

**File: `restart.com`**

```
%rwf=/scratch/yourid/jobid/name.rwf
%NoSave
%chk=name.chk
%mem=5000mb
%nprocs=16
#p restart

(one blank line)
```


## Examples

An example of an input file and `*.sh` scripts can be found in `/opt/software/gaussian/version/examples/`, where `version` is g03.d10, g09.e01, g16.a03 or g16.b01.


## Remarks

NBO7 is included only in version g16.c01 with the use of the keywords nbo6 and nbo7.

NBO6 is included in versions g09.e01 and g16.b01.

See the tutorial [Gaussian16 and NBO7 on Graham and Cedar](link-to-tutorial).


## Errors

You will find the solution to several errors in [Gaussian – Error Messages](link-to-error-messages-page).


**(Retrieved from "https://docs.alliancecan.ca/mediawiki/index.php?title=Gaussian/fr&oldid=138173")**
