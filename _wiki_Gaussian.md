# Gaussian

Other languages: English, français

See also: [Gaussian error messages](Gaussian_error_messages)


Gaussian is a computational chemistry application produced by Gaussian, Inc.


## Contents

* [Limitations](#limitations)
* [License agreement](#license-agreement)
* [Running Gaussian on Graham and Cedar](#running-gaussian-on-graham-and-cedar)
    * [Job submission](#job-submission)
        * [G16 (G09, G03)](#g16-g09-g03)
        * [g16 (g09, g03)](#g16-g09-g03-1)
        * [Submit the job](#submit-the-job)
    * [Interactive jobs](#interactive-jobs)
    * [Restart jobs](#restart-jobs)
    * [Examples](#examples)
* [Notes](#notes)
* [Errors](#errors)


## Limitations

We currently support Gaussian only on Graham and Cedar.

Cluster/network parallel execution of Gaussian, also known as "Linda parallelism", is not supported at any of your national systems. Only "shared-memory multiprocessor parallel execution" is supported. Therefore, no Gaussian job can use more than a single compute node.


## License agreement

In order to use Gaussian, you must agree to certain conditions. Please contact support with a copy of the following statement:

* I am not a member of a research group developing software competitive to Gaussian.
* I will not copy the Gaussian software, nor make it available to anyone else.
* I will properly acknowledge Gaussian Inc. and the Alliance in publications.
* I will notify the Alliance of any change in the above acknowledgement.

If you are a sponsored user, your sponsor (PI) must also have such a statement on file with us. We will then grant you access to Gaussian.


## Running Gaussian on Graham and Cedar

The `gaussian` module is installed on Graham and Cedar. To check what versions are available, use the `module spider` command as follows:

```bash
[name@server $] module spider gaussian
```

For module commands, please see [Using modules](Using_modules).


### Job submission

The national clusters use the Slurm scheduler; for details about submitting jobs, see [Running jobs](Running_jobs).

Since only the "shared-memory multiprocessor" parallel version of Gaussian is supported, your jobs can use only one node and up to the maximum cores per node: 48 on Cedar and 32 on Graham. If your jobs are limited by the amount of available memory on a single node, be aware that there are a few nodes at each site with more than the usual amount of memory. Please refer to the pages [Cedar](Cedar) and [Graham](Graham) for the number and capacity of such nodes.

Besides your input file (in our example `name.com`), you have to prepare a job script to define the compute resources for the job; both input file and job script must be in the same directory.

There are two options to run your Gaussian job on Graham and Cedar, based on the location of the default runtime files and the job size.


#### G16 (G09, G03)

This option will save the default runtime files (unnamed `.rwf`, `.inp`, `.d2e`, `.int`, `.skr` files) to `/scratch/username/jobid/`. Those files will stay there when the job is unfinished or failed for whatever reason; you could locate the `.rwf` file for restart purposes later.

The following example is a G16 job script:

Note that for coherence, we use the same name for each file, changing only the extension (`name.sh`, `name.com`, `name.log`).

**File:** `mysub.sh`

```bash
#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --mem=16G             # memory, roughly 2 times %mem defined in the input name.com file
#SBATCH --time=02-00:00       # expect run time (DD-HH:MM)
#SBATCH --cpus-per-task=16    # No. of cpus for the job as defined by %nprocs in the name.com file
module load gaussian/g16.c01
G16 name.com # G16 command, input: name.com, output: name.log
```

To use Gaussian 09 or Gaussian 03, simply modify the `module load gaussian/g16.b01` to `gaussian/g09.e01` or `gaussian/g03.d01`, and change `G16` to `G09` or `G03`. You can modify the `--mem`, `--time`, `--cpus-per-task` to match your job's requirements for compute resources.


#### g16 (g09, g03)

This option will save the default runtime files (unnamed `.rwf`, `.inp`, `.d2e`, `.int`, `.skr` files) temporarily in `$SLURM_TMPDIR` (`/localscratch/username.jobid.0/`) on the compute node where the job was scheduled. The files will be removed by the scheduler when a job is done (successful or not). If you do not expect to use the `.rwf` file to restart later, you can use this option.

`/localscratch` is ~800G shared by all jobs running on the same node. If your job files would be bigger than or close to that size range, you would instead use the G16 (G09, G03) option.

The following example is a `g16` job script:

**File:** `mysub.sh`

```bash
#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --mem=16G             # memory, roughly 2 times %mem defined in the input name.com file
#SBATCH --time=02-00:00       # expect run time (DD-HH:MM)
#SBATCH --cpus-per-task=16    # No. of cpus for the job as defined by %nprocs in the name.com file
module load gaussian/g16.c01
g16 < name.com # g16 command, input: name.com, output: slurm-<jobid>.out by default
```


#### Submit the job

```bash
sbatch mysub.sh
```


### Interactive jobs

You can run interactive Gaussian jobs for testing purposes on Graham and Cedar. It's not a good practice to run interactive Gaussian jobs on a login node. You can start an interactive session on a compute node with `salloc`. The example for an hour, 8 cpus, and 10G memory Gaussian job is like:

Go to the input file directory first, then use the `salloc` command:

```bash
[name@server ~] $ salloc --time=1:0:0 --cpus-per-task=8 --mem=10g
```

Then use either:

```bash
[name@server ~] $ module load gaussian/g16.c01
[name@server ~] $ G16 g16_test2.com # G16 saves runtime file (.rwf etc.) to /scratch/yourid/93288/
```

or

```bash
[name@server ~] $ module load gaussian/g16.c01
[name@server ~] $ g16 < g16_test2.com > & g16_test2.log & # g16 saves runtime file to /localscratch/yourid/
```


### Restart jobs

Gaussian jobs can always be restarted from the previous `rwf` file.

Geometry optimization can be restarted from the `chk` file as usual.

One-step computation, such as Analytic frequency calculations, including properties like ROA and VCD with ONIOM; CCSD and EOM-CCSD calculations; NMR; Polar=OptRot; CID, CISD, CCD, QCISD, and BD energies, can be restarted from the `rwf` file.

To restart a job from a previous `rwf` file, you need to know the location of this `rwf` file from your previous run. The restart input is simple: first, you need to specify `%rwf` path to the previous `rwf` file, secondly change the keywords line to be `#p restart`, then leave a blank line at the end.

A sample restart input is like:

**File:** `restart.com`

```
%rwf=/scratch/yourid/jobid/name.rwf
%NoSave
%chk=name.chk
%mem=5000mb
%nprocs=16
#p restart

(one blank line)
```


### Examples

An example input file and the run scripts `*.sh` can be found in `/opt/software/gaussian/version/examples/` where `version` is either `g03.d10`, `g09.e01`, or `g16.b01`.


## Notes

NBO7 is included in `g16.c01` version only; both `nbo6` and `nbo7` keywords will run NBO7 in `g16.c01`. NBO6 is available in `g09.e01` and `g16.b01` versions.

You can watch a recorded webinar/tutorial: [Gaussian16 and NBO7 on Graham and Cedar](Gaussian16_and_NBO7_on_Graham_and_Cedar)


## Errors

Some of the error messages produced by Gaussian have been collected, with suggestions for their resolution. See [Gaussian error messages](Gaussian_error_messages).


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Gaussian&oldid=138134](https://docs.alliancecan.ca/mediawiki/index.php?title=Gaussian&oldid=138134)"
