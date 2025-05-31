# MATLAB

This page is a translated version of the page MATLAB and the translation is 100% complete.

Other languages: [English](link-to-english-page), [français](current-page-url)


There are two ways to use MATLAB on our clusters:

1.  Run MATLAB directly, but you need access to a license:
    *   The license provided on Cedar, Béluga, or Narval for students, professors, and researchers;
    *   An external license held by your institution, faculty, department, or laboratory (see the section *Using an external license* below).

2.  Compile your MATLAB code with the `mcc` compiler and use the generated executable on one of our clusters. You can use this executable regardless of the license.


Details for these approaches are provided below.


## Contents

*   [Using an external license](#using-an-external-license)
*   [Preparing your .matlab directory](#preparing-your-matlab-directory)
*   [Toolboxes](#toolboxes)
*   [Running a sequential MATLAB program](#running-a-sequential-matlab-program)
*   [Running in parallel](#running-in-parallel)
*   [Running multiple parallel tasks simultaneously](#running-multiple-parallel-tasks-simultaneously)
*   [Using the Compiler and Runtime Libraries](#using-the-compiler-and-runtime-libraries)
*   [Using MATLAB Parallel Server](#using-matlab-parallel-server)
    *   [Extension module for Slurm](#extension-module-for-slurm)
    *   [Modifying the extension after installation](#modifying-the-extension-after-installation)
    *   [Validation](#validation)
*   [External resources](#external-resources)


## Using an external license

We are hosting providers for MATLAB. In this context, MATLAB is installed on our clusters, and you may have access to an external license to use our infrastructure; in the case of some institutions, this is done automatically. To find out if you have access to a license, do the following test:

```bash
[name@cluster ~]$ module load matlab/2023b.2
[name@cluster ~]$ matlab -nojvm -nodisplay -batch license

987654
[name@cluster ~]$
```

If everything is in order, a license number will be printed. Be sure to perform this test on each cluster you want to use MATLAB with, as some licenses are not available everywhere.

If you get the message:

```
This version is newer than the version of the license.dat file and/or network license manager on the server machine
```

try entering an older version of MATLAB in the `module load` line.

Otherwise, your institution may not have a license, it may not be possible to use the license in this way, or no agreement has been reached with us to use the license. To find out if you can use an external license, contact your institution's MATLAB license administrator or your MATLAB account manager.

If you can use an external license, some configuration steps are required. First, you must create a file similar to:

**File:** `matlab.lic`

```
# license server specifications
SERVER
<ip address>
ANY
<port>
USE_SERVER
```

and place this file in the directory `$HOME/.licenses/`, where the IP address and port number correspond to the values of your institution's license server. Our technical team will then need to contact the technical staff managing your license so that your server can connect to our compute nodes. To arrange this, contact [technical support](link-to-support).

Consult the technical documentation [http://www.mathworks.com/support](http://www.mathworks.com/support) and product information [http://www.mathworks.com](http://www.mathworks.com).


## Preparing your .matlab directory

Since the `/home` directory of some compute nodes is read-only, you must create a symbolic link `.matlab` so that the profile and job data are logged in `/scratch` instead.

```bash
[name@cluster ~]$ cd $HOME
[name@cluster ~]$ if [ -d ".matlab" ]; then
  mv .matlab scratch/
else
  mkdir -p scratch/.matlab
fi && ln -sn scratch/.matlab .matlab
```


## Toolboxes

To get the list of toolboxes available with the license and the cluster you are working on, use:

```bash
[name@cluster ~]$  module load matlab
[name@cluster ~]$  matlab -nojvm -batch "ver"
```


## Running a sequential MATLAB program

**Important:** For all large-scale computations (duration of more than five minutes or memory of 1 GB), the task must be submitted to the scheduler as shown in the following example. For more information, see [Running tasks](link-to-running-tasks).

Here is an example of code:

**File:** `cosplot.m`

```matlab
function cosplot()
% example to approximate a sawtooth signal
% by a truncated Fourier series
nterms = 5;
fourbypi = 4.0 / pi;
np = 100;
y(1:np) = pi / 2.0;
x(1:np) = linspace(-2.0 * pi, 2 * pi, np);
for k = 1:nterms
twokm = 2 * k - 1;
y = y - fourbypi * cos(twokm * x) / twokm^2;
end
plot(x, y)
print -dpsc matlab_test_plot.ps
quit
end
```

Here is a script for the Slurm scheduler that runs `cosplot.m`:

**File:** `matlab_slurm.sl`

```bash
#!/bin/bash -l
#SBATCH --job-name=matlab_test
#SBATCH --account=def-someprof # account name used to submit jobs
#SBATCH --time=0-03:00         # time limit (DD-HH:MM)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1      # modify if you use parallel commands
#SBATCH --mem=4000             # memory required per node (in megabytes by default)
# load the module for the desired version
module load matlab/2024b.1
matlab -singleCompThread -batch "cosplot"
```

Submit the task with `sbatch`.

```bash
[name@server ~]$ sbatch matlab_slurm.sl
```

Each time MATLAB is launched, a file like `java.log.12345` may be created. To save storage space, this file should be deleted once MATLAB is closed.  However, the creation of this file can be avoided by using the `-nojvm` option, but this may interfere with some plotting functions.

For more information on command-line options including `-nodisplay`, `-nojvm`, `-singleCompThread`, `-batch`, and others, see [MATLAB (Linux) on the MathWorks site](link-to-mathworks-site).


## Running in parallel

MATLAB supports [several parallel execution modes](link-to-parallel-modes).

For most of you, it will suffice to run MATLAB in a parallel `Threads` environment on a single node.

Here is an example inspired by [the MathWorks documentation on `parfor`](link-to-mathworks-parfor-doc).

**File:** `timeparfor.m`

```matlab
function timeparfor()
nthreads = str2num(getenv('SLURM_CPUS_PER_TASK'))
parpool("Threads", nthreads)
tic
n = 200;
A = 500;
a = zeros(1, n);
parfor i = 1:n
a(i) = max(abs(eig(rand(A))));
end
toc
end
```

Save the above code in a file named `timeparfor.m`. Then create the following script and submit it with `sbatch matlab_parallel.sh` to run the function in parallel with four cores.

**File:** `matlab_parallel.sh`

```bash
#!/bin/bash -l
#SBATCH --account=def-someprof
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=2000
module load matlab/2024b.1
matlab -nojvm -batch "timeparfor"
```

You can experiment by giving `--cpus-per-task` smaller values (e.g., 1, 2, 6, 8) to see the effect on performance.


## Running multiple parallel tasks simultaneously

If you use a parallel `Cluster` environment as [described here](link-to-cluster-description), the following problem may occur. When two or more parallel tasks initialize `parpool` at the same time, each task tries to read and write to the same `.dat` file in the directory `$HOME/.matlab/local_cluster_jobs/R*`. This corrupts the local parallel profile used by the other tasks. If this happens, delete the `local_cluster_jobs` directory when no tasks are running.

To avoid this problem, we recommend that each task creates its own parallel profile in a unique location by specifying the property of the `parallel.Cluster` object, as shown here.

**File:** `parallel_main.m`

```matlab
local_cluster = parcluster('local')
local_cluster.JobStorageLocation = getenv('SLURM_TMPDIR')
parpool(local_cluster);
```

**References:**

*   FAS Research Computing, [MATLAB Parallel Computing Toolbox simultaneous job problem](link-to-fas-research-computing)
*   MathWorks, [Why am I unable to start a local MATLABPOOL from multiple MATLAB sessions that use a shared preference directory using Parallel Computing Toolbox 4.0 (R2008b)?](link-to-mathworks-problem)


## Using the Compiler and Runtime Libraries

**Important:** As with all high-demand tasks, the MCR code must always be included in a task submitted to the scheduler; see [Running tasks](link-to-running-tasks).

You can also compile your code with MATLAB Compiler, one of the modules we host.  See the [MATLAB Compiler documentation](link-to-matlab-compiler-doc).

For now, `mcc` is available for versions 2014a, 2018a and later.

To compile the example with `cosplot.m` above, you would use the command:

```bash
[name@yourserver ~]$ mcc -m -R -nodisplay cosplot.m
```

This produces the binary `cosplot` and the wrapper script `run_cosplot.sh`. To run the binary on our servers, you only need the binary. The wrapper script will not work as is on our servers because MATLAB expects certain libraries to be in specific locations. Use the wrapper script `run_mcr_binary.sh` instead, which sets the correct paths.

Load the MCR module corresponding to the MATLAB version you used to create your executable:

```bash
[name@server ~]$ module load mcr/R2024b
```

Run the command:

```bash
[name@server ~]$ setrpaths.sh --path cosplot
```

Then, in the script for the task (and not in the login nodes), use the binary as follows:

```bash
run_mcr_binary.sh cosplot
```

The `setrpaths.sh` command only needs to be run once for each of the compiled binaries; `run_mcr_binary.sh` will prompt you to run it if it hasn't been done.


## Using MATLAB Parallel Server

MATLAB Parallel Server is only useful if your parallel MATLAB task has more processes (called `workers`) than the CPU cores available on a single compute node. The standard MATLAB installation described above allows you to run parallel tasks with one node (up to 64 `workers` per task depending on the cluster and node); to use more than one node.

This solution usually allows you to submit parallel MATLAB tasks from the local MATLAB interface on your computer.

Some improvements to the security of our clusters were made in May 2023, and since MATLAB uses an SSH mode that is no longer allowed, it is no longer possible to submit a task from a local computer as long as MATLAB does not use a new method to connect. There is currently no solution.


## Extension module for Slurm

The following procedure does not work due to the Slurm extension no longer being available and also the SSH problem mentioned in the previous section.

However, we have kept it for when the solution becomes available.

Install MATLAB R2022a (or later), including the Parallel Computing Toolbox.

From the MathWorks Slurm Plugin page, download and run the `*.mlpkginstall` file (Download button to the right of the page, under the Overview tab).

Enter your MathWorks credentials. If the configuration does not start automatically, run the command `parallel.cluster.generic.runProfileWizard()` in MATLAB.

Enter the following information:

*   Select Unix (usually the only option offered)
*   Shared location: No
*   Cluster host:
    *   For Béluga: `beluga.computecanada.ca`
    *   For Narval: `narval.computecanada.ca`
*   Username (optional): (enter your username; if necessary, the identity file can be defined later)
*   Remote job storage: `/scratch`
*   Check Use unique subfolders.
*   Maximum number of workers: 960
*   Matlab installation folder for workers: (local and remote versions must match)
    *   For R2022a: `/cvmfs/restricted.computecanada.ca/easybuild/software/2020/Core/matlab/2022a`
*   License type: Network license manager
*   Profile Name: `beluga` or `narval`

Click Create and Finish to complete the profile.


## Modifying the extension after installation

In the MATLAB terminal, go to the `nonshared` directory by running the command `cd(fullfile(matlabshared.supportpkg.getSupportPackageRoot, 'parallel', 'slurm', 'nonshared'))`.

Open the file `independentSubmitFcn.m`; around line 117, replace:

```matlab
additionalSubmitArgs = sprintf('--ntasks=1 --cpus-per-task=%d', cluster.NumThreads);
```

with:

```matlab
additionalSubmitArgs = ccSBATCH().getSubmitArgs();
```

Open the file `communicatingSubmitFcn.m`; around line 126, replace:

```matlab
additionalSubmitArgs = sprintf('--ntasks=%d --cpus-per-task=%d', environmentProperties.NumberOfTasks, cluster.NumThreads);
```

with:

```matlab
additionalSubmitArgs = ccSBATCH().getSubmitArgs();
```

Open the file `communicatingJobWrapper.sh`; around line 20 (after the copyright statement), add the following command and adjust the module version according to your local Matlab version:

```bash
module load matlab/2022a
```

Restart MATLAB and return to your `/home` directory with `cd(getenv('HOME'))`  # or under Windows, `cd(getenv('HOMEPATH'))`


## Validation

Do not use the Cluster Profile Manager validation tool, but run the TestParfor example with a properly configured `ccSBATCH.m` script file.

Download and extract code examples from [https://github.com/ComputeCanada/matlab-parallel-server-samples](https://github.com/ComputeCanada/matlab-parallel-server-samples).

In MATLAB, open the `TestParfor` directory you just extracted.

Follow the instructions given in the file [https://github.com/ComputeCanada/matlab-parallel-server-samples/blob/master/README.md](https://github.com/ComputeCanada/matlab-parallel-server-samples/blob/master/README.md).

Note: When `ccSBATCH.m` is in your current directory, you can use the Cluster Profile Manager validation tool for the first two tests because the others are not yet supported.


## External resources

See also the resources offered by MathWorks.

*   Documentation: [https://www.mathworks.com/help/matlab/](https://www.mathworks.com/help/matlab/) (some pages are in French)
*   Self-learning: [https://matlabacademy.mathworks.com/](https://matlabacademy.mathworks.com/) (also in EN, JP, ES, KR, CN versions)

Some universities have their own documentation, such as for example scripts: [https://rcs.ucalgary.ca/MATLAB](https://rcs.ucalgary.ca/MATLAB)


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=MATLAB/fr&oldid=178649](https://docs.alliancecan.ca/mediawiki/index.php?title=MATLAB/fr&oldid=178649)"
