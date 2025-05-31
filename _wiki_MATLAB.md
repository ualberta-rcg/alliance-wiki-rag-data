# MATLAB on Compute Canada Clusters

There are two ways to use MATLAB on Compute Canada clusters:

1.  **Running MATLAB directly:** This requires a license. You can run MATLAB on Béluga, Cedar, or Narval, which have licenses available for students, professors, and academic researchers. Alternatively, use an external license owned by your institution, faculty, department, or lab. See [Using an external license](#using-an-external-license) below.

2.  **Compiling your MATLAB code:** Use the MATLAB Compiler (`mcc`) and run the generated executable on any cluster. This method does not require a license.


## Contents

*   [Using an external license](#using-an-external-license)
*   [Preparing your .matlab folder](#preparing-your-matlab-folder)
*   [Available toolboxes](#available-toolboxes)
*   [Running a serial MATLAB program](#running-a-serial-matlab-program)
*   [Parallel execution of MATLAB](#parallel-execution-of-matlab)
    *   [Simultaneous parallel MATLAB jobs](#simultaneous-parallel-matlab-jobs)
*   [Using the Compiler and Runtime libraries](#using-the-compiler-and-runtime-libraries)
*   [Using the MATLAB Parallel Server](#using-the-matlab-parallel-server)
    *   [Slurm plugin for MATLAB](#slurm-plugin-for-matlab)
    *   [Edit the plugin once installed](#edit-the-plugin-once-installed)
    *   [Validation](#validation)
*   [External resources](#external-resources)


## Using an external license

Compute Canada is a hosting provider for MATLAB.  MATLAB is installed on our clusters, and you can access an external license to run computations. Arrangements are made with several Canadian institutions for automatic access. To check if you have access:

```bash
[name@cluster ~]$ module load matlab/2023b.2
[name@cluster ~]$ matlab -nojvm -nodisplay -batch license

987654
[name@cluster ~]$
```

If a license number is printed, you have access. Run this test on each cluster where you want to use MATLAB, as licenses may not be available everywhere.

If you get the message:

```
This version is newer than the version of the license.dat file and/or network license manager on the server machine
```

Try an older MATLAB version in the `module load` line.

Otherwise, your institution may not have a MATLAB license, may not allow its use this way, or arrangements haven't been made. Contact your institution's MATLAB license administrator (faculty, department) or your MathWorks account manager to determine if you're allowed to use the license in this way.

If allowed, technical configuration is required. Create a file like this:

**File: matlab.lic**

```
# MATLAB license server specifications
SERVER
<ip address>
ANY
<port>
USE_SERVER
```

Place this file in the `$HOME/.licenses/` directory. Replace `<ip address>` and `<port>` with your campus license server values.  The license server on your campus must be reachable by Compute Canada's compute nodes. Contact technical support to arrange this.


For online documentation, see [http://www.mathworks.com/support](http://www.mathworks.com/support).

For product information, visit [http://www.mathworks.com](http://www.mathworks.com).


## Preparing your .matlab folder

Because the `/home` directory is read-only on some compute nodes, create a `.matlab` symbolic link to write MATLAB profile and job data to the `/scratch` space:

```bash
[name@cluster ~]$ cd $HOME
[name@cluster ~]$ if [ -d ".matlab" ]; then
  mv .matlab scratch/
else
  mkdir -p scratch/.matlab
fi && ln -sn scratch/.matlab .matlab
```


## Available toolboxes

To see available MATLAB toolboxes:

```bash
[name@cluster ~]$  module load matlab
[name@cluster ~]$  matlab -nojvm -batch "ver"
```


## Running a serial MATLAB program

**Important:**  Significant MATLAB calculations (taking more than ~5 minutes or 1 GB of memory) must be submitted to the scheduler.  See [Running jobs](Running jobs) for more on scheduler usage.


**Example code (cosplot.m):**

```matlab
function cosplot()
% MATLAB file example to approximate a sawtooth
% with a truncated Fourier expansion.
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

**Slurm script (matlab_slurm.sl):**

```bash
#!/bin/bash -l
#SBATCH --job-name=matlab_test
#SBATCH --account=def-someprof # adjust this to match the accounting group you are using to submit jobs
#SBATCH --time=0-03:00         # adjust this to match the walltime of your job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1      # adjust this if you are using parallel commands
#SBATCH --mem=4000             # adjust this according to the memory requirement per node you need
# Choose a version of MATLAB by loading a module:
module load matlab/2024b.1
matlab -singleCompThread -batch "cosplot"
```

Submit the job using `sbatch`:

```bash
[name@server ~]$ sbatch matlab_slurm.sl
```

Delete files like `java.log.12345` after MATLAB runs to save storage space.  You can suppress their creation with the `-nojvm` option, but this might interfere with plotting functions. See MATLAB (Linux) on the MathWorks website for command-line options (`-nodisplay`, `-nojvm`, `-singleCompThread`, `-batch`, etc.).


## Parallel execution of MATLAB

MATLAB supports various parallel execution modes.  Most users will find it sufficient to use a Threads parallel environment on a single node.

**Example code (timeparfor.m):**

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

**Job script (matlab_parallel.sh):**

```bash
#!/bin/bash -l
#SBATCH --account=def-someprof
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=2000
module load matlab/2024b.1
matlab -nojvm -batch "timeparfor"
```

Experiment with `--cpus-per-task` to observe performance effects.


### Simultaneous parallel MATLAB jobs

When using a Cluster parallel environment, multiple parallel MATLAB jobs calling `parpool` simultaneously may try to read/write to the same `.dat` file in `$HOME/.matlab/local_cluster_jobs/R*`, corrupting the parallel profile.  If this happens, delete the `local_cluster_jobs` folder when no jobs are running.

To avoid this, ensure each job creates its own parallel profile in a unique location by setting the `JobStorageLocation` property of the `parallel.Cluster` object:

**Example code (parallel_main.m):**

```matlab
local_cluster = parcluster('local')
local_cluster.JobStorageLocation = getenv('SLURM_TMPDIR')
parpool(local_cluster);
```

**References:**

*   FAS Research Computing, [MATLAB Parallel Computing Toolbox simultaneous job problem](MATLAB Parallel Computing Toolbox simultaneous job problem)
*   MathWorks, [Why am I unable to start a local MATLABPOOL from multiple MATLAB sessions that use a shared preference directory using Parallel Computing Toolbox 4.0 (R2008b)?](Why am I unable to start a local MATLABPOOL from multiple MATLAB sessions that use a shared preference directory using Parallel Computing Toolbox 4.0 (R2008b)?)


## Using the Compiler and Runtime libraries

**Important:** Like other intensive jobs, always run MCR code within a job submitted to the scheduler. See the [Running jobs](Running jobs) page.

Compile code using MATLAB Compiler (available in hosted modules). See MathWorks documentation. `mcc` is available for 2014a, 2018a, and later versions.

To compile `cosplot.m`:

```bash
[name@yourserver ~]$ mcc -m -R -nodisplay cosplot.m
```

This produces a `cosplot` binary and a wrapper script.  Only the binary is needed on Compute Canada servers. The provided `run_mcr_binary.sh` sets correct paths (unlike the generated `run_cosplot.sh`).

Load an MCR module matching your executable's MATLAB version:

```bash
[name@server ~]$ module load mcr/R2024b
```

Run:

```bash
[name@server ~]$ setrpaths.sh --path cosplot
```

Then, in your submission script (not on login nodes):

```bash
run_mcr_binary.sh cosplot
```

Run `setrpaths.sh` once per compiled binary. `run_mcr_binary.sh` will prompt you if needed.


## Using the MATLAB Parallel Server

MATLAB Parallel Server is only worthwhile if you need more workers than available CPU cores on a single node.  A regular MATLAB installation allows parallel jobs within one node (up to 64 workers, depending on the node and cluster). MATLAB Parallel Server is the licensed solution for running parallel jobs on multiple nodes.

This usually involves submitting jobs from a local MATLAB interface.  Since May 2023, security improvements prevent this until MATLAB uses a new connection method. There is currently no workaround.


### Slurm plugin for MATLAB

The following procedure no longer works due to unavailability of the Slurm plugin and the SSH issue.  The steps are kept until a workaround is found.

*   Install MATLAB R2022a or newer with the Parallel Computing Toolbox.
*   Download and run the `*.mlpkginstall` file from the MathWorks Slurm Plugin page.
*   Enter MathWorks credentials. If the configuration wizard doesn't start, run `parallel.cluster.generic.runProfileWizard()` in MATLAB.
*   In the wizard:
    *   Select Unix.
    *   Shared location: No
    *   Cluster host: beluga.computecanada.ca (Béluga) or narval.computecanada.ca (Narval)
    *   Username (optional): Your Alliance username.
    *   Remote job storage: `/scratch`
    *   Keep "Use unique subfolders" checked.
    *   Maximum number of workers: 960
    *   Matlab installation folder for workers: `/cvmfs/restricted.computecanada.ca/easybuild/software/2020/Core/matlab/2022a` (for local R2022a)
    *   License type: Network license manager
    *   Profile Name: beluga or narval
    *   Click Create and Finish.

### Edit the plugin once installed

1.  In MATLAB: `cd(fullfile(matlabshared.supportpkg.getSupportPackageRoot, 'parallel', 'slurm', 'nonshared'))`
2.  Open `independentSubmitFcn.m`. Around line 117, replace:

    ```matlab
    additionalSubmitArgs = sprintf('--ntasks=1 --cpus-per-task=%d', cluster.NumThreads);
    ```

    with:

    ```matlab
    additionalSubmitArgs = ccSBATCH().getSubmitArgs();
    ```

3.  Open `communicatingSubmitFcn.m`. Around line 126, replace:

    ```matlab
    additionalSubmitArgs = sprintf('--ntasks=%d --cpus-per-task=%d', environmentProperties.NumberOfTasks, cluster.NumThreads);
    ```

    with:

    ```matlab
    additionalSubmitArgs = ccSBATCH().getSubmitArgs();
    ```

4.  Open `communicatingJobWrapper.sh`. After the copyright statement (around line 20), add:

    ```bash
    module load matlab/2022a
    ```

    Adjust the module version as needed.

5.  Restart MATLAB and `cd(getenv('HOME'))`.


### Validation

Do not use the built-in validation tool in the Cluster Profile Manager. Instead, try the TestParfor example with a `ccSBATCH.m` script file:

Download and extract code samples from [https://github.com/ComputeCanada/matlab-parallel-server-samples](https://github.com/ComputeCanada/matlab-parallel-server-samples). Follow instructions in the README.  When `ccSBATCH.m` is in your working directory, you can try the Cluster Profile Manager validation tool, but only the first two tests will work.


## External resources

MathWorks documentation and training:

*   Documentation ([https://www.mathworks.com/help/matlab/](https://www.mathworks.com/help/matlab/))
*   Self-paced online courses ([https://matlabacademy.mathworks.com/](https://matlabacademy.mathworks.com/))

University MATLAB documentation:

*   More examples with job scripts: [https://rcs.ucalgary.ca/MATLAB](https://rcs.ucalgary.ca/MATLAB)

