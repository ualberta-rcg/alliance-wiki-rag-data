# R

R is a statistical computing and graphics tool. It is a programming language with a graphical environment, a debugger, access to certain system functions, and the ability to run scripts.

Even though R was not developed for high-performance computing, its popularity in several scientific disciplines, including engineering, mathematics, statistics, and bioinformatics, makes it an essential tool on supercomputers dedicated to university research.  Some features are written in C, compiled, and parallelized by execution threads, allowing for reasonable performance on a single compute node.  Due to R's modular nature, users can customize their configuration by installing packages in their personal directory from the [Comprehensive R Archive Network (CRAN)](https://cran.r-project.org/).

You may find useful information in Julie Fortin's blog post, [How to run your R script with Compute Canada](https://www.computecanada.ca/en/blog/how-to-run-your-r-script-with-compute-canada/).


## Interpreter

First, load an R module. Since several versions are available, consult the list by running the command:

```bash
[name@server ~]$ module spider r
```

To load a particular R module, use a variation of the command:

```bash
[name@server ~]$ module load gcc/9.3.0 r/4.0.2
```

For more information, see [Using Modules](link-to-using-modules-page).

You can now start the interpreter and enter R code into this environment:

```bash
[name@server ~]$ R
R version 4.0.2 (2020-06-22) -- "Taking Off Again"
Copyright (C) 2020 The R Foundation for Statistical Computing
Platform: x86_64-pc-linux-gnu (64-bit)

R is free software and comes with ABSOLUTELY NO WARRANTY.
You are welcome to redistribute it under certain conditions.
Type 'license()' or 'licence()' for distribution details.

Natural language support but running in an English locale

R is a collaborative project with many contributors.
Type 'contributors()' for more information and 'citation()' on how to cite R or R packages in publications.

Type 'demo()' for some demos, 'help()' for on-line help, or 'help.start()' for an HTML browser interface to help.
Type 'q()' to quit R.

> values <- c(3, 5, 7, 9)
> values
[1] 3 5 7 9
> q()
```

To run R scripts, use the `Rscript` command followed by the file containing the R commands:

```bash
[name@server ~]$ Rscript computation.R
```

This command will automatically pass the appropriate options for batch processing, namely `--slave` and `--no-restore`, to the R interpreter. These options will prevent the creation of unnecessary workspace files with `--no-save` during batch processing.

Calculations lasting more than two or three minutes should not be run by a compute node but submitted to the scheduler.

Here is an example of a simple script:

**File: job.sh**

```bash
#!/bin/bash
#SBATCH --account=def-someacct   # replace this with your own account
#SBATCH --mem-per-cpu=2000M      # memory; default unit is megabytes
#SBATCH --time=0-00:15           # time (DD-HH:MM)
module load gcc/9.3.0 r/4.0.2
# Adjust version and add the gcc module used for installing packages.
Rscript computation.R
```

For more information, see [Running Jobs](link-to-running-jobs-page).


## Installing R Packages

### `install.packages()`

To install packages from CRAN, you can use `install.packages` in an interactive R session on a login node. Since the compute nodes of most of our clusters do not have internet access, it is not possible to install R packages in a batch job or an interactive job. Because several R packages are developed with the GNU compiler family, we recommend loading a `gcc` module before installing them and always using the same version of `gcc`.

```bash
[name@server ~]$ module load gcc/9.3.0 r/4.0.2
```

#### Installation for a Particular Version of R

For example, to install the `sp` package, which provides classes and methods for spatial data, use this command on a login node:

```bash
[name@server ~]$ R
[...]
> install.packages('sp', repos='https://cloud.r-project.org/')
```

If the `repos` argument is not specified, you will be asked to select a mirror for the download. Ideally, this mirror will be geographically close to the cluster you are using.

Before installation, some packages require the definition of the `TMPDIR` environment variable.


#### Installation for One or More Versions of R

Specify the local directory, depending on the R module that is loaded.

```bash
[name@server ~]$ mkdir -p ~/.local/R/$EBVERSIONR/
[name@server ~]$ export R_LIBS=~/.local/R/$EBVERSIONR/
```

Install the package:

```bash
[name@server ~]$ R -e 'install.packages("sp", repos="https://cloud.r-project.org/")'
```

In the submission script, you must then load the desired R module and configure the local directory for the library with `export R_LIBS=~/.local/R/$EBVERSIONR/`.


### Dependencies

Some packages use libraries that are already installed on our clusters. If the library is in the list of [available software](link-to-available-software-page), load the appropriate module before installing the package.

For example, the `rgdal` package uses the `gdal` library. By running the command `module spider gdal/2.2.1`, we see that the `nixpkgs` and `gcc` modules are required. If you have loaded `gcc` as indicated above, these two modules should already be loaded. Check this with the command:

```bash
[name@server ~]$ module list
```

If the installation of a package fails, pay attention to the error message, which may indicate other required modules. For more information on `module` commands, see [Using Modules](link-to-using-modules-page).


### Downloading Packages

If you are looking to install a package that you have downloaded, i.e., you did not use `install.packages()`, you can install it as follows. For example, with the package `archive_package.tgz`, you would run the following command in the interpreter (shell):

```bash
[name@server ~]$ R CMD INSTALL -l 'path for your local (home) R library' archive_package.tgz
```


## System Calls

The R command `system()` allows you to execute commands in the active environment, inside R; this can cause problems on our clusters because R gives an incorrect value to the `LD_LIBRARY_PATH` environment variable. Instead, use the syntax:

```r
system("LD_LIBRARY_PATH=$RSNT_LD_LIBRARY_PATH <my system call>")
```

in your system calls.


## Arguments Passed to an R Script

It can sometimes be useful to pass parameters as arguments to an R script to avoid having to modify the script for several similar tasks or having to manage several copies of the same script. This can be used to specify numerical parameters or the names of input or output files. For example, instead of using syntax like:

**File: filename**

```r
filename = "input.csv"
iterations = 5
```

and changing the code each time a parameter is modified, the parameters can be passed to the script at the beginning with:

```bash
[name@server ~]$ Rscript myscript.R input_1.csv 5
```

and subsequently:

```bash
[name@server ~]$ Rscript myscript.R input_2.csv 10
```

In the next example, there must be exactly two arguments. The first should be a string representing the name of the variable and the second should be the number of the variable.

**File: arguments_test.R**

```r
args = commandArgs(trailingOnly = TRUE)
# test if there is at least two arguments: if not, return an error
if (length(args) < 2) {
  stop("At least two arguments must be supplied ('name' (text) and 'numer' (integer) )", call. = FALSE)
}
name <- args[1]
# read first argument as string
number <- as.integer(args[2])
# read second argument as integer
print(paste("Processing with name:'", name, "' and number:'", number, "'", sep = ''))
```

This script can be used as follows:

```bash
[name@server ~]$ Rscript arguments_test.R Hello 42
[1] "Processing with name:'Hello' and number:'42'"
```


## Parallelization

If the processors of our clusters are unremarkable, what makes these supercomputers interesting is that they offer thousands of CPUs on a very high-performance network. To take advantage of this, you must use parallel programming. However, before allocating a lot of time and effort to parallelizing your R code, make sure that your sequential implementation is as efficient as possible. As in any interpreted language, significant bottlenecks are caused by loops and especially nested loops, which impacts performance. Whenever possible, try to use vector functions and other more functional elements like the `apply` family of functions and the `ifelse` function. You will often gain performance by eliminating a loop rather than parallelizing its execution with multiple CPU cores.

The [CRAN Task View on High-Performance and Parallel Computing with R](link-to-cran-task-view) mentions a large number of packages that can be used with R for parallel programming.

You will find an excellent overview and advice in the content of the Compute Ontario colloquium of October 11, 2023, entitled [High-Performance Computing in R (slides)](link-to-compute-ontario-colloquium).

You will find other information and examples in the subsections below.


**Terminology:** In our documentation, the terms *node* and *host* are sometimes used to refer to a separate computer; a grouping of *nodes* or *hosts* constitutes a *cluster*.  *Node* often designates a *worker process*; a grouping of these processes constitutes a *cluster*.  Take for example the following quote: “Following `snow`, a pool of worker processes listening via sockets for commands from the master is called a 'cluster' of nodes.”<sup>[1]</sup>


### `doParallel` and `foreach`

#### Usage

`foreach` can be seen as a unified interface for all backends like `doMC`, `doMPI`, `doParallel`, `doRedis`, etc., and works on all platforms provided that the backend is functional. `doParallel` acts as an interface between `foreach` and the parallel package and can be loaded alone. Some [known performance issues](link-to-performance-issues) occur with `foreach` when executing a very large number of very small tasks. Note that the simple example below does not use the `foreach()` call optimally.

Register the backend by indicating the number of cores available. If the backend is not registered, `foreach` assumes that the number of cores is 1 and executes the iterations sequentially.

The general method for using `foreach` is:

1. Load `foreach` and the backend package;
2. Register the backend package;
3. Call `foreach()` leaving it on the same line as the `%do%` (serial) or `%dopar%` operator.


#### Execution

1. Place the R code in a script file, here the file `test_foreach.R`.

**File: test_foreach.R**

```r
# library(foreach) # optional if doParallel is used
library(doParallel)

# a very simple function
test_func <- function(var1, var2) {
  # some heavy workload
  sum <- 0
  for (i in c(1:3141593)) {
    sum <- sum + var1 * sin(var2 / i)
  }
  return(sqrt(sum))
}

# we will iterate according to two sets of values that you can modify to test the operation of foreach
var1.v = c(1:8)
var2.v = seq(0.1, 1, length.out = 8)

# The SLURM_CPUS_PER_TASK environment variable contains the number of cores per task.
# It is defined by SLURM.
# Avoid setting a number of cores manually in the source code.
ncores = Sys.getenv("SLURM_CPUS_PER_TASK")
registerDoParallel(cores = ncores)
# Requests ncores "Parallel Workers"
print(ncores)
# Displays the number of cores available and requested
getDoParWorkers()
# Displays the number of current "Parallel Workers"
# attention! foreach() and %dopar% must be on the same line of code!
foreach(var1 = var1.v, .combine = rbind) %:%
  foreach(var2 = var2.v, .combine = rbind) %dopar% {
    test_func(var1 = var1, var2 = var2)
  }
```

2. Copy the following into the `job_foreach.sh` script.

**File: job_foreach.sh**

```bash
#!/bin/bash
#SBATCH --account=def-someacct   # replace this with your supervisors account
#SBATCH --nodes=1                # number of node MUST be 1
#SBATCH --cpus-per-task=4        # number of processes
#SBATCH --mem-per-cpu=2048M      # memory; default unit is megabytes
#SBATCH --time=0-00:15           # time (DD-HH:MM)
#SBATCH --mail-user=yourname@someplace.com # Send email updates to you or someone else
#SBATCH --mail-type=ALL          # send an email in all cases (job started, job ended, job aborted)
module load gcc/9.3.0 r/4.0.2
export R_LIBS=~/local/R_libs/
R CMD BATCH --no-save --no-restore test_foreach.R
```

3. Submit the job.

```bash
[name@server ~]$ sbatch job_foreach.sh
```

For more information on how to submit jobs, see [Running Jobs](link-to-running-jobs-page).


### `doParallel` and `makeCluster`

#### Usage

It is necessary to register the backend by giving it the names of the nodes, multiplied by the desired number of processes. For example, we would create a cluster composed of the hosts `node1 node1 node2 node2`. The cluster type `PSOCK` executes commands via SSH connections to the nodes.


#### Execution

1. Place the R code in a script file, here `test_makecluster.R`.

**File: test_makecluster.R**

```r
library(doParallel)

# Create an array from the NODESLIST environment variable
nodeslist = unlist(strsplit(Sys.getenv("NODESLIST"), split = " "))

# Create the cluster with the nodes name. One process per count of node name.
# nodeslist = node1 node1 node2 node2, means we are starting 2 processes on node1, likewise on node2.
cl = makeCluster(nodeslist, type = "PSOCK")
registerDoParallel(cl)

# Compute (Source : https://cran.r-project.org/web/packages/doParallel/vignettes/gettingstartedParallel.pdf)
x <- iris[which(iris[, 5] != "setosa"), c(1, 5)]
trials <- 10000
foreach(icount(trials), .combine = cbind) %dopar% {
  ind <- sample(100, 100, replace = TRUE)
  result1 <- glm(x[ind, 2] ~ x[ind, 1], family = binomial(logit))
  coefficients(result1)
}

# Don't forget to release resources
stopCluster(cl)
```

2. Copy the following lines into a script to submit the job, here `job_makecluster.sh`.

**File: job_makecluster.sh**

```bash
#!/bin/bash
#SBATCH --account=def-someacct  # à remplacer par un compte approprié
#SBATCH --ntasks=4              # nombre de processus
#SBATCH --mem-per-cpu=512M      # mémoire par coeur CPU; valeur en Mo par défaut
#SBATCH --time=00:05:00         # temps (HH:MM:SS)
module load gcc/9.3.0 r/4.0.2
# Export the nodes names.
# If all processes are allocated on the same node, NODESLIST contains : node1 node1 node1 node1
# Cut the domain name and keep only the node name
export NODESLIST=$(echo $(srun hostname | cut -f 1 -d '.'))
R -f test_makecluster.R
```

In this example, the scheduler could place the four processes on a single node. This may be suitable, but if you want to prove that the same task can be processed if the processes are placed on different nodes, add the line `#SBATCH --ntasks-per-node=2`.

3. Submit the job with:

```bash
[name@server ~]$ sbatch job_makecluster.sh
```

For more information on how to submit a job, see [Running Jobs](link-to-running-jobs-page).


### Rmpi

The following instructions do not work on Cedar; use another cluster instead.


#### Installation

The following procedure installs `Rmpi`, a wrapper for MPI routines that allows R to run in parallel.

1. See the available R modules with the command `module spider r`.
2. Select the version and load the appropriate OpenMPI module. In our example, version 4.0.3 is used so that the processes run correctly.

```bash
module load gcc/11.3.0
module load r/4.2.1
module load openmpi/4.1.4
```

3. Download the latest version of Rmpi, replacing the version number as needed.

```bash
wget https://cran.r-project.org/src/contrib/Rmpi_0.6-9.2.tar.gz
```

4. Specify the directory where you want to copy the files; you must have write permission for this directory. The directory name can be modified.

```bash
mkdir -p ~/local/R_libs/
export R_LIBS=~/local/R_libs/
```

5. Run the installation command.

```bash
R CMD INSTALL --configure-args="--with-Rmpi-include=$EBROOTOPENMPI/include   --with-Rmpi-libpath=$EBROOTOPENMPI/lib --with-Rmpi-type='OPENMPI' " Rmpi_0.6-9.2.tar.gz
```

Pay attention to the error message that appears when the installation of a package fails; it may indicate other modules that may be necessary.


#### Execution

1. Place the R code in a script file, here the file `test.R`.

**File: test.R**

```r
#Tell all slaves to return a message identifying themselves.
library("Rmpi")
sprintf("TEST mpi.universe.size() =  %i", mpi.universe.size())
ns <- mpi.universe.size() - 1
sprintf("TEST attempt to spawn %i slaves", ns)
mpi.spawn.Rslaves(nslaves = ns)
mpi.remote.exec(paste("I am", mpi.comm.rank(), "of", mpi.comm.size()))
mpi.remote.exec(paste(mpi.comm.get.parent()))
#Send execution commands to the slaves
x <- 5
#These would all be pretty correlated one would think
x <- mpi.remote.exec(rnorm, x)
length(x)
x
mpi.close.Rslaves()
mpi.quit()
```

2. Copy the following into the `job.sh` script.

**File: job.sh**

```bash
#!/bin/bash
#SBATCH --account=def-someacct   # à remplacer par un compte approprié
#SBATCH --ntasks=5               # nombre de processus MPI
#SBATCH --mem-per-cpu=2048M      # mémoire par coeur CPU; valeur en Mo par défaut
#SBATCH --time=0-00:15           # temps (JJ-HH:MM)
module load gcc/11.3.0
module load r/4.2.1
module load openmpi/4.1.4
export R_LIBS=~/local/R_libs/
mpirun -np 1 R CMD BATCH test.R test.txt
```

3. Submit the job.

```bash
sbatch job.sh
```

For more information on how to submit jobs, see [Running Jobs](link-to-running-jobs-page).

[^1]: https://stat.ethz.ch/R-manual/R-devel/library/parallel/doc/parallel.pdf


**(Remember to replace placeholder links like `link-to-using-modules-page` with actual links to your documentation.)**
