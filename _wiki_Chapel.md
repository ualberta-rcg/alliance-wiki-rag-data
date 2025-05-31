# Chapel

**Other languages:** English, français

Chapel is a general-purpose, compiled, high-level parallel programming language with built-in abstractions for shared- and distributed-memory parallelism. There are two styles of parallel programming in Chapel:

1.  **Task parallelism**, where parallelism is driven by programmer-specified tasks.
2.  **Data parallelism**, where parallelism is driven by applying the same computation on subsets of data elements, which may be in the shared memory of a single node, or distributed over multiple nodes.

These high-level abstractions make Chapel ideal for learning parallel programming for a novice HPC user. Chapel is incredibly intuitive, striving to merge the ease-of-use of Python and the performance of traditional compiled languages such as C and Fortran. Parallel blocks that typically take tens of lines of MPI code can be expressed in only a few lines of Chapel code. Chapel is open source and can run on any Unix-like operating system, with hardware support from laptops to large HPC systems.

Chapel has a relatively small user base, so many libraries that exist for C, C++, Fortran have not yet been implemented in Chapel. Hopefully, that will change in coming years if Chapel adoption continues to gain momentum in the HPC community.

For more information, please watch our [Chapel webinars](link-to-webinars).


## Single-locale Chapel

Single-locale (single node; shared-memory only) Chapel on our general-purpose clusters is provided by the module `chapel-multicore`. You can use `salloc` to test Chapel codes either in serial:

```bash
[name@server ~]$ module load gcc/12.3 chapel-multicore/2.3.0
[name@server ~]$ salloc --time=0:30:0 --ntasks=1 --mem-per-cpu=3600 --account=def-someprof
[name@server ~]$ chpl test.chpl -o test
[name@server ~]$ ./test
```

or on multiple cores on the same node:

```bash
[name@server ~]$ module load gcc/12.3 chapel-multicore/2.3.0
[name@server ~]$ salloc --time=0:30:0 --ntasks=1 --cpus-per-task=3 --mem-per-cpu=3600 --account=def-someprof
[name@server ~]$ chpl test.chpl -o test
[name@server ~]$ ./test
```

For production jobs, please write a job submission script and submit it with `sbatch`.


## Multi-locale Chapel

Multi-locale (multiple nodes; hybrid shared- and distributed-memory) Chapel on our InfiniBand clusters is provided by the module `chapel-ucx`.

Consider the following Chapel code printing basic information about the nodes available inside your job:

**File: `probeLocales.chpl`**

```chapel
use MemDiagnostics;

for loc in Locales do on loc {
  writeln("locale #", here.id, "...");
  writeln("  ...is named: ", here.name);
  writeln("  ...has ", here.numPUs(), " processor cores");
  writeln("  ...has ", here.physicalMemory(unit=MemUnits.GB, retType=real), " GB of memory");
  writeln("  ...has ", here.maxTaskPar, " maximum parallelism");
}
```

To run this code on an InfiniBand cluster, you need to load the `chapel-ucx` module:

```bash
[name@server ~]$ module load gcc/12.3 chapel-ucx/2.3.0
[name@server ~]$ salloc --time=0:30:0 --nodes=4 --cpus-per-task=3 --mem-per-cpu=3500 --account=def-someprof
```

Once the interactive job starts, you can compile and run your code from the prompt on the first allocated compute node:

```bash
[name@server ~]$ chpl --fast probeLocales.chpl
[name@server ~]$ ./probeLocales -nl 4
```

For production jobs, please write a Slurm submission script and submit your job with `sbatch` instead.


## Multi-locale Chapel with NVIDIA GPU support

To enable GPU support, please use the module `chapel-ucx-cuda`. It adds NVIDIA GPU support to multi-locale Chapel on our InfiniBand clusters.

Consider the following basic Chapel GPU code:

**File: `probeGPU.chpl`**

```chapel
use GpuDiagnostics;

startGpuDiagnostics();
writeln("Locales: ", Locales);
writeln("Current locale: ", here, " named ", here.name, " with ", here.maxTaskPar, " CPU cores", " and ", here.gpus.size, " GPUs");

// same code can run on GPU or CPU
var operateOn = if here.gpus.size > 0 then here.gpus[0] // use the first GPU
                 else here; // use the CPU
writeln("operateOn: ", operateOn);

on operateOn {
  var A: [1..10] int;
  @assertOnGpu
  foreach a in A do // thread parallelism on a CPU or a GPU
    a += 1;
  writeln(A);
}

stopGpuDiagnostics();
writeln(getGpuDiagnostics());
```

To run this code on an InfiniBand cluster, you need to load the `chapel-ucx-cuda` module:

```bash
[name@server ~]$ module load gcc/12.3 cuda/12.2 chapel-ucx-cuda/2.3.0
[name@server ~]$ salloc --time=0:30:0 --mem-per-cpu=3500 --gpus-per-node=1 --account=def-someprof
```

Once the interactive job starts, you can compile and run your code from the prompt on the allocated compute node:

```bash
[name@server ~]$ chpl --fast probeGPU.chpl
[name@server ~]$ ./probeGPU -nl 1
```

For production jobs, please write a Slurm submission script and submit your job with `sbatch` instead.


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Chapel&oldid=171198](https://docs.alliancecan.ca/mediawiki/index.php?title=Chapel&oldid=171198)"
