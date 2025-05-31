# Chapel

Chapel is a high-level, general-purpose, compiled parallel programming language with built-in abstractions for shared and distributed memory parallelism. Chapel offers two styles of parallel programming: (1) task parallelism, where parallelism is achieved through programmatically specified tasks, and (2) data parallelism, where parallelism is achieved by performing the same computations on subsets of data that may reside in the shared memory of a single node or be distributed across multiple nodes.

These high-level abstractions make Chapel an ideal tool for learning parallel programming for high-performance computing.  The language is incredibly intuitive and strives to merge the ease of use of Python with the performance of traditional compiled languages such as C and Fortran. Parallel blocks that typically take tens of lines of MPI code can be expressed in just a few lines of Chapel code. Chapel is open source and can run on any Unix-like operating system, with hardware support ranging from laptops to large HPC systems.

Chapel has a relatively small user base, so many libraries that exist for C, C++, and Fortran have not yet been implemented in Chapel. Hopefully, this will change in the coming years if Chapel adoption continues to gain momentum within the HPC community.

For more information, see our [Chapel webinars](link-to-web-page-here).


## Simple Computations

The `chapel-multicore` module is used on our general-purpose clusters with a single node and shared memory only. You can use `salloc` to test if your code works sequentially.

```bash
[name@server ~]$ module load gcc/12.3 chapel-multicore/2.3.0
[name@server ~]$ salloc --time=0:30:0 --ntasks=1 --mem-per-cpu=3600 --account=def-someprof
[name@server ~]$ chpl test.chpl -o test
[name@server ~]$ ./test
```

Or with multiple cores on the same node:

```bash
[name@server ~]$ module load gcc/12.3 chapel-multicore/2.3.0
[name@server ~]$ salloc --time=0:30:0 --ntasks=1 --cpus-per-task=3 --mem-per-cpu=3600 --account=def-someprof
[name@server ~]$ chpl test.chpl -o test
[name@server ~]$ ./test
```

For production jobs, please prepare a [job submission script](link-to-job-submission-script-here) and submit it with `sbatch`.


## Distributed Computations

For jobs with multiple nodes and hybrid memory (shared and distributed) on our InfiniBand clusters, load the `chapel-ofi` module.

The following code prints basic information about the available nodes in your job.

**File: `probeLocales.chpl`**

```chapel
use MemDiagnostics;

for loc in Locales do
  on loc {
    writeln("locale #", here.id, "...");
    writeln("  ...is named: ", here.name);
    writeln("  ...has ", here.numPUs(), " processor cores");
    writeln("  ...has ", here.physicalMemory(unit=MemUnits.GB, retType=real), " GB of memory");
    writeln("  ...has ", here.maxTaskPar, " maximum parallelism");
  }
```

To run this code on the June InfiniBand cluster, you need to load the `chapel-ucx` module.

```bash
[name@server ~]$ module load gcc/12.3 chapel-ucx/2.3.0
[name@server ~]$ salloc --time=0:30:0 --nodes=4 --cpus-per-task=3 --mem-per-cpu=3500 --account=def-someprof
```

Once the [interactive job](link-to-interactive-job-page-here) is launched, you can compile and run your code from the prompt on the first allocated compute node.

```bash
[name@server ~]$ chpl --fast probeLocales.chpl -o probeLocales
[name@server ~]$ ./probeLocales -nl 4
```

For production jobs, please prepare a [job submission script](link-to-job-submission-script-here) and submit the job with `sbatch`.


## Distributed Computation with NVIDIA GPUs

To use a GPU, load the `chapel-ucx-cuda` module, which supports NVIDIA GPUs on our InfiniBand clusters.

This is basic code to use a GPU with Chapel.

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
  foreach a in A do
    // thread parallelism on a CPU or a GPU
    a += 1;
  writeln(A);
}

stopGpuDiagnostics();
writeln(getGpuDiagnostics());
```

To run this code on an InfiniBand cluster, load the `chapel-ucx-cuda` module.

```bash
[name@server ~]$ module load gcc/12.3 cuda/12.2 chapel-ucx-cuda/2.3.0
[name@server ~]$ salloc --time=0:30:0 --mem-per-cpu=3500 --gpus-per-node=1 --account=def-someprof
```

Once the [interactive job](link-to-interactive-job-page-here) is launched, you can compile and run your code from the prompt on the allocated compute node.

```bash
[name@server ~]$ chpl --fast probeGPU.chpl
[name@server ~]$ ./probeGPU -nl 1
```

For production jobs, please prepare a [job submission script](link-to-job-submission-script-here) and submit the job with `sbatch`.


**(Remember to replace the bracketed placeholders like `[link-to-web-page-here]` with the actual links.)**
