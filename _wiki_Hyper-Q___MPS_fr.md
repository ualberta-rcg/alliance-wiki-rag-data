# Hyper-Q / MPS

Hyper-Q (or MPS for Multi-Process Service) is a feature of NVIDIA GPUs compatible with CUDA version 3.5 and later [1], which is the case for all our general-purpose clusters (Béluga, Cedar, Graham, and Narval).

According to NVIDIA's documentation (free translation):  "The MPS execution architecture is designed to transparently allow the use of parallel and cooperative CUDA applications (as are typically MPI tasks) by leveraging the Hyper-Q features of the latest NVIDIA GPUs (Kepler and later). Hyper-Q allows CUDA kernels to be processed simultaneously on the same GPU, which improves performance when the GPU's computing capacity is underutilized by a single process."

Our tests have shown that MPS can increase the number of floating-point operations performed per second (flops) even when the GPU is shared between unrelated CPU processes. This means that MPS is the ideal feature for CUDA applications that process problems whose relatively small size prevents them from fully utilizing modern GPUs with thousands of cores.

MPS is not enabled by default, but simply launch the following commands before starting your CUDA application:

```bash
[name@server ~]$ export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
[name@server ~]$ export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
[name@server ~]$ nvidia-cuda-mps-control -d
```

You can then use MPS if you have more than one CPU thread accessing the GPU. This occurs when you run a hybrid MPI/CUDA application, a hybrid OpenMP/CUDA application, or multiple sequential CUDA applications (GPU farming).

For more information on MPS, see the NVIDIA documentation.


## GPU Farming

The MPS feature is very useful for running multiple instances of the same CUDA application when it is too small to fully occupy a modern GPU. MPS allows you to run all these instances, provided there is sufficient GPU memory. In many cases, the production of results could be greatly increased.

The following script is an example of how to configure GPU farming:

**File: script.sh**

```bash
#!/bin/bash
#SBATCH --gpus-per-node=v100:1
#SBATCH --time=0-10:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=8
mkdir -p $HOME/tmp
export CUDA_MPS_LOG_DIRECTORY=$HOME/tmp
nvidia-cuda-mps-control -d
for (( i=0; i<SLURM_CPUS_PER_TASK; i++ ))
do
  echo $i
  ./my_code $i &
done
wait
```

In this example, a V100 GPU is shared by 8 instances of `my_code`, which only takes the loop index `$i` as an argument. Since we request 8 CPU cores (`#SBATCH --cpus-per-task=8`), there is one CPU core for each instance of the application. The two important elements are `&` on the code execution line, which moves the processes to the background, and the `wait` command at the end of the script, which ensures that the GPU farm continues until all background processes are complete.

<sup>See the table of models, architectures and CUDA compute capabilities at [https://en.wikipedia.org/wiki/Nvidia_Tesla](https://en.wikipedia.org/wiki/Nvidia_Tesla).</sup>
