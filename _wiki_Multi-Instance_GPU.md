# Multi-Instance GPU (MIG)

Many programs are unable to fully use modern GPUs such as Nvidia A100s and H100s. Multi-Instance GPU (MIG) is a technology that allows partitioning a single GPU into multiple instances, making each one a completely independent virtual GPU. Each of the GPU instances gets a portion of the original GPU's computational resources and memory, all detached from the other instances by on-chip protections. Using GPU instances is less wasteful, and usage is billed accordingly. Jobs submitted on such instances use less of your allocated priority compared to a full GPU; you will then be able to execute more jobs and have shorter wait time.

## Choosing between a full GPU and a GPU instance

Jobs that use less than half of the computing power of a full GPU and less than half of the available memory should be evaluated and tested on an instance. In most cases, these jobs will run just as fast and consume less than half of the computing resource.

See section [Finding which of your jobs should use an instance](#finding-which-of-your-jobs-should-use-an-instance) for more details.

## Limitations

The MIG technology does not support CUDA Inter-Process Communication (IPC), which optimizes data transfers between GPUs over NVLink and NVSwitch. This limitation also reduces communication efficiency between instances. Consequently, launching an executable on more than one instance at a time does *not* improve performance and should be avoided.

Please note that graphic APIs are not supported (for example, OpenGL, Vulkan, etc.); see [Application Considerations](Application Considerations).

GPU jobs requiring many CPU cores may also require a full GPU instead of an instance. The maximum number of CPU cores per instance depends on the number of cores per full GPU and on the configured MIG profiles. Both vary between clusters and between GPU nodes in a cluster.

## Available configurations

As of Dec. 2024, Narval A100 nodes offer all types of GPU instances. While there are many possible MIG configurations and profiles, only the following profiles have been implemented:

*   `1g.5gb`
*   `2g.10gb`
*   `3g.20gb`
*   `4g.20gb`

The profile name describes the size of the instance. For example, a `3g.20gb` instance has 20 GB of RAM and offers 3/8 of the computing performance of a full A100-40gb GPU. Using less powerful profiles will have a lower impact on your allocation and priority.

On Narval, the recommended maximum number of CPU cores and amount of system memory per instance are:

*   `1g.5gb`: maximum 2 cores and 15GB
*   `2g.10gb`: maximum 3 cores and 31GB
*   `3g.20gb`: maximum 6 cores and 62GB
*   `4g.20gb`: maximum 6 cores and 62GB

To request an instance of a certain profile, your job submission must include the `--gres` parameter:

*   `1g.5gb`: `--gres=gpu:a100_1g.5gb:1`
*   `2g.10gb`: `--gres=gpu:a100_2g.10gb:1`
*   `3g.20gb`: `--gres=gpu:a100_3g.20gb:1`
*   `4g.20gb`: `--gres=gpu:a100_4g.20gb:1`

Note: For the job scheduler on Narval, the prefix `a100_` is required at the beginning of the profile name.

## Job examples

Requesting an instance of power 3/8 and size 20GB for a 1-hour interactive job:

```bash
[name@server ~]$ salloc --account=def-someuser --gres=gpu:a100_3g.20gb:1 --cpus-per-task=2 --mem=40gb --time=1:0:0
```

Requesting an instance of power 4/8 and size 20GB for a 24-hour batch job using the maximum recommended number of cores and system memory:

**File:** `a100_4g.20gb_mig_job.sh`

```bash
#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --gres=gpu:a100_4g.20gb:1
#SBATCH --cpus-per-task=6    # There are 6 CPU cores per 3g.20gb and 4g.20gb on Narval.
#SBATCH --mem=62gb           # There are 62GB GPU RAM per 3g.20gb and 4g.20gb on Narval.
#SBATCH --time=24:00:00
hostname
nvidia-smi
```

## Finding which of your jobs should use an instance

You can find information on current and past jobs on the [Narval usage portal (writing in progress)](Narval usage portal).

Power consumption is a good indicator of the total computing power requested from the GPU. For example, the following job requested a full A100 GPU with a maximum TDP of 400W, but only used 100W on average, which is only 50W more than the idle electric consumption:

[Example GPU Power usage of a job on a full A100 GPU]

GPU functionality utilization may also provide insights on the usage of the GPU in cases where the power consumption is not sufficient. For this example job, GPU utilization graph supports the conclusion of the GPU power consumption graph, in that the job uses less than 25% of the available computing power of a full A100 GPU:

[Example GPU Utilization of a job on a full A100 GPU]

The final metrics to consider are the maximum amount of GPU memory and the average number of CPU cores required to run the job. For this example, the job uses a maximum of 3GB of GPU memory out of the 40GB of a full A100 GPU.

[Example GPU memory usage of a job on a full A100 GPU]

It was also launched using a single CPU core. When taking into account these three last metrics, we can confirm that the job should easily run on a 3g.20GB or 4g.20GB GPU instance with power and memory to spare.

Another way to monitor the usage of a running job is by attaching to the node where the job is currently running and then by using `nvidia-smi` to read the GPU metrics in real time. This will not provide maximum and average values for memory and power usage of the entire job, but it may be helpful to identify and troubleshoot underperforming jobs.

## Can I use multiple instances on the same GPU?

No. While this is possible in principle, we don't support this. If you want to run multiple independent tasks on a GPU, you should use MPS rather than MIG.


**(Retrieved from "https://docs.alliancecan.ca/mediawiki/index.php?title=Multi-Instance_GPU&oldid=175840")**
