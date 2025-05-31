# ARM DDT and MAP on Graham and Niagara Clusters

## Introduction

ARM DDT (formerly Allinea DDT) is a commercial parallel debugger with a graphical user interface.  It debugs serial, MPI, multi-threaded, and CUDA programs (or combinations thereof) written in C, C++, and FORTRAN.

MAP, an efficient parallel profiler, is another useful tool from ARM (formerly Allinea).

**Available Modules:**

* **Graham:** `ddt-cpu` (CPU debugging and profiling), `ddt-gpu` (GPU or mixed CPU/GPU debugging)
* **Niagara:** `ddt`

Because this is a GUI application, log in using `ssh -Y` and an SSH client like MobaXTerm (Windows) or XQuartz (Mac) for proper X11 tunneling.

DDT and MAP are typically used interactively through their GUI, usually via the `salloc` command (see below). MAP can also be used non-interactively with `sbatch`.

**License Limit:** The current license limits DDT/MAP to a maximum of 512 CPU cores across all users, and DDT-GPU to 8 GPUs.


## Usage

### CPU-only code, no GPUs

1. Allocate nodes for debugging/profiling using `salloc`:

   ```bash
   salloc --x11 --time=0-1:00 --mem-per-cpu=4G --ntasks=4
   ```

2. Load the appropriate module:

   ```bash
   module load ddt-cpu
   ```

3. Run the `ddt` or `map` command:

   ```bash
   ddt path/to/code
   map path/to/code
   ```

   Ensure the MPI implementation is the default OpenMPI in the DDT/MAP window before pressing "Run". If not, use the "Change" button to select the correct option and specify the number of CPU cores.

4. Exit the shell to terminate the allocation.

**IMPORTANT:** Current DDT and OpenMPI versions have a compatibility issue affecting message queue display ("Tools" menu).  A workaround is to execute this command *before* running DDT:

```bash
export OMPI_MCA_pml=ob1
```

This workaround may slow down MPI code; use it only during debugging.


### CUDA code

1. Allocate nodes using `salloc`:

   ```bash
   salloc --x11 --time=0-1:00 --mem-per-cpu=4G --ntasks=1 --gres=gpu:1
   ```

2. Load the appropriate module:

   ```bash
   module load ddt-gpu
   ```

   If this fails, load an older OpenMPI version as suggested, then reload `ddt-gpu`:

   ```bash
   module load openmpi/2.0.2
   module load ddt-gpu
   ```

3. Load a CUDA module:

   ```bash
   module load cuda
   ```

4. Run the `ddt` command:

   ```bash
   ddt path/to/code
   ```

   If DDT reports a CUDA driver/toolkit version mismatch, execute this command (using the correct version) and run DDT again:

   ```bash
   export ALLINEA_FORCE_CUDA_VERSION=10.1
   ```

5. Exit the shell to terminate the allocation.


### Using VNC to fix lag (gra-vdi.computecanada.ca program on graham.computecanada.ca)

X11 forwarding is sensitive to latency.  For remote use, the DDT interface may be laggy.  VNC can fix this. Follow the instructions on the [VNC page](link-to-vnc-page-here) to set up a VNC session.

If your VNC session is on the compute node, start your program under DDT as described above.  If it's on the login node or graham vdi node, manually launch the job:

1. From the DDT startup screen, select "manually launch backend yourself".
2. Enter job information and press "listen".
3. Press "help" (next to "waiting for you to start the job...") to get the command to start your job.
4. Allocate a job on the cluster and start your program as directed.  Example (replace `$USER` and `$PROGRAM`):

   ```bash
   salloc ...
   /cvmfs/restricted.computecanada.ca/easybuild/software/2020/Core/allinea/20.2/bin/forge-client --ddtsessionfile /home/$USER/.allinea/session/gra-vdi3-1 $PROGRAM ...
   ```


## Known Issues

On graham, if X11 fails, change your home directory permissions to allow only your access:

1. Check current permissions: `ls -ld /home/$USER` (output should start with `drwx------`).
2. Remove group and other user read/execute permissions: `chmod go-rx /home/$USER`
3. (Optional) Restore permissions after using DDT (see [Sharing_data](link-to-sharing-data-page-here)).


## See Also

* ["Debugging your code with DDT"](link-to-video-here), a 55-minute video tutorial.

