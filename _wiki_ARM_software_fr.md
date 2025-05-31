# ARM Software (French)

This page is a translated version of the page ARM software and the translation is 100% complete.

Other languages: English, français


## Introduction

ARM DDT (formerly Allinea DDT) is a powerful commercial tool for debugging parallel code.  With a graphical user interface, it is used for debugging serial, MPI, multithreaded, CUDA, or any combination of these application types in C, C++, and Fortran. ARM also produces the MAP profiler for parallel code.

The following modules are available on Graham:

*   `ddt-cpu`: To debug and profile code on the CPU.
*   `ddt-gpu`: To debug code on GPUs or CPU/GPU.

The following module is available on Niagara:

*   `ddt`

Since this is a graphical application, connect with `ssh -Y` and use an SSH client like MobaXTerm (on Windows) or XQuartz (on Mac) to ensure X11 forwarding.

DDT and MAP are generally used interactively via the user interface with the `salloc` command. The MAP profiler can also be used non-interactively by submitting jobs to the scheduler with the `sbatch` command.

With the current license, DDT/MAP can concurrently use up to 512 cores for all users, while DDT-GPU can only use 8 GPUs.


## Utilisation

### Avec CPU seulement (aucun GPU)

1.  Allocate the node(s) on which you want to debug or profile; this opens a shell session on the node(s) in question.

    ```bash
    salloc --x11 --time=0-1:00 --mem-per-cpu=4G --ntasks=4
    ```

2.  Load the appropriate module, for example:

    ```bash
    module load ddt-cpu
    ```

3.  Launch the `ddt` or `map` command.

    ```bash
    ddt path/to/code
    map path/to/code
    ```

    Before clicking "Run", ensure that the default MPI implementation in the DDT/MAP window is OpenMPI. If not, click the "Change" button (near "Implementation") and select the correct option from the dropdown menu. Also specify the number of CPU cores in this window.

4.  Exit the shell to end the allocation.


**IMPORTANT:** Current versions of DDT and OpenMPI have a compatibility issue that prevents DDT from displaying message queues (Tools dropdown menu). To work around this, run the command:

```bash
export OMPI_MCA_pml=ob1
```

Since this can slow down MPI code, only use this command during debugging.


### CUDA

1.  Allocate the node(s) on which you want to debug or profile with `salloc`; this opens a shell session on the node(s) in question.

    ```bash
    salloc --x11 --time=0-1:00 --mem-per-cpu=4G --ntasks=1 --gres=gpu:1
    ```

2.  Load the appropriate module, for example:

    ```bash
    module load ddt-gpu
    ```

    You may be asked to load a previous version of OpenMPI first. In this case, reload the OpenMPI module using the suggested command and then reload the `ddt-gpu` module.

    ```bash
    module load openmpi/2.0.2
    module load ddt-gpu
    ```

3.  Make sure a CUDA module is loaded.

    ```bash
    module load cuda
    ```

4.  Launch the `ddt` command.

    ```bash
    ddt path/to/code
    ```

    If DDT encounters difficulties due to incompatibility between the CUDA driver and the toolkit version, run the following command and launch DDT again (using the same version as in the command).

    ```bash
    export ALLINEA_FORCE_CUDA_VERSION=10.1
    ```

5.  Exit the shell to end the allocation.


### Problème de latence

DDT on `gra-vdi.computecanada.ca`, program on `graham.computecanada.ca`

The above instructions use X-11 forwarding, which is very sensitive to packet latency issues. If you are not on the same campus as the cluster, the DDT interface will likely be slow and frustrating. To remedy this, use DDT under VNC.

To do this, prepare a VNC session. If your VNC session is on the compute node, you can start your `ddt` program directly as described above. If your VNC session is on the login node or if you are using Graham's vdi node, you must launch the task from the DDT startup screen as follows:

*   Select the "manually launch backend yourself" option for task launching;
*   Enter the information for your task and click the "listen" button;
*   Click the "help" button to the right of "waiting for you to start the job...".

This will give you the command to use to launch your task. Allocate a task on the cluster and start your program as indicated. In the following example, `$USER` is your username and `$PROGRAM` is the command to start your program.

```bash
[name@cluster-login:~]$ salloc ...
[name@cluster-node:~]$ /cvmfs/restricted.computecanada.ca/easybuild/software/2020/Core/allinea/20.2/bin/forge-client --ddtsessionfile /home/$USER/.allinea/session/gra-vdi3-1 $PROGRAM ...
```


## Problèmes connus

If you have problems with X11 on Graham, change the permissions of your `/home` directory so that only you have access.

First, check (and note if necessary) the current permissions with:

```bash
ls -ld /home/$USER
```

The result should start with `drwx------`. If some hyphens are replaced by letters, your group and other users have read, write (unlikely), or execute permissions for your directory.

The following command will remove read and execute permissions for the group and other users:

```bash
chmod go-rx /home/$USER
```

When you are finished with DDT, you can optionally revert to the previous permissions (assuming you noted them). For more information, see Data Sharing.


## Pour plus d'information

*   "Debugging your code with DDT" (55-minute video)
*   Parallel debugging with DDT (short tutorial)


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=ARM_software/fr&oldid=157696](https://docs.alliancecan.ca/mediawiki/index.php?title=ARM_software/fr&oldid=157696)"
