# ParaView

## Client-server visualization

**NOTE 1:** An important setting in ParaView's preferences is `Render View -> Remote/Parallel Rendering Options -> Remote Render Threshold`. If you set it to the default (20MB) or similar, small rendering will be done on your computer's GPU, and mouse rotation will be fast. However, anything modestly intensive (under 20MB) will be shipped to your computer, and visualization might be slow depending on your connection. If you set it to 0MB, all rendering will be remote, including rotation. This means you will be using the cluster resources for everything, which is good for large data processing but not for interactivity. Experiment with the threshold to find a suitable value.

**NOTE 2:** ParaView requires the same major version on the local client and the remote host. This prevents incompatibility, which typically shows as a failed handshake when establishing the client-server connection. For example, to use ParaView server version 5.10.0 on the cluster, you need client version 5.10.x on your computer.

Please use the tabs below to select the remote system:

* Cedar
* Graham
* Béluga
* Narval
* Niagara
* Cloud VM

### Client-server visualization on Cedar, Graham, Béluga and Narval

On Cedar, Graham, Béluga, and Narval, you can do client-server rendering on both CPUs (in software) and GPUs (hardware acceleration). Due to additional complications with GPU rendering, we strongly recommend starting with CPU-only visualization, allocating as many cores as necessary to your rendering. The easiest way to estimate the number of necessary cores is to look at the amount of memory you think you will need for your rendering and divide it by ~3.5 GB/core. For example, a 40GB dataset (loaded into memory at once, e.g., a single timestep) would require at least 12 cores just to hold the data. Since software rendering is CPU-intensive, we do not recommend allocating more than 4GB/core.  Additionally, it is important to allocate some memory for filters and data processing (e.g., a structured to unstructured dataset conversion will increase your memory footprint by ~3X); depending on your workflow, you may want to start this rendering with 32 cores or 64 cores. If your ParaView server gets killed when processing these data, you will need to increase the number of cores.

#### CPU-based visualization

You can also do interactive client-server ParaView rendering on cluster CPUs. For some types of rendering, modern CPU-based libraries such as OSPRay and OpenSWR offer performance quite similar to GPU-based rendering. Also, since the ParaView server uses MPI for distributed-memory processing, for very large datasets one can do parallel rendering on a large number of CPU cores, either on a single node or scattered across multiple nodes.

1. First, install on your computer the same ParaView version as the one available on the cluster you will be using; log into Cedar or Graham and start a serial CPU interactive job.

```bash
[name@server ~]$ salloc --time=1:00:00 --ntasks=1 --mem-per-cpu=3600 --account=def-someprof
```

The job should automatically start on one of the CPU interactive nodes.

2. At the prompt that is now running inside your job, load the offscreen ParaView module and start the server.

```bash
[name@server ~]$ module load paraview/5.13.1
```

and then

```bash
[name@server ~]$ pvserver --force-offscreen-rendering
Waiting for client...
Connection URL: cs://cdr774.int.cedar.computecanada.ca:11111
Accepting connection (s): cdr774.int.cedar.computecanada.ca:11111
```

Wait for the server to be ready to accept client connection.

3. Make a note of the node (in this case `cdr774`) and the port (usually 11111) and in another terminal on your computer (on Mac/Linux; in Windows use a terminal emulator) link the port 11111 on your computer and the same port on the compute node (make sure to use the correct compute node).

```bash
[name@computer $] ssh <username>@cedar.computecanada.ca -L 11111:cdr774:11111
```

4. Start ParaView on your computer, go to `File -> Connect` (or click on the green `Connect` button in the toolbar) and click on `Add Server`. You will need to point ParaView to your local port 11111, so you can do something like `name = cedar`, `server type = Client/Server`, `host = localhost`, `port = 11111`; click `Configure`, select `Manual` and click `Save`. Once the remote is added to the configuration, simply select the server from the list and click on `Connect`. The first terminal window that read `Accepting connection` will now read `Client connected`.

5. Open a file in ParaView (it will point you to the remote filesystem) and visualize it as usual.

**NOTE:** An important setting in ParaView's preferences is `Render View -> Remote/Parallel Rendering Options -> Remote Render Threshold`. If you set it to default (20MB) or similar, small rendering will be done on your computer's GPU, the rotation with a mouse will be fast, but anything modestly intensive (under 20MB) will be shipped to your computer and (depending on your connection) visualization might be slow. If you set it to 0MB, all rendering will be remote including rotation, so you will really be using the cluster resources for everything, which is good for large data processing but not so good for interactivity. Experiment with the threshold to find a suitable value.

If you want to do parallel rendering on multiple CPUs, start a parallel job; don't forget to specify the correct maximum walltime limit.

```bash
[name@server ~]$ salloc --time=0:30:00 --ntasks=8 --mem-per-cpu=3600 --account=def-someprof
```

Start the ParaView server with `srun`.

```bash
[name@server ~]$ module load paraview-offscreen/5.13.1
[name@server ~]$ srun pvserver --force-offscreen-rendering
```

To check that you are doing parallel rendering, you can pass your visualization through the Process Id Scalars filter and then color it by "process id".


#### GPU-based ParaView visualization

Cedar and Graham have a number of interactive GPU nodes that can be used for remote client-server visualization.

1. First, install on your computer the same version as the one available on the cluster you will be using; log into Cedar or Graham and start a serial GPU interactive job.

```bash
[name@server ~]$ salloc --time=1:00:00 --ntasks=1 --mem-per-cpu=3600 --gres=gpu:1 --account=def-someprof
```

The job should automatically start on one of the GPU interactive nodes.

2. At the prompt that is now running inside your job, load the ParaView GPU+EGL module, change your display variable so that ParaView does not attempt to use the X11 rendering context, and start the ParaView server.

```bash
[name@server ~]$ module load paraview/5.13.1
[name@server ~]$ unset DISPLAY
[name@server ~]$ pvserver
Waiting for client...
Connection URL: cs://cdr347.int.cedar.computecanada.ca:11111
Accepting connection (s): cdr347.int.cedar.computecanada.ca:11111
```

Wait for the server to be ready to accept client connection.

3. Make a note of the node (in this case `cdr347`) and the port (usually 11111) and in another terminal on your computer (on Mac/Linux; in Windows use a terminal emulator), link the port 11111 on your computer and the same port on the compute node (make sure to use the correct compute node).

```bash
[name@computer $] ssh <username>@cedar.computecanada.ca -L 11111:cdr347:11111
```

4. Start ParaView on your computer, go to `File -> Connect` (or click on the green `Connect` button on the toolbar) and click on `Add Server`. You will need to point ParaView to your local port 11111, so you can do something like `name = cedar`, `server type = Client/Server`, `host = localhost`, `port = 11111`; click on `Configure`, select `Manual` and click on `Save`. Once the remote is added to the configuration, simply select the server from the list and click on `Connect`. The first terminal window that read `Accepting connection` will now read `Client connected`.

5. Open a file in ParaView (it will point you to the remote filesystem) and visualize it as usual.

#### Rendering with NVIDIA's IndeX plugin

NVIDIA IndeX is a 3D volumetric interactive renderer on NVIDIA GPUs enabled as a ParaView server plugin. To use IndeX, connect via client-server to ParaView 5.10 (provided by `paraview-offscreen-gpu/5.10.0`) running inside an interactive GPU job as described above. Then in your client go to `Tools | Manage Plugins` and enable the `pvNVIDIAIndeX` plugin first locally and then remotely. Loading it locally might not be necessary on all platforms, but we saw a bug in several configurations where ParaView server would crash if the local plugin was not selected first. After enabling the plugin, load your dataset and in the Representation drop-down menu select NVIDIA Index.

Our license lets you run NVIDIA IndeX in parallel on multiple GPUs, however parallel speedup is far from perfect. Before doing any production rendering with IndeX on multiple GPUs, please test your parallel scaling and verify that using more than one GPU leads to better performance for your dataset, otherwise use a single GPU.


### Client-server visualization on Niagara

Niagara does not have GPUs, therefore, you are limited to software rendering. With ParaView, you need to explicitly specify one of the mesa flags to tell it to not use OpenGL hardware acceleration, e.g.,

```bash
[name@server ~]$ module load paraview
[name@server ~]$ paraview --mesa-swr
```

or use one of the flags below.

To access interactive resources on Niagara, you will need to start a `debugjob`. Here are the steps:

1. Launch an interactive job (`debugjob`).

```bash
[name@server ~]$ debugjob
```

After getting a compute node, let's say `niaXYZW`, load the ParaView module and start a ParaView server.

```bash
[name@server ~]$ module load paraview
```

The `--mesa-swr-avx2` flag has been reported to offer faster software rendering using the OpenSWR library.

Now, you have to wait a few seconds for the server to be ready to accept client connections.

```
Waiting for client...
Connection URL: cs://niaXYZW.scinet.local:11111
Accepting connection (s): niaXYZW.scinet.local:11111
```

Open a new terminal without closing your `debugjob`, and SSH into Niagara using the following command:

```bash
[name@computer $] ssh YOURusername@niagara.scinet.utoronto.ca -L11111:niaXYZW:11111 -N
```

This will establish a tunnel mapping the port 11111 in your computer (`localhost`) to the port 11111 on the Niagara's compute node, `niaXYZW`, where the ParaView server will be waiting for connections.

Start ParaView on your local computer, go to `File -> Connect` and click on `Add Server`. You will need to point ParaView to your local port `11111`, so you can do something like `name = niagara`, `server type = Client/Server`, `host = localhost`, `port = 11111`; then click on `Configure`, select `Manual` and click on `Save`. Once the remote server is added to the configuration, simply select the server from the list and click on `Connect`. The first terminal window that read `Accepting connection...` will now read `Client connected`.

Open a file in ParaView (it will point you to the remote filesystem) and visualize it as usual.

#### Multiple CPUs

For performing parallel rendering using multiple CPUs, `pvserver` should be run using `srun`, i.e., either submit a job script or request a job using

```bash
[name@server ~]$ salloc --ntasks=N*40 --nodes=N --time=1:00:00
[name@server ~]$ module load paraview
[name@server ~]$ srun pvserver --mesa
```

where you need to replace `N` with the number of nodes and `N*40` with the single number (the product of multiplication).


### Client-server visualization on a cloud

#### Prerequisites

The Cloud Quick Start Guide explains how to launch a new virtual machine (VM). Once you log into the VM, you will need to install some additional packages to be able to compile ParaView or VisIt. For example, on a CentOS VM you can type

```bash
[name@VM $] sudo yum install xauth wget gcc gcc-c++ ncurses-devel python-devel libxcb-devel
[name@VM $] sudo yum install patch imake libxml2-python mesa-libGL mesa-libGL-devel
[name@VM $] sudo yum install mesa-libGLU mesa-libGLU-devel bzip2 bzip2-libs libXt-devel zlib-devel flex byacc
[name@VM $] sudo ln -s /usr/include/GL/glx.h /usr/local/include/GL/glx.h
```

If you have your own private-public SSH key pair (as opposed to the cloud key), you may want to copy the public key to the VM to simplify logins, by issuing the following command on your computer

```bash
[name@computer $] cat ~/.ssh/id_rsa.pub | ssh -i ~/.ssh/cloudwestkey.pem centos@vm.ip.address 'cat >>.ssh/authorized_keys'
```

#### Compiling with OSMesa

Since the VM does not have access to a GPU (most Arbutus VMs don't), we need to compile ParaView with OSMesa support so that it can do offscreen (software) rendering. The default configuration of OSMesa will enable OpenSWR (Intel's software rasterization library to run OpenGL). What you will end up with is a ParaView server that uses OSMesa for offscreen CPU-based rendering without X but with both `llvmpipe` (older and slower) and `SWR` (newer and faster) drivers built. We recommend using SWR.

Back on the VM, compile `cmake`:

```bash
[name@VM $] wget https://cmake.org/files/v3.7/cmake-3.7.0.tar.gz
[name@VM $] # unpack and cd there
[name@VM $] ./bootstrap
[name@VM $] make
[name@VM $] sudo make install
```

Next, compile `llvm`:

```bash
# cd
[name@VM $] wget http://releases.llvm.org/3.9.1/llvm-3.9.1.src.tar.xz
# unpack and cd there
[name@VM $] mkdir -p build && cd build
[name@VM $] cmake \
-DCMAKE_BUILD_TYPE=Release \
-DLLVM_BUILD_LLVM_DYLIB=ON \
-DLLVM_ENABLE_RTTI=ON \
-DLLVM_INSTALL_UTILS=ON \
-DLLVM_TARGETS_TO_BUILD:STRING=X86 \
..
[name@VM $] make
[name@VM $] sudo make install
```

Next, compile Mesa with OSMesa:

```bash
# cd
[name@VM $] wget ftp://ftp.freedesktop.org/pub/mesa/mesa-17.0.0.tar.gz
# unpack and cd there
[name@VM $] ./configure \
--enable-opengl --disable-gles1 --disable-gles2 \
--disable-va --disable-xvmc --disable-vdpau \
--enable-shared-glapi \
--disable-texture-float \
--enable-gallium-llvm --enable-llvm-shared-libs \
--with-gallium-drivers=swrast,swr \
--disable-dri \
--disable-egl --disable-gbm \
--disable-glx \
--disable-osmesa --enable-gallium-osmesa
[name@VM $] make
[name@VM $] sudo make install
```

Next, compile the ParaView server:

```bash
# cd
[name@VM $] wget http://www.paraview.org/files/v5.2/ParaView-v5.2.0.tar.gz
# unpack and cd there
[name@VM $] mkdir -p build && cd build
[name@VM $] cmake \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_PREFIX=/home/centos/paraview \
-DPARAVIEW_USE_MPI=OFF \
-DPARAVIEW_ENABLE_PYTHON=ON \
-DPARAVIEW_BUILD_QT_GUI=OFF \
-DVTK_OPENGL_HAS_OSMESA=ON \
-DVTK_USE_OFFSCREEN=ON \
-DVTK_USE_X=OFF \
..
[name@VM $] make
[name@VM $] make install
```

#### Client-server mode

You are now ready to start ParaView server on the VM with SWR rendering:

```bash
./paraview/bin/pvserver --mesa-swr-avx2
```

Back on your computer, organize an SSH tunnel from the local port 11111 to the VM's port 11111:

```bash
ssh centos@vm.ip.address -L 11111:localhost:11111
```

Finally, start the ParaView client on your computer and connect to `localhost:11111`. If successful, you should be able to open files on the remote VM. During rendering in the console you should see the message `SWR detected AVX2`.


## Remote VNC desktop on Graham VDI nodes

For small interactive visualizations requiring up to 256GB memory and 16 cores, you can use the Graham VDI nodes. Unlike client-server visualizations you will be using VNC remote desktop. Here are the basic steps:

1. Connect to `gra-vdi` as described in the TigerVNC documentation, then open a terminal window and run:

```bash
module load CcEnv
```

2. Show the available paraview module versions:

```bash
module spider paraview
```

3. Show how to load a specific version such as:

```bash
module spider paraview/5.11.0
```

4. Load the required modules and start paraview:

```bash
module load StdEnv/2020  gcc/9.3.0  openmpi/4.0.3
module load paraview/5.11.0
paraview
```

The most recent versions of ParaView require disabling the VirtualGL dynamic linker faker `LD_PRELOAD` when running the `paraview` command

```bash
LD_PRELOAD=${LD_PRELOAD/libdlfaker.so/} paraview
```

## Batch rendering

For large-scale and automated visualization, we strongly recommend switching from interactive client-server to off-screen batch visualization. ParaView supports Python scripting, so you can script your workflow and submit it as a regular, possibly parallel production job on a cluster. If you need any help with this, please contact Technical support.
