# ParaView

This page is a translated version of the page ParaView and the translation is 100% complete.

Other languages: English français

## Visualisation client-serveur

**NOTE 1:** An important preference option is the threshold defined with `Render View -> Remote/Parallel Rendering Options -> Remote Render Threshold`. With the default value (20MB) or a similar value, small rendering tasks will be performed with your computer's GPU; rotation using the mouse will be fast, but any somewhat intensive task (under 20MB) will be directed to your computer and depending on the connection, visualization could be slow. With 0MB, all rendering will be done remotely; the cluster resources will do all the work, which is a good thing for intensive processing, but less desirable for the interactive aspect. Experiment with different values to find an acceptable threshold.

**NOTE 2:** The same major version must be installed on the local client and the remote host computer; otherwise, some incompatibilities may prevent the client-server connection. For example, to use version 5.10.0 of the ParaView server on our clusters, you need client version 5.10.x on your computer.

Select the appropriate tab: Cedar, Graham, Béluga, Niagara, Cloud

### Visualisation client-serveur avec Cedar, Graham, Béluga et Narval

On Cedar, Graham, Béluga and Narval, client-server mode rendering can be done with a CPU (by software) and a GPU (by hardware). Since GPU rendering is somewhat complicated, we recommend using only CPUs and allocating as many cores as needed.

The easiest way to estimate the number of cores is to divide by ~3.5GB/core the amount of memory you think you need. For example, 40GB of data (loaded in bulk in a single step) would require at least 12 cores just to contain the data.

Since software rendering is very CPU intensive, we recommend not exceeding 4GB/core.  You also need to allocate some memory for filters and data processing (for example, converting a structured dataset to an unstructured dataset would require about three times more memory); if your tasks allow it, you could start with 32 or 64 cores. If the ParaView server stops during execution, you will need to increase the number of cores.


#### Avec CPU

ParaView can also be used on a cluster's CPU. In some cases, libraries for modern CPUs, such as OSPray or OpenSWR, offer performance comparable to that obtained with a GPU. Also, since the ParaView server uses distributed memory MPI, very large datasets can be processed in parallel with multiple CPU cores on a single node or on multiple distributed nodes.

1. On your workstation, install the same version of ParaView as the one on the cluster you will be using; connect to the cluster and launch a serial interactive task with a CPU.

```bash
[name@server ~]$ salloc --time=1:00:0 --ntasks=1 --mem-per-cpu=3600 --account=def-someprof
```

The task should start automatically on one of the interactive CPU nodes.

2. At the prompt in your task, load the ParaView off-screen rendering module; start the server.

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

Wait for the server to be ready to accept the client connection.

3. Note the node (here cdr774) and the port (usually 11111); in another terminal on your Mac/Linux workstation (on Windows, use a terminal emulator), link port 11111 to your workstation and the same port to the compute node (make sure to use the correct compute node).

```bash
[name@computer $] ssh <username>@cedar.computecanada.ca -L 11111:cdr774:11111
```

4. On your workstation, start ParaView; go to `File -> Connect` (or click the green `Connect` button in the toolbar); click `Add Server`. Point ParaView to your local port 11111 to have settings similar to `name = cedar, server type = Client/Server, host = localhost, port = 11111`; click `Configure`; click `Manual` then `Save`.

Once the connection is added to the configuration, select the server from the displayed list and click `Connect`. In the first terminal window, the message `Accepting connection ...` now reads `Client connected`.

5. Open a ParaView file (which directed you to the remote file system) to visualize it.

**NOTE:** An important preference option is the threshold defined with `Render View -> Remote/Parallel Rendering Options -> Remote Render Threshold`. With the default value (20MB) or a similar value, small rendering tasks will be performed with your computer's GPU; rotation using the mouse will be fast, but any somewhat intensive task (under 20MB) will be directed to your computer and depending on the connection, visualization could be slow. With 0MB, all rendering will be done remotely; the cluster resources will do all the work, which is a good thing for intensive processing, but less desirable for the interactive aspect. Experiment with different values to find an acceptable threshold.

For parallel rendering with multiple CPUs, launch a parallel task remembering to specify the limit for the maximum real-time execution time.

```bash
[name@server ~]$ salloc --time=0:30:0 --ntasks=8 --mem-per-cpu=3600 --account=def-someprof
```

Start the ParaView server with `srun`.

```bash
[name@server ~]$ module load paraview-offscreen/5.13.1
[name@server ~]$ srun pvserver --force-offscreen-rendering
```

To verify that the rendering is performed in parallel, use the `Process Id Scalars` filter and apply the color with `process id`.


#### Avec GPU

Cedar and Graham offer several interactive nodes with GPUs to work in client-server mode.

1. On your workstation, install the same version as the one on the cluster you will be using; connect to the cluster and launch a serial interactive task with a GPU.

```bash
[name@server ~]$ salloc --time=1:00:0 --ntasks=1 --mem-per-cpu=3600 --gres=gpu:1 --account=def-someprof
```

The task should start automatically on one of the interactive GPU nodes.

2. At the prompt in your task, load the GPU+EGL module; modify the display variable to prevent ParaView from using the X11 rendering context; start the ParaView server.

```bash
[name@server ~]$ module load paraview/5.13.1
[name@server ~]$ unset DISPLAY
[name@server ~]$ pvserver
Waiting for client...
Connection URL: cs://cdr347.int.cedar.computecanada.ca:11111
Accepting connection (s): cdr347.int.cedar.computecanada.ca:11111
```

Wait for the server to be ready to accept the client connection.

3. Note the node (here cdr347) and the port (usually 11111); in another terminal on your Mac/Linux workstation (on Windows, use a terminal emulator), link port 11111 to your workstation and the same port to the compute node (make sure to use the correct compute node).

```bash
[name@computer $] ssh <username>@cedar.computecanada.ca -L 11111:cdr347:11111
```

4. On your computer, start ParaView; go to `File -> Connect` (or click the green `Connect` button in the toolbar); click `Add Server`. Point ParaView to your local port 11111 to have settings similar to this: `name = cedar, server type = Client/Server, host = localhost, port = 11111`; click `Configure`, select `Manual` and click `Save`.

Once the connection is added to the configuration, select the server from the displayed list and click `Connect`. In the first terminal window, the message `Accepting connection ...` now reads `Client connected`.

5. Open a ParaView file (which directed you to the remote file system) to visualize it.


#### Utiliser NVDIA IndeX pour produire des rendus

NVIDIA IndeX is an interactive 3D volumetric rendering engine that installs as a plugin with ParaView. To use it, you must connect in client-server mode to ParaView 5.10 (provided by `paraview-offscreen-gpu/5.10.0`) which is running in an interactive task with GPU, as described above. In your client, go to `Tools -> Manage Plugins` and first activate `pvNVIDIAIndeX` locally and then remotely. It may not be necessary to activate the plugin locally on all platforms, but in several configurations, a bug causes ParaView to close abnormally if the local plugin is not activated first. Once the plugin is activated, load your dataset, then select NVIDIA IndeX from the `Representation` drop-down menu.

With our license, you can use NVIDIA IndeX in parallel on multiple GPUs, but the acceleration leaves much to be desired. Before going into production with multiple GPUs, we recommend that you test your parallel scalability and verify that using multiple GPUs provides better performance; if not, you should use a single GPU.


### Visualisation client-serveur avec Niagara

Since Niagara does not have a GPU, it is necessary to limit oneself to software rendering. You must explicitly use one of the mesa flags so that ParaView does not use hardware OpenGL acceleration.

```bash
[name@server ~]$ module load paraview
[name@server ~]$ paraview --mesa-swr
```

or use one of the flags below.

To access the interactive resources of Niagara, launch a `debugjob` task as follows: Launch an interactive task (`debugjob`).

```bash
[name@server ~]$ debugjob
```

Once connected to the compute node (e.g., `niaXYZW`), load the ParaView module and start a ParaView server.

```bash
[name@server ~]$ module load paraview
```

With the OpenSWR library, rendering is sometimes faster with the `--mesa-swr-avx2` flag.

Wait a few seconds for the server to be ready to accept client connections.

```
Waiting for client...
Connection URL: cs://niaXYZW.scinet.local:11111
Accepting connection (s): niaXYZW.scinet.local:11111
```

Open a new terminal without closing `debugjob` and connect via SSH.

```bash
[name@laptop $] ssh YOURusername@niagara.scinet.utoronto.ca -L11111:niaXYZW:11111 -N
```

This sets up a tunnel that associates port 11111 of your computer (`localhost`) to port 11111 of the compute node `niaXYZW` where the ParaView server is waiting for connections.

Start ParaView on your computer. Select `File -> Connect` and click `Add Server`.

To direct ParaView to your local port `11111`, you can do the following:

```
name = niagara
server type = Client/Server
host = localhost
port = 11111
```

Then click `Configure`, select `Manual` and click `Save`.

Once the remote server is part of the configuration, select it from the list and click `Connect`.

The contents of the terminal window change from `Accepting connection...` to `Client connected`.

Open a file in ParaView (you will be directed to the remote file system) and visualize the data as usual.


#### CPU multiples

To perform parallel rendering with multiple CPUs, `pvserver` should be run with `mpiexec`, i.e., you submit a job script or request a job with

```bash
[name@server ~]$ salloc --ntasks=N*40 --nodes=N --time=1:00:00
[name@server ~]$ module load paraview
[name@server ~]$ srun pvserver --mesa
```

where `N` is replaced by the number of nodes and `N*40` is replaced by the simple number (the product of the multiplication).


### Visualisation client-serveur sur le cloud

#### Prérequis

The page [Cloud: Getting Started](link-to-cloud-getting-started-page) describes the creation of an instance. Once connected to the instance, you will need to install some packages to be able to compile ParaView and VisIt; for example, on a CentOS instance,

```bash
[name@VM $] sudo yum install xauth wget gcc gcc-c++ ncurses-devel python-devel libxcb-devel
[name@VM $] sudo yum install patch imake libxml2-python mesa-libGL mesa-libGL-devel
[name@VM $] sudo yum install mesa-libGLU mesa-libGLU-devel bzip2 bzip2-libs libXt-devel zlib-devel flex byacc
[name@VM $] sudo ln -s /usr/include/GL/glx.h /usr/local/include/GL/glx.h
```

If you have your own SSH key pair (and not the key generated by OpenStack for the cloud), you could copy the public key into the instance to simplify the connection; to do this, run the following command on your computer:

```bash
[name@laptop $] cat ~/.ssh/id_rsa.pub | ssh -i ~/.ssh/cloudwestkey.pem centos@vm.ip.address 'cat >>.ssh/authorized_keys'
```

#### Compiler avec OSMesa

Since the instances do not have access to a GPU, and this is the case for most instances in Artbutus, it is necessary to compile ParaView with OSMesa to obtain off-screen rendering. The default OSMesa configuration enables OpenSWR, Intel's software rasterization library that allows OpenGL to operate. The result will be a ParaView server that uses OSMesa to build an off-screen X-less render with a processor, but with the newer and faster `llvmpipe` and `SWR` drivers. We recommend SWR.

Back in the instance, compile `cmake`:

```bash
[name@VM $] wget https://cmake.org/files/v3.7/cmake-3.7.0.tar.gz
[name@VM $] unpack and cd there
[name@VM $] ./bootstrap
[name@VM $] make
[name@VM $] sudo make install
```

Then, compile `llvm`.

```bash
cd
wget http://releases.llvm.org/3.9.1/llvm-3.9.1.src.tar.xz
unpack and cd there
mkdir -p build && cd build
cmake \
-DCMAKE_BUILD_TYPE=Release \
-DLLVM_BUILD_LLVM_DYLIB=ON \
-DLLVM_ENABLE_RTTI=ON \
-DLLVM_INSTALL_UTILS=ON \
-DLLVM_TARGETS_TO_BUILD:STRING=X86 \
..
make
sudo make install
```

Then, compile Mesa with OSMesa.

```bash
cd
wget ftp://ftp.freedesktop.org/pub/mesa/mesa-17.0.0.tar.gz
unpack and cd there
./configure \
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
make
sudo make install
```

Then, compile the ParaView server.

```bash
cd
wget http://www.paraview.org/files/v5.2/ParaView-v5.2.0.tar.gz
unpack and cd there
mkdir -p build && cd build
cmake \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_PREFIX=/home/centos/paraview \
-DPARAVIEW_USE_MPI=OFF \
-DPARAVIEW_ENABLE_PYTHON=ON \
-DPARAVIEW_BUILD_QT_GUI=OFF \
-DVTK_OPENGL_HAS_OSMESA=ON \
-DVTK_USE_OFFSCREEN=ON \
-DVTK_USE_X=OFF \
..
make
make install
```

#### Mode client-serveur

You can now start the ParaView server on the instance for SWR rendering.

```bash
./paraview/bin/pvserver --mesa-swr-avx2
```

From your computer, organize an SSH tunnel from local port 11111 to port 11111 of the instance.

```bash
ssh centos@vm.ip.address -L 11111:localhost:11111
```

Now start the ParaView client on your computer and connect to `localhost:11111`; you should be able to open the instance files remotely. During rendering, the console will display `SWR detected AVX2`.


## Bureau distant VNC sur les nœuds VDI de Graham

For small interactive visualizations that require up to 250GB and 16 cores, you can use the Graham VDI nodes. Unlike client-server visualizations, you will use a remote VNC desktop by following these steps:

1. You must use a VNC client (Tiger VNC preferably) to connect to the VDI node.
2. Open a terminal window and run the following commands:
3. Check how to load a particular version, for example `module spider paraview/5.11.0`
4. Load the required modules and launch Paraview.

```bash
module load StdEnv/2020  gcc/9.3.0  openmpi/4.0.3
module load paraview/5.11.0
paraview
```

With the latest version of ParaView, the command `paraview` must first disable the fake dynamic linker originally used for VirtualGL.

```bash
LD_PRELOAD=${LD_PRELOAD/libdlfaker.so/} paraview
```

## Rendus non interactifs

For intensive and automatic visualizations, we recommend using off-screen non-interactive tasks. Since it is possible to work with Python scripts, you can prepare your work and submit the script as a possibly parallel task. For assistance, contact [technical support](link-to-technical-support).


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=ParaView/fr&oldid=165916](https://docs.alliancecan.ca/mediawiki/index.php?title=ParaView/fr&oldid=165916)"
