# Parallel Debugging with DDT

This is a draft, a work in progress intended for publication as a complete article.  It may not be ready for inclusion in the main wiki and should not be considered factual or authoritative.

## Introduction

The Distributed Debugging Tool (DDT) is a powerful commercial debugger with a graphical user interface (GUI). Its primary use is debugging parallel MPI codes, but it can also be used with serial, threaded (OpenMP/pthreads), and GPU (CUDA; also mixed MPI and CUDA) programs.  Developed by Allinea (U.K.), it's installed on the Graham cluster.  The DDT software page can be found [here](link_to_software_page_here).

DDT supports C, C++, and Fortran 77/90/95/2003. Detailed documentation (the User Guide) is available as a PDF file on Graham, in `${EBROOTALLINEA}/doc` (load the corresponding module first; see below).


## Preparing your program

Your code must be compiled with the `-g` switch, which tells the compiler to generate the symbolic information required by any debugger.  Normally, all optimizations must be turned off. For example:

```bash
f77 -g -O0 -o program program.f
mpicc -g -O0 -o code code.c
```

For CUDA code, compile the CUDA part (*.cu files) using `nvcc`:

```bash
nvcc -g -G -c cuda_code.cu
```

Then link with the non-CUDA part of the code, using `cc` for serial code:

```bash
cc -g main.c cuda_code.o -lcudart
```

or `mpicc` for mixed MPI/CUDA code:

```bash
mpicc -g main.c cuda_code.o -lcudart
```

## Launching DDT

DDT is a GUI application, so you must set up the environment to run X-windows (graphical) applications on Graham:

* **Microsoft Windows:** Use the free Mobaxterm app.
* **Linux/Unix:** No installation needed.
* **Mac:** Install the free XQuartz app.

In all cases, add `-Y` to all your `ssh` commands for X11 tunneling.

DDT is an interactive GUI application. Before using it, allocate a compute node (or nodes) with `salloc`:

```bash
salloc -A account_name --x11 --time=0-3:00 --mem-per-cpu=4G --ntasks=4  # for CPU codes
salloc -A account_name --x11 --time=0-3:00 --mem-per-cpu=4G --ntasks=1 --gres=gpu:1  # for GPU codes
```

Once the resource is available, load the corresponding DDT module:

```bash
module load allinea-cpu  # or
module load allinea-gpu
```

For MPI codes, execute:

```bash
export OMPI_MCA_pml=ob1
```

Otherwise, the Message Queue display feature will not work. To debug a code, type:

```bash
ddt program [optional arguments]
```

After a few seconds, the DDT window will appear.


## User Interface

DDT uses a tabbed-document interface. Each component is a dockable window that can be dragged around by a handle. Components can be double-clicked or dragged outside of DDT to form a new window. Some user-modified parameters and windows are saved by right-clicking and selecting a save option (Groups; Evaluations; Breakpoints, etc.).

DDT can load and save all these options concurrently to minimize inconvenience when restarting sessions. Saving the session stores process groups, the contents of the Evaluate window, and more. When DDT starts a session, source code is automatically found from the information compiled into the executable.

The "Find" and "Find In Files" dialogs are in the "Search" menu. "Find" searches the currently visible source file. "Find In Files" searches all source and header files associated with your program and lists matches in a result box. Click a match to display the file in the main Source Code window and highlight the matching line.

DDT has a "Goto line" function (in the "Search" menu or Ctrl-G).


## Controlling Program Execution

### Process Control and Process Groups

Ability to group many processes together with one-click access to the whole group. Three predefined groups: All, Root, Workers (newest DDT versions only have one group - All). Groups can be created, deleted, or modified at any time.

### Starting, Stopping, and Restarting a Program

Use the Session Control dialog.

### Stepping Through a Program

Play/Continue, Pause, Step Into, Step Over, Step Out.

### Setting Breakpoints

All breakpoints are listed under the breakpoints tab. You can suspend, jump to, delete, load, or save breakpoints. Breakpoints can be made conditional.

### Synchronizing Processes in a Group

"Run to here" command (right mouse click). Can work with individual processes or threads (by changing "Focus").

### Setting a Watch (for the current process)

### Working with Stacks

Moving (Down/Up/Bottom Stack Frame). Aligning of stacks for parallel codes (Ctrl-A).

### Viewing Stacks in Parallel

Stacks tab. Shows a tree of functions merged from every process in the group. Click any branch to see its location in the Source Code viewer. Hover the mouse to see the list of process ranks at this location in the code. Can automatically gather processes at a function together in a new group.

### Browsing Source Code

Highlights the current source line. Different color coding for synchronous and asynchronous state in the group.

### Simultaneously Viewing Multiple Files

Right-click to split the source window into two or more, each with its own set of tabs.

### Signal Handling

Will stop on the following signals:

* SIGSEGV (segmentation fault)
* SIGFPE (Floating Point Exception)
* SIGPIPE (Broken Pipe)
* SIGILL (Illegal Instruction)

To enable FPE handling, extra compiler options are required:  Intel: `-fp0`


## Variables and Data

### Current Line tab

Viewing all variables for the current line(s) (click and drag for multiple lines).

### Locals tab

Shows all local variables for the process.

### Evaluate window

Can be used to view values for arbitrary expressions and global variables. Support for Fortran modules (Fortran Modules tab in the Project Navigator window).

### Changing Data Values

Right-click and select "Edit value" (in Evaluate window).

### Examining Pointers

Drag a pointer into the Evaluate window. Can be viewed as Vector, Reference, or Dereference (right-click to choose).

### Multi-Dimensional Arrays

Can be viewed in the Variable View. Multi-Dimensional Array viewer – visualization of a 2-D slice of an array using OpenGL.

### Cross-Process Comparison

Right-click on a variable name, or from the View menu: Cross-Process/Thread Comparison (type in any valid expression). Three modes – Compare, Statistics, and Visualize.


## Message Queues

**Attention:** This feature will not work on Graham unless this command is executed first, before running `ddt`:

```bash
export OMPI_MCA_pml=ob1
```

View -> Message Queues (older versions), or Tools -> Message Queues in newer versions, in the control panel. Produces both a graphical view and a table for all active communications. Helps debug MPI problems such as deadlocks.


## Memory Debugging

Can intercept memory allocation and deallocation calls and perform complex heap- and bounds-checking. Off by default, it can be turned on before starting a debugging session (in Advanced settings). Different levels of memory debugging – from minimal to high. On error, stops with a message.

### Check Validity

In the Evaluate window, right-click to "Check pointer is valid". Useful for checking function arguments.

### Detecting Memory Leaks

"View->Current Memory Usage" window ("Tools->Current Memory Usage" in newer versions). Shows current memory usage across processes in the group. Click on a color bar to get allocation details. For more details, choose "Table View".

### Memory Statistics

Menu option "View->Overall Memory Stats" ("Tools->Overall Memory Stats" in newer versions).


## Debugging OpenMP Programs

The current version of DDT has almost the same OpenMP functionality as for MPI: single-click access to threads, viewing stacks in parallel, setting thread-specific breakpoints, comparing expressions across threads, etc.


## Debugging GPU Programs

Compilation instructions were given earlier. Can be mixed MPI/CUDA code, but can only use one CPU and one GPU per node, up to 8 nodes.
