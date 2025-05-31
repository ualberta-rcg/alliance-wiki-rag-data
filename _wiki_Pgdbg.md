# Pgdbg

## Description

PGDBG is a powerful and simple tool for debugging both MPI-parallel and OpenMP thread-parallel Linux applications. It is included in the PGI compiler package and configured for OpenMP thread-parallel debugging.

For most C, C++, or Fortran 77 codes, one can use a regular GNU debugger such as GDB. However, GDB does not handle Fortran 90/95 programs very well.  The Portland Group developed `pgdbg`, a debugger more suited for such codes.  `pgdbg` is provided in two modes: a graphical mode (with enabled X11 forwarding) or a text mode.


## Quickstart Guide

Using PGDBG usually consists of two steps:

1. **Compilation:** Compile the code with debugging enabled.
2. **Execution and debugging:** Execute the code and analyze the results.

The actual debugging can be accomplished in either command-line mode or graphical mode.


### Environment Modules

Before starting profiling with PGDBG, load the appropriate module.

PGDBG is part of the PGI compiler package; run `module avail pgi` to see available versions with your loaded compiler, MPI, and CUDA modules. For a comprehensive list of PGI modules, run `module -r spider '.*pgi.*'`.

As of December 2018, these were available:

*   `pgi/13.10`
*   `pgi/17.3`

Use `module load pgi/version` to select a version; for example, to load PGI compiler version 17.3, use:

```bash
module load pgi/17.3
```

### Compiling Your Code

To debug with `pgdbg`, compile your code with debugging information enabled.  Add the `-g` debugging flag:

```bash
pgcc -g program.c -o program
```

### Command-Line Mode

Once your code is compiled, run PGDBG. The debugger's default interface is graphical; however, if you don't want to run in GUI mode or lack X11 forwarding, run `pgdbg` in text mode by adding the `-text` option:

```bash
pgdbg -text program arg1 arg2
```

Once invoked in command-line mode, you'll access a prompt:

```
PGDBG 17.3-0 x86-64 (Cluster, 256 Process)
PGI Compilers and Tools Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
Loaded: /home/user/program
pgdbg>
```

Before debugging, execute `run` at the prompt:

```bash
pgdbg> run
```

PGDBG automatically attaches to new threads as they're created.  PGDBG indicates when a new thread is created.

During a debug session, PGDBG operates within a single thread (the current thread).  Select the current thread using the `thread` command. The `threads` command lists all threads:

```bash
pgdbg> threads
0  ID  PID  STATE  SIGNAL  LOCATION
3  18399 Stopped SIGTRAP main line: 31 in "omp.c" address: 0x80490ab =>
2  18398 Stopped SIGTRAP main line: 32 in "omp.c" address: 0x80490cf
1  18397 Stopped SIGTRAP main line: 31 in "omp.c" address: 0x80490ab
0  18395 Stopped SIGTRAP f line:  5 in "omp.c" address: 0x8048fa0
```

For example, switch the context to thread ID 2 using the `thread` command:

```bash
pgdbg> thread 3
pgdbg> threads
0  ID  PID  STATE  SIGNAL  LOCATION
=> 3  18399 Stopped SIGTRAP main line: 31 in "omp.c" address: 0x80490ab
2  18398 Stopped SIGTRAP main line: 32 in "omp.c" address: 0x80490cf
1  18397 Stopped SIGTRAP main line: 31 in "omp.c" address: 0x80490ab
0  18395 Stopped SIGTRAP f line:  5 in "omp.c" address: 0x8048fa0
```

### Graphical Mode

This is the default PGDBG interface. If X11 forwarding is set, PGDBG starts in graphical mode in a pop-up window.

(Image of PGDGB in graphical mode would be inserted here)

The GUI is divided into several areas:

*   Menu bar
*   Main toolbar
*   Source window
*   Program I/O window
*   Debug information tabs

#### Menu Bar

The main menu bar contains menus: File, Edit, View, Connections, Debug, and Help.  Navigate using the mouse or keyboard shortcuts.

#### Main Toolbar

The main toolbar contains buttons and four drop-down lists. The first drop-down displays the current process (or thread).  The label changes depending on whether processes or threads are shown. Use this to specify the current process or thread when multiple are available.

(Image of Drop-Down Lists on Toolbar would be inserted here)

The second drop-down ("Apply") determines the processes and threads to which action commands apply. The third ("Display") determines the processes and threads for data display commands. The fourth ("File") displays the source file containing the current target location.

#### Source Window

The source window (and all debug information tabs) is dockable (detachable from the main window by double-clicking the tab). It shows the source code for the current session.

(Image of the source window would be inserted here)

#### Program I/O Window

Program output is displayed in the Program I/O tab's central window. Program input is entered into this tab's Input field.

#### Debug Information Tab

Debug information tabs occupy the lower half of the GUI. Each tab provides a specific function or view of debug information.


## References

*   PGI Debugger User's Guide
*   PGI webpage


**(Retrieved from "https://docs.alliancecan.ca/mediawiki/index.php?title=Pgdbg&oldid=77628")**
