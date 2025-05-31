# Pgdbg

This page is a translated version of the page Pgdbg and the translation is 100% complete.

Other languages: English, français

## Description

PGDBG is a simple yet powerful tool for debugging MPI and OpenMP parallel applications under Linux.  The tool is part of the PGI compiler package and is configured for OpenMP with parallel threads. It can be used in graphical mode with X11 forwarding or in command-line mode.

A GNU debugger like GDB will suffice for most C, C++, or Fortran77 programs. However, GDB does not work very well with Fortran 90/95 programs; this is why the Portland Group developed `pgdbg`.


## Getting Started

Working with PFDBG generally involves two steps:

1. **Compilation:** The code is compiled (with the `-g` option to obtain debugging symbols).
2. **Execution and Debugging:** The code is executed and the results are analyzed. Debugging can be done in command-line mode or graphical mode.


### Environment Modules

First, load the module for the PGI package. To see the available versions for the compiler, MPI, and CUDA modules you have loaded, run `module avail pgi`. To see a complete list of available PGI modules, run `module -r spider '.*pgi.*'`. As of December 2018, the available versions are `pgi/13.10` and `pgi/17.3`. To load a module, run `module load pgi/version`; for example, for version 17.3, the command is:

```bash
[name@server ~]$ module load pgi/17.3
```

### Compilation

Before debugging, the code must first be compiled by adding the `-g` flag to obtain information useful for debugging:

```bash
[name@server ~]$ pgcc -g program.c -o program
```

### Command-Line Execution and Debugging

Once the code is compiled with the appropriate options, launch PGDBG to perform the analysis. By default, the display is done through the graphical interface. However, if you do not want to use this interface or do not have X11 forwarding, you can work in command-line mode by adding the `text` option when launching PDGDB:

```bash
[name@server ~]$ pgdbg -text program
```

This will output:

```
PGDBG 17.3-0 x86-64 (Cluster, 256 Process)
PGI Compilers and Tools Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
Loaded: /home/user/program
pgdbg>
```

At the prompt, run the `run` command:

```bash
[name@server ~]$ pgdbg> run
```

During program execution, PGDBG automatically attaches to the threads and describes each one as they are created. During debugging, PGDBG works on only one thread at a time, the current thread. The `thread` command is used to select the current thread, and the `threads` command lists the threads currently used by an active program.

```bash
[name@server ~]$ pgdbg> threads
0  ID   PID  STATE  SIGNAL LOCATION
3  18399 Stopped SIGTRAP main line: 31 in "omp.c" address: 0x80490ab =>
2  18398 Stopped SIGTRAP main line: 32 in "omp.c" address: 0x80490cf
1  18397 Stopped SIGTRAP main line: 31 in "omp.c" address: 0x80490ab
0  18395 Stopped SIGTRAP f   line:  5 in "omp.c" address: 0x8048fa0
```

For example, to select ID 2 as the current thread, the `thread` command would be:

```bash
[name@server ~]$ pgdbg> thread 3
pgdbg> threads
0  ID   PID  STATE  SIGNAL LOCATION
=> 3  18399 Stopped SIGTRAP main line: 31 in "omp.c" address: 0x80490ab
2  18398 Stopped SIGTRAP main line: 32 in "omp.c" address: 0x80490cf
1  18397 Stopped SIGTRAP main line: 31 in "omp.c" address: 0x80490ab
0  18395 Stopped SIGTRAP f   line:  5 in "omp.c" address: 0x8048fa0
```

### Graphical Execution and Debugging

The graphical interface is used by default. If you have configured X11 forwarding, PGDBG starts in graphical mode in a new window.

**(Graphical Interface Image would be inserted here)**

The graphical interface elements are:

* Menu bar
* Toolbar
* Source code pane
* Input/Output (I/O) pane
* Debug pane


#### Menu Bar

The main menu bar displays *File*, *Edit*, *View*, *Connections*, *Debug*, and *Help*. Navigation is possible with the mouse or keyboard shortcuts.

#### Main Toolbar

The main toolbar contains several buttons and four drop-down lists. The first list, *Current Process*, shows the current process, in other words, the current thread. The label changes depending on whether the process or thread is described. When multiple threads are available, this list is used to select the process or thread that should be current.

**(Toolbar Dropdown Lists Image would be inserted here)**

The second list, *Apply*, determines the group of processes and threads to which action commands apply.

The third list, *Display*, determines the group of processes and threads to which data display commands apply.

The fourth list, *File*, displays the source file containing the current target.

#### Source Code and Debugging Tools Panes

The source code pane shows the code for the current session. This pane and the tabs in the debug pane are dockable elements; double-clicking on them allows you to detach them from the main window.

**(Source Code Pane Image would be inserted here)**

The panel showing the source code offers tools that allow you to see how the code is executed.

#### Program Input/Output Pane

The results produced by the program are displayed in this pane. Use the *Input* field to enter input to the program.

#### Debug Pane

Located at the bottom of the window, this pane has tabs that serve different debugging and information visualization functions.


## References

* PGI Debugger User's Guide
* PGI website


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Pgdbg/fr&oldid=77725](https://docs.alliancecan.ca/mediawiki/index.php?title=Pgdbg/fr&oldid=77725)"
