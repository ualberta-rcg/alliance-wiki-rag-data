# make

`make` is a software that automatically builds libraries or often executable files from basic elements such as source code.  The `make` command interprets and executes the instructions in the `makefile`. Unlike a simple script, `make` executes commands only if necessary. The goal is to achieve a result (compiled or installed software, created documentation, etc.) without necessarily repeating all the steps.

The `makefile` contains information about dependencies, among other things. For example, since the program's executable depends on the source files, if some of these files have changed, a reassembly of the program is necessary. Similarly, object files depending on their associated source files, if a source file has been modified, the latter must be recompiled to recreate the new object file. All these dependencies must be included in the `makefile`. Thus, it is no longer necessary to recompile all the source files with each modification; the `make` command takes care of recompiling and reassembling only what is necessary.


## Usage Examples

The main argument of the `make` command is generally the target. This is the component that `make` must build. The available targets depend on the contents of the `makefile`, but some targets are very common, for example `all`, `test`, `check`, `clean`, and `install`, which are often used. In the following example of `make`, no target is specified:

```bash
[name@server ~]$ make
```

The typical behavior is to build everything, which is equivalent to:

```bash
[name@server ~]$ make all
```

The `test` or `check` targets are generally used to run tests to validate that the compiled application or library works correctly. In general, these targets depend on the `all` target. You can thus verify the compilation via the command:

```bash
[name@server ~]$ make all && make check
```

or

```bash
[name@server ~]$ make all && make test
```

The `clean` target erases all previously compiled binary files in order to resume compilation from scratch. There is sometimes also the `distclean` target which erases not only the files created by `make`, but also the files created during the configuration operation by `configure` or `cmake`. Thus, to clean the compilation directory, you can generally execute:

```bash
[name@server ~]$ make clean
```

and sometimes

```bash
[name@server ~]$ make distclean
```

The `install` target normally proceeds with the installation of the compiled application or library. The installation location depends on the `makefile`, but can often be modified via an additional parameter `prefix` like this:

```bash
[name@server ~]$ make install prefix=$HOME/PROGRAM
```

These targets `all`, `test`, `check`, `clean`, `distclean`, and `install` are however only conventions and the author of a `makefile` could very well choose another convention. For more information on typical targets, including those supported by all GNU applications, see [this page](link_to_gnu_make_targets). The options for configuring installation directories and others are listed [here](link_to_gnu_make_options).


## Makefile Example

The following example, of general use, includes many explanations and comments. For an in-depth guide on creating `makefile` files, visit the [GNU Make website](link_to_gnu_make_website).

**File: Makefile**

```makefile
# Makefile for easily updating the compilation of a program (.out)
# --------
#
# by Alain Veilleux, August 4, 1993
#     Last revision, March 30, 1998
#
# PURPOSE AND OPERATION OF THIS SCRIPT:
#    Script in makefile form allowing to update a program
#    including several separate routines on the disk. This script is not  #    executed by itself, but is rather read and interpreted by the command
#    make. When called, the make command checks the dates of the
#    different files composing your compiled program. Only the
#    routines that have been modified since the compilation of the final program
#    will be recompiled in object form (files ending in .o). The
#    .o files will then be linked together to reform an updated version of the #    final program.
#
# TO ADAPT THIS SCRIPT TO YOUR PROGRAM:
#    Modify the contents of the variables in the section below. Comments
#    will guide you in this direction.
#
# USING make ON THE UNIX COMMAND LINE:
#    1- Type "make" to update the entire program.
#    2- Type "make NomRoutine" to update only the
#          NomRoutine routine.
#
#====================  Definition of variables  ====================
# Note: variables are sometimes called "macros" in Makefile files
# Name of the compiler to use (FORTRAN, C or other)
NomCompilateur = xlf
# Compilation options: below, you will find the options normally
#                          used to compile in FORTRAN. You can
#                          assign values other than those suggested to the
#                          OptionsDeCompilation variable.
#OptionsDeCompilation= -O3
# Remove the # character below to enable compilation in debug mode
#OptionsDeCompilation= -g
# Remove the # character below to use gprof which indicates the calculation time of
#    each subroutine
#OptionsDeCompilation= -O3 -pg
# List of routines to compile: the object versions are named here
# Place a \ at the end of each line if you want to continue the list
#    of routines on the next line.
FichiersObjets = trnb3-1.part.o mac4251.o inith.o dsite.o initv.o main.o \
entree.o gcals.o defvar1.o defvar2.o magst.o mesure.o
# Name of the executable program finally produced
ProgrammeOut = trnb3-1.out
#=====  End of variable definition  =====
#===============  There is nothing to change from here  ===============
# Defines a rule: how to build an object file (ending in .o)
#                   from a source file (ending in .f)
# note: the $< symbols will be replaced by the name of the program to compile
# Compilation of programs in Fortran language
.f.o:
	$(NomCompilateur) $(OptionsDeCompilation) -c $<
# Defines a rule: how to build an object file (ending in .o)
#                   from a source file (ending in .c)
# note: the $< symbols will be replaced by the name of the program to compile
# Compilation of programs in C language
.c.o:
	$(NomCompilateur) $(OptionsDeCompilation) -c $<
# Defines a rule: how to build an object file (ending in .o)
#                   from a source file (ending in .C)
# note: the $< symbols will be replaced by the name of the program to compile
# Compilation of programs in C language
.C.o:
	$(NomCompilateur) $(OptionsDeCompilation) -c $<
# Dependency of the executable program on the object files (.o) composing it.
# The dependency of the object files on the source files (.f and .c) is
#    implied by the rules defined above.
$(ProgrammeOut): $(FichiersObjets)
	$(NomCompilateur) $(OptionsDeCompilation) -o $(ProgrammeOut) \
	$(FichiersObjets)
```

Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Make/fr&oldid=158118](https://docs.alliancecan.ca/mediawiki/index.php?title=Make/fr&oldid=158118)"


**Note:**  Replace `link_to_gnu_make_targets`, `link_to_gnu_make_options`, and `link_to_gnu_make_website` with the actual URLs.
