# Autotools

This page is a translated version of the page Autotools and the translation is 100% complete.

Other languages: English, français


## Description

The Autotools suite (also called GNU build system) includes the `autoconf` tool. This tool automates the creation of `Makefile` files for the `Make` utility for different systems and (possibly) different compilers.

When a program is built using `autoconf`, the first step is to call the `configure` script:

```bash
[name@server ~]$ ./configure
```

`Autoconf` checks that the necessary compiler and software versions are installed on the computer and generates the appropriate `Makefile`.

Then, `Make` is called in the usual way:

```bash
[name@server ~]$ make
```

Files are installed in the correct locations by `make install`.

To reserve exclusive access to the software, you normally need to specify where your software will be installed, which is usually done like this:

```bash
[name@server ~]$ mkdir $HOME/LOGICIEL
```

In some cases, you need to use the `--prefix` option instead of `make`; refer to the documentation of the software you want to install. To indicate the paths to our new software to the system, you need to create a module.

A basic compilation of a program using Autoconf can be as simple as:

```bash
[name@server ~]$ ./configure && make && make install --prefix=$HOME/LOGICIEL
```


## Options for `configure` scripts

There are many options whose use varies depending on the project. To obtain the detailed list, run:

```bash
[name@server ~]$ ./configure --help
```

Here are the most common options.


### Installation Directory

An always available option is `--prefix`. This allows you to define in which directory `make install` will install the application or library. For example, to install an application in the `programmes` subdirectory of your home directory, you can use:

```bash
[name@server ~]$ ./configure --prefix=$HOME/programmes/
```


### Feature Options

Most configuration scripts allow you to enable or disable certain features of the program or library to be compiled; they are generally of the type `--enable-feature` or `--disable-feature`. In high-performance computing, it is common for these options to include parallelization via threads or via MPI, for example:

```bash
[name@server ~]$ ./configure --enable-mpi
```

or

```bash
[name@server ~]$ ./configure --enable-threads
```

It is also common to have `--with-...` options to specifically parameterize the features.  It is generally recommended not to use these options and let Autoconf find the parameters automatically. However, it is sometimes necessary to specify parameters via the `--with-...` options, for example:

```bash
[name@server ~]$ ./configure --enable-mpi --with-mpi-dir=$MPIDIR
```


### Options Defined by Variables

You can usually specify the compiler to use and the options that must be passed to it by declaring variables after the `./configure` command. For example, to define the C compiler and the options to be passed to it, you could run:

```bash
[name@server ~]$ ./configure CC=icc CFLAGS="-O3 -xHost"
```

Here is a description of the most common options:

| Option      | Description                                      |
|-------------|--------------------------------------------------|
| `CFLAGS`    | for the C compiler                              |
| `CPPFLAGS`  | for the preprocessor and C, C++, Objective C and Objective C++ compilers |
| `CXXFLAGS`  | for the C++ compiler                            |
| `DEFS`      | to define a macro for the preprocessor           |
| `FCFLAGS`   | for the Fortran compiler                         |
| `FFLAGS`    | for the Fortran 77 compiler                      |
| `LDFLAGS`   | for the linker                                   |
| `LIBS`      | for the libraries to link                         |

The exhaustive list of typical variables and options is available in the [Autoconf documentation](link-to-autoconf-docs-here).


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Autotools/fr&oldid=67706](https://docs.alliancecan.ca/mediawiki/index.php?title=Autotools/fr&oldid=67706)"
