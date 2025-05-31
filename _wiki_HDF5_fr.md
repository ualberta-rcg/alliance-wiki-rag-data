# HDF5

This page is a translated version of the page HDF5 and the translation is 100% complete.

Other languages: English, français


## Généralités

### Description

HDF5 (for Hierarchical Data Format) is a library for formatting scientific data and facilitates its storage, reading, visualization, manipulation, and analysis. It handles all types of data and its design allows for both flexible and efficient input/output and support for large volumes of data. It is portable and extensible and can accompany applications in their evolution.

The HDF5 Technology suite includes tools and applications for managing, manipulating, visualizing, and analyzing data in HDF5 format.

HDF (also called HDF4) is a multi-format library and file format for storage and management on multiple computers. HDF4 is the original format and although it is still supported, the HDF5 version is recommended.


HDF was designed for:

* Large volumes of data and complex data, but can be used for low volumes and simple data
* All sizes and all types of systems (portable)
* Flexible and efficient storage and input/output
* Applications can evolve and handle new models

HDF includes:

* A file format for storing HDF4/HDF5 data
* A model for organizing and accessing HDF4/HDF5 data with various applications
* Several software including libraries, language modules, and several format-specific tools


**References:**

* **Project website:** https://www.hdfgroup.org/solutions/hdf5/
* **Documentation:** https://support.hdfgroup.org/documentation/
* **Download:** https://www.hdfgroup.org/downloads/hdf5


### Points forts

* Data is independent of the hardware architecture (endianness).
* Data structured in physical units allows tracking of relevant information.
* Usable in parallel (MPI-IO)
* Data can be compressed when writing (zlib or szip).
* Interfaces for C, C++, Fortran 90, Java, and Python
* Manages all data types (more than NetCDF).
* Reading and writing in Matlab's .mat format.
* Free for most platforms


### Points faibles

* More complicated interface than NetCDF.
* HDF5 does not require UTF-8; ASCII is usually used.
* Datasets cannot be released without creating a copy of the file with another tool.


## Guide de démarrage

We address the configuration details here.


### Modules d'environnement

The following modules are available on Cedar and Graham via CVMFS:

* `hdf`: version 4.1 and earlier
* `hdf5`: latest version of HDF5
* `hdf5-mpi`: to use MPI

Execute `module avail hdf` to know the available versions for the compiler and the MPI modules you have loaded. For the complete list of HDF4/HDF5 modules, execute `module -r spider '.*hdf.*'`.

Use `module load hdf/version` or `module load hdf5/version` to configure the environment according to the selected version. For example, to load HDF5 version 1.8.18, run:

```bash
[name@server ~]$ module load hdf5/1.8.18
```


### Scripts de soumission de tâche

For examples of scripts for the Slurm scheduler, see [Exécuter des tâches](link-to-executing-tasks-page). We recommend using the `module load ...` command in your script.


### Lier à des bibliothèques HDF

Here are examples in sequential and parallel mode:


#### Mode séquentiel

```bash
[name@server ~]$ module load hdf5/1.8.18
[name@server ~]$ gcc example.c -lhdf5
```


#### Mode parallèle

```bash
[name@server ~]$ module load hdf5-mpi/1.8.18
[name@server ~]$ mpicc example.c -lhdf5
```


#### Exemple

See [an example](link-to-example-page) of reading and writing in a dataset. Integers are first written with data space dimensions of DIM0xDIM1, then the file is closed. The file is then reopened, the data is read and displayed.

Compile and run with:

```bash
[name@server ~]$ module load hdf5-mpi
[name@server ~]$ mpicc h5ex_d_rdwr.c -o h5ex_d_rdwr -lhdf5
[name@server ~]$ mpirun -n 2 ./h5ex_d_rdwr
```


### Utilitaires

You will find [the complete list](link-to-hdfgroup-utilities-page) on the Hdfgroup website.  The following utilities are highlighted:

* **HDF5 ODBC Connector:** SQL interface for HDF5 data format in Excel, Tableau, and others
* **HDFView:** Java browser and object package for HDF5-1.10 (64-bit object identification) and HDF 4.2.12 (and later)
* Several command-line tools:
    * `gif2h5/h52gif`
    * `h5cc`, `h5fc`, `h5c++`
    * `h5debug`
    * `h5diff`
    * `h5dump`
    * `h5import`
    * `h5check`: verification of the validity of an HDF5 file
    * `h5edit`: editing tools


**(Note:  Replace bracketed links like `[link-to-executing-tasks-page]` with actual links to relevant pages within your documentation.)**
