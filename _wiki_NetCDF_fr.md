# NetCDF

NetCDF (for Network Common Data Form) is:

*   an interface for array-oriented data access, and
*   a library that provides an implementation of this interface.

Its self-documenting, machine-independent data format allows for the creation, access, and sharing of scientific data.  Several modifications were made to the library with version 4 released in 2008; previous versions will not be discussed here. NetCDF 4.x is backward compatible, but older versions cannot use the new files.

**Project Website:** https://www.unidata.ucar.edu/software/netcdf

**Documentation:** https://www.unidata.ucar.edu/software/netcdf/docs

**Downloads:** https://www.unidata.ucar.edu/downloads/netcdf/index.jsp

**FAQ:** https://www.unidata.ucar.edu/software/netcdf/docs/faq.html


## Généralités

### Points forts

*   Data is machine-independent ([https://fr.wikipedia.org/wiki/Endianness endianness]).
*   Data structured in physical units allows for tracking relevant information.
*   NetCDF4 writes and reads in parallel if built with a parallel version of HDF5.
*   Data can be compressed on writing.
*   Simpler interface than HDF5.
*   Free for most platforms.

### Points faibles

*   The Python interface does not allow parallelization (version 1.0.5).
*   Some files produced with HDF5 cannot be read by NetCDF.


## Guide de démarrage

We address here the configuration details.

### Modules d'environnement

The following modules are available via CVMFS:

*   `netcdf`: for linking with programs containing only C instructions.
*   `netcdf-c++`: for linking with programs containing C and C++ instructions.
*   `netcdf-fortran`: for linking with programs containing Fortran instructions.

Other modules use MPI to allow parallel input/output:

*   `netcdf-mpi`: for linking with programs containing C instructions and calling MPI libraries.
*   `netcdf-c++-mpi`: for linking with programs containing C and C++ instructions and calling MPI libraries.
*   `netcdf-fortran-mpi`: for linking with programs containing Fortran instructions and calling MPI libraries.


Execute `module avail netcdf` to know the available versions for the compiler and MPI modules you have loaded. For the complete list of NetCDF modules, execute `module -r spider '.*netcdf.*'`.

Use `module load netcdf/version` to configure the environment according to the selected version. For example, to load the NetCDF version 4.1.3 library for C, run:

```bash
[name@server ~]$ module load netcdf/4.1.3
```

### Soumettre un script

Consult "Running Jobs" for examples of scripts submitted to the Slurm scheduler. We recommend including the `module load ...` command in your script.

### Lier des programmes à des bibliothèques NetCDF

The following examples show how to link NetCDF libraries to programs in C and Fortran.

#### NetCDF en série

**C Program:**

```bash
[name@server ~]$ module load netcdf/4.4.1
[name@server ~]$ gcc example.c -lnetcdf
```

**Fortran Program:** It is necessary to specify two libraries in the appropriate order.

```bash
[name@server ~]$ module load gcc netcdf-fortran
[name@server ~]$ gfortran example.f90 -I $EBROOTNETCDFMINFORTRAN/include -lnetcdf -lnetcdff
```

#### NetCDF en parallèle

**C Program using MPI:**

```bash
[name@server ~]$ module load netcdf-mpi
[name@server ~]$ gcc example.c -lnetcdf
```

#### Exemple

In this example, a NetCDF file is created and contains a single two-dimensional variable named `data` whose dimensions are `x` and `y`.

To compile the example:

```bash
[name@server ~]$ module load netcdf
[name@server ~]$ gcc ex_netcdf4.c -lnetcdf
```

### Utilitaires

Several utilities can read and write files in different formats.

*   **`ncdump`**: This tool generates the CDL text representation of a netCDF dataset with the option to exclude some or all variable data. The result can in principle be used as input with `ncgen`. `ncdump` and `ncgen` can therefore be used to convert a binary representation to a text representation and vice versa. See the [ncdump section](link_to_ncdump_section_on_UCAR_website) of the UCAR website.

*   **`ncgen`**: Conversely to `ncdump`, this tool generates a NetCDF binary file. See the [ncgen section](link_to_ncgen_section_on_UCAR_website).

*   **`nccopy`**: Copies a netCDF file, allowing modification of the binary format, block size, compression, and other storage parameters. See the [nccopy section](link_to_nccopy_section_on_UCAR_website).

To help you find the commands for linking and compiling, use the utilities `nf-config` and `nc-config`; see the [documentation](link_to_documentation).


## PnetCDF

PnetCDF is another library for reading and writing files in NetCDF format. The names of its procedures are different from those of NetCDF. The library also offers non-blocking procedures. See the [PnetCDF website](link_to_PnetCDF_website) for more information.


**(Note:  Please replace bracketed placeholders like `[link_to_ncdump_section_on_UCAR_website]` with the actual links.)**
