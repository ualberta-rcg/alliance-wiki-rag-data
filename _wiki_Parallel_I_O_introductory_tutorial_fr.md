# Parallel I/O Introductory Tutorial

This self-study tutorial will discuss issues in handling large amounts of data in HPC and various parallel I/O strategies for large-scale Input/Output (I/O) with parallel jobs.  We will focus on using MPI-IO and then introduce parallel I/O libraries such as NetCDF, HDF5, and ADIOS.

## HPC I/O Issues & Goal

Many modern problems are computationally expensive, requiring large parallel runs on large distributed-memory machines (clusters).  There are three main I/O activities in these jobs:

1.  Reading initial datasets or conditions from a file.
2.  Writing data to disk for post-processing (mostly at the end of a calculation). Parallel applications commonly need to write distributed arrays to disk.
3.  Writing application state to a file for restarting in case of system failure.

The figure below shows a simple sketch of the I/O bottleneck problem when using many CPUs or nodes in a parallel job.  Amdahl's law states that the speedup of a parallel program is limited by the sequential fraction. If the I/O part works sequentially, performance will not scale as desired.

Efficient I/O without stressing the HPC system is challenging. Load/store operations from memory or hard-disk take much longer than CPU multiply operations.  Total execution time consists of computation time, communication time, and I/O time. Efficient I/O handling is key to optimal performance.

**Equation:** Total Execution Time = Computation Time + Communication Time + I/O time

**Goal:** Optimize all components of the above equation for best performance!

### Disk Access Rates Over Time

I/O systems in HPC are typically slower than other parts. The figure in this slide shows how internal drive access rates have improved over time. From 1960 to 2014, top supercomputer speeds increased by 11 orders of magnitude. However, single hard disk drive capacity grew by only 6 orders of magnitude, and average internal drive access rate grew by 3-4 orders of magnitude. This discrepancy means we produce far more data than we can store proportionally, necessitating careful data storage strategies.

### How to Calculate I/O Speed

We need to understand two performance measurements:

*   **IOPs (I/O Operations Per Second):** Includes read/write, etc.  It's the inverse of latency.
*   **I/O Bandwidth:** The quantity read/written (similar to internet connection bandwidth).

Parallel (distributed) filesystems are optimized for efficient I/O by multiple users on multiple machines/nodes; they do not provide "supercomputing" I/O performance.  Performance is limited by disk access time and network communication (limited bandwidth, many users).

### I/O Software + Hardware Stack

The I/O stack consists of several layers:

1.  **I/O Hardware:** Physical hard disks attached to the cluster.
2.  **Parallel Filesystem:** (e.g., Lustre) Maintains logical partitions and provides efficient data access.
3.  **I/O Middleware:** Organizes access from many processes, optimizes I/O, and provides data sieving.
4.  **High-End I/O Library:** (e.g., HDF5, NetCDF) Maps application abstractions to storage abstractions.
5.  **Application:** Your program, which decides whether to use a high-end I/O library or I/O middleware.

## Parallel Filesystem

National systems use parallel filesystems designed to scale efficiently to tens of thousands of nodes.  For better performance, files can be striped across multiple drives.  Parallel filesystems use locks to manage concurrent file access. Files are divided into "lock" units scattered across multiple drives. Client nodes obtain locks on units before I/O. This enables caching on clients; locks are reclaimed when others need access.

**Key Points:**

*   Optimized for large shared files.
*   Poor performance with many small reads/writes: Avoid millions of small files.
*   Your usage affects everyone! (Unlike CPU and RAM, which are not shared).
*   Critical factors: read/write methods, file format, number of files in a directory, and frequency of commands like `ls`.
*   The filesystem is shared over the network: Heavy I/O can hinder process communication.
*   Filesystems are limited: bandwidth, IOPs, number of files, space, etc.

### Best Practices for I/O

*   **Plan your data needs:** How much will you generate? How much needs saving? Where will you keep it?
*   **Monitor and control usage:** Minimize use of filesystem commands like `ls` and `du` in large directories.
*   **Check disk usage regularly:** Use the `quota` command.
*   **Warning:** More than 100,000 files or average data file size less than 100 MB (for large output) requires attention.
*   **Do housekeeping regularly:** Use `gzip`, `tar`, and delete unnecessary files.


## Data Formats

### ASCII

Human-readable but inefficient. Good for small input or parameter files.  Takes more storage and is expensive for read/write operations.  (e.g., `fprintf()` in C, `open(..., form='formatted')` in Fortran).

**Pros:** Human-readable, portable.
**Cons:** Inefficient storage, expensive read/write.

### Binary

Much more efficient than ASCII.

**Example (writing 128M doubles):**

| Format     | /scratch | /tmp (disk) |
| ----------- | -------- | ----------- |
| ASCII       | 173 s    | 260 s       |
| Binary      | 6 s      | 20 s        |

**Pros:** Efficient storage, efficient read/write.
**Cons:** Requires knowledge of the format to read, portability issues (endians). (e.g., `fwrite()` in C, `open(..., form='unformatted')` in Fortran).

### MetaData (XML)

Stores additional information about the data (number of variables, dimensions, sizes, etc.). Useful for sharing binary files with others or other programs.  Can also be handled by HDF5 and NetCDF.

### Database

Good for many small records. Simplifies data organization and analysis.  (e.g., SQLite, PostgreSQL, MySQL).

### Standard Scientific Dataset Libraries

(e.g., HDF5, NetCDF) Efficiently store large arrays, include data descriptions (metadata), provide data portability, and offer optional compression.

## Serial and Parallel I/O

In large parallel calculations, datasets are distributed across many processors/nodes.  Using a parallel filesystem alone is insufficient; you must organize parallel I/O. Data can be written as raw binary, HDF5, or NetCDF.

### Serial I/O (Single CPU)

A "spokesperson" process collects data from other processes and writes it to a single file.  Simple but doesn't scale well due to bandwidth and memory limitations on the single node.

**Pros:** Trivially simple for small I/O.
**Cons:** Bandwidth limited, may not have enough memory, won't scale.

### Serial I/O (N Processors)

Each process writes to its own file.  Better than single-CPU serial I/O, but creates many small files, leading to poor filesystem performance and requiring post-processing.

**Pros:** No interprocess communication. Possibly better scaling than single sequential I/O.
**Cons:** Many small files (won't scale), data often needs post-processing, uncoordinated I/O can swamp the filesystem.

### Parallel I/O (N Processes to/from 1 File)

Each process writes simultaneously to a single file using parallel I/O.  Requires coordinated I/O to avoid swamping the filesystem.

**Pros:** Only one file, avoids post-processing, scales if done correctly.
**Cons:** Uncoordinated I/O will swamp the filesystem, requires careful design.

### Parallel I/O Should Be Collective!

Collective I/O (coordinated access) allows the filesystem to optimize operations, improving performance.  Independent I/O only specifies what a single process will do.

## Parallel I/O Techniques

### MPI-IO

Part of the MPI-2 standard. Good for writing raw binary files.  HDF5, NetCDF, and ADIOS are built on top of MPI-IO.

**Key Features:**

*   Part of the MPI-2 standard.
*   ROMIO is the implementation in OpenMPI (default on many systems).
*   Exploits analogies with MPI (writing/reading similar to send/receive).
*   File access grouped via communicator: collective operations.
*   User-defined MPI datatypes.

#### Basic MPI-IO Operations in C

*   `MPI_File_open()`
*   `MPI_File_seek()`
*   `MPI_File_set_view()`
*   `MPI_File_read()`
*   `MPI_File_write()`
*   `MPI_File_close()`

#### Basic MPI-IO Operations in Fortran 90

*   `MPI_FILE_OPEN()`
*   `MPI_FILE_SEEK()`
*   `MPI_FILE_SET_VIEW()`
*   `MPI_FILE_READ()`
*   `MPI_FILE_WRITE()`
*   `MPI_FILE_CLOSE()`

#### Opening a File Requires a...

*   Communicator
*   Filename
*   File handle
*   File access mode (`MPI_MODE_RDONLY`, `MPI_MODE_RDWR`, `MPI_MODE_WRONLY`, etc.)  Combine modes using bitwise OR ("|" in C, "+" in Fortran).
*   Info argument (usually `MPI_INFO_NULL`).

#### C Example (Contiguous Data)

```c
#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv) {
    int rank, i; char a[10];
    MPI_Offset n = 10; MPI_File fh; MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    for (i=0; i<10; i++)
        a[i] = (char)( '0' + rank);
    MPI_File_open (MPI_COMM_WORLD, "data.out", MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    MPI_Offset displace = rank*n*sizeof(char);
    MPI_File_set_view (fh , displace , MPI_CHAR, MPI_CHAR, "native" ,MPI_INFO_NULL);
    MPI_File_write(fh, a, n, MPI_CHAR, &status);
    MPI_File_close(&fh );
    MPI_Finalize ();
    return 0;
}
```

#### Summary: MPI-IO

MPI-IO is part of the standard MPI-2 library and widely installed.  It's straightforward to implement but writes raw data (not portable, hard to append variables, no data description).

### NetCDF

Popular package for storing data.  Uses MPI-IO under the hood but simplifies data storage.  Data is stored as binary, is self-describing (metadata in header), and is portable.  Supports optional compression and various visualization packages (e.g., Paraview).

#### Example in C (NetCDF)

```c
#include <stdlib.h>
#include <stdio.h>
#include <netcdf.h>
#define FILE_NAME "simple_xy.nc" #define NDIMS 2
#define NX 3
#define NY 4
int main() {
    int ncid, x_dimid, y_dimid, varid; int dimids[NDIMS];
    int data_out[NX][NY];
    int x, y, retval;
    for (x = 0; x < NX; x++)
         for (y = 0; y < NY; y++)
              data_out[x][y] = x * NY + y;
    retval = nc_create(FILE_NAME, NC_CLOBBER, &ncid);
    retval = nc_def_dim(ncid, "x", NX, &x_dimid);
    retval = nc_def_dim(ncid, "y", NY, &y_dimid);
    dimids[0] = x_dimid;
    dimids[1] = y_dimid;
    retval = nc_def_var(ncid, "data", NC_INT, NDIMS, dimids, &varid);
    retval = nc_enddef(ncid);
    retval = nc_put_var_int(ncid, varid, &data_out[0][0]);
    retval = nc_close(ncid);
    return 0;
}
```

### HDF5

Also very popular. Supports most NetCDF features. More general than NetCDF with an object-oriented description of datasets, groups, attributes, types, data spaces, and property lists.

## References

*   [Link 1](https://www.nhr.kit.edu/userdocs/horeka/parallel_IO/)
*   [Link 2](https://hpc-forge.cineca.it/files/CoursesDev/public/2017/Parallel_IO_and_management_of_large_scientific_data/Roma/MPI-IO_2017.pdf)
*   [Link 3](https://janth.home.xs4all.nl/MPIcourse/PDF/08_MPI_IO.pdf)
*   [Link 4](https://events.prace-ri.eu/event/176/contributions/59/attachments/170/326/Advanced_MPI_II.pdf)
*   [Link 5](https://www.cscs.ch/fileadmin/user_upload/contents_publications/tutorials/fast_parallel_IO/MPI-IO_NS.pdf)

