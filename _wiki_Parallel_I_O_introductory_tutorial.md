# Parallel I/O Introductory Tutorial

This self-study tutorial will discuss issues in handling large amounts of data in HPC and various parallel I/O strategies for large-scale Input/Output (I/O) with parallel jobs.  We will focus on using MPI-IO and introduce parallel I/O libraries such as NetCDF, HDF5, and ADIOS.

## HPC I/O Issues & Goal

Many modern problems are computationally expensive, requiring large parallel runs on large distributed-memory machines (clusters).  There are three main I/O activities in these jobs:

1.  Reading initial datasets or conditions from a file.
2.  Writing data to disk for post-processing (mostly at the end of a calculation). Parallel applications commonly write distributed arrays to disk.
3.  Writing application state to a file for restarting in case of system failure.

The figure below (not included in original text, needs to be added) shows a simple sketch of the I/O bottleneck problem when using many CPUs or nodes in a parallel job.  Amdahl's law states that the speedup of a parallel program is limited by the sequential fraction. If the I/O part is sequential, performance will not scale as desired.

Efficient I/O without stressing the HPC system is challenging. Load/store operations from memory or hard-disk are much slower than CPU multiply operations. Total execution time consists of computation time, communication time, and I/O time. Efficient I/O handling is key to optimal performance.

**Total Execution Time = Computation Time + Communication Time + I/O time**

Optimize all components of the above equation for best performance!

### Disk access rates over time

HPC I/O systems are typically slower than other components. The figure in this slide (not included in original text, needs to be added) shows how internal drive access rates have improved over time. From 1960 to 2014, top supercomputer speeds increased by 11 orders of magnitude. However, single hard disk drive capacity grew by 6 orders of magnitude, and average internal drive access rate grew by 3-4 orders of magnitude. This discrepancy means we produce much more data than we can store proportionally, requiring careful attention to data storage.

### How to calculate I/O speed

Two key performance measurements are:

*   **IOPs:** I/O operations per second (read/write, etc.). The inverse of latency.
*   **I/O Bandwidth:** The quantity read/written.

Parallel (distributed) filesystems are optimized for efficient I/O by multiple users on multiple machines/nodes; they do not provide "supercomputing" I/O performance.  Performance is limited by disk-access time and communication over the network (limited bandwidth, many users).

### I/O Software + Hardware stack

The I/O stack consists of:

1.  **I/O Hardware:** Physical hard disks attached to the cluster.
2.  **Parallel Filesystem:** (e.g., Lustre) Maintains logical partitions and provides efficient data access.
3.  **I/O Middleware:** Organizes access from many processes, optimizes I/O, and provides data sieving.
4.  **High-end I/O Library:** (e.g., HDF5, NetCDF) Maps application abstractions to storage abstractions.
5.  **Application:** Your program, which decides whether to use a high-end I/O library or I/O middleware.

## Parallel Filesystem

National systems use parallel filesystems designed to scale efficiently to tens of thousands of nodes. For better performance, files can be striped across multiple drives.  Parallel filesystems use locks to manage concurrent file access. Files are divided into "lock" units scattered across multiple hard drives. Client nodes obtain locks on units before I/O. This enables caching on clients; locks are reclaimed when others need access.

**Key Points:**

*   Optimized for large shared files.
*   Poor performance with many small reads/writes: Avoid millions of small files.
*   Your usage affects everyone! (Unlike CPU and RAM, which are not shared).
*   Critical factors: read/write methods, file format, number of files in a directory, and frequency of commands like `ls`.
*   The filesystem is shared over the network: Heavy I/O can impact process communication and affect other users.
*   Filesystems are limited: bandwidth, IOPs, number of files, space, etc.

### Best Practices for I/O

*   **Plan your data needs:** How much data will be generated? How much needs saving? Where will it be stored?
*   **Monitor and control usage:** Minimize use of filesystem commands like `ls` and `du` in large directories.
*   **Check disk usage regularly:** Use the `quota` command.
*   **Warning:** More than 100,000 files or average file size less than 100 MB (for large output) requires attention.
*   **Do housekeeping regularly:** Use `gzip`, `tar`, and delete unnecessary files.


## Data Formats

### ASCII

Human-readable but inefficient. Good for small input/parameter files.  Takes more storage and is expensive for read/write operations.  Use `fprintf()` in C or `open()` with the `formatted` option in Fortran.

**Pros:** Human-readable, portable.
**Cons:** Inefficient storage, expensive read/write.

### Binary

Much more efficient than ASCII.

**Experiment (data not shown, needs to be added):** Writing 128M doubles to `/scratch` and `/tmp` showed significantly faster write times for binary compared to ASCII.

**Pros:** Efficient storage, efficient read/write.
**Cons:** Requires knowledge of the format to read, portability issues (endians).  Use `fwrite()` in C or `open()` with the `unformatted` option in Fortran.

### MetaData (XML)

Stores additional information (number of variables, dimensions, sizes, etc.). Useful for describing binary files and sharing data with others or other programs.  Can also be done using HDF5 and NetCDF.

### Database

Good for many small records. Simplifies data organization and analysis.  Not common in numerical simulation.  Examples include SQLite, PostgreSQL, and MySQL.

### Standard scientific dataset libraries

HDF5 and NetCDF are excellent for storing large arrays efficiently. They include data descriptions (metadata), provide data portability, and offer optional compression.

## Serial and Parallel I/O

In large parallel calculations, datasets are distributed across many processors/nodes.  Using a parallel filesystem alone is insufficient; you must organize parallel I/O. Data can be written as raw binary, HDF5, or NetCDF.

### Serial I/O (Single CPU)

One approach is to have a "spokesperson" process collect data from other processes and then write it to a file. This is simple but doesn't scale well due to bandwidth limitations and memory constraints on the single node.

**Pros:** Trivially simple for small I/O.
**Cons:** Bandwidth limited, memory limitations, doesn't scale.

### Serial I/O (N processors)

Each process writes to its own file.  This is better than the single-CPU approach but still has limitations.  Many files can lead to poor parallel filesystem performance. Post-processing is also required to combine the data.

**Pros:** No interprocess communication.
**Cons:** Many small files (doesn't scale), requires post-processing, uncoordinated I/O can swamp the filesystem.

### Parallel I/O (N processes to/from 1 file)

Each process writes simultaneously to a single file using parallel I/O.  Coordination is crucial to avoid swamping the filesystem.

**Pros:** Only one file, avoids post-processing, scales if done correctly.
**Cons:** Uncoordinated I/O can swamp the filesystem, requires more design and planning.

### Parallel I/O should be collective!

Parallel middleware (like MPI-IO) offers coordinated and uncoordinated writing options. Collective I/O allows the filesystem to optimize operations for better performance.

## Parallel I/O techniques

### MPI-IO

Part of the MPI-2 standard.  Good for writing raw binary files. HDF5, NetCDF, and ADIOS are built on top of MPI-IO.  MPI-IO exploits analogies with MPI: writing/reading is similar to MPI send/receive. File access is grouped via communicators; user-defined MPI datatypes are available.

#### Basic MPI-IO operations in C

*   `MPI_File_open()`
*   `MPI_File_seek()`
*   `MPI_File_set_view()`
*   `MPI_File_read()`
*   `MPI_File_write()`
*   `MPI_File_close()`

#### Basic MPI-IO operations in F90

*   `MPI_FILE_OPEN()`
*   `MPI_FILE_SEEK()`
*   `MPI_FILE_SET_VIEW()`
*   `MPI_FILE_READ()`
*   `MPI_FILE_WRITE()`
*   `MPI_FILE_CLOSE()`

#### Opening a file requires a...

*   Communicator
*   Filename
*   File handle
*   File access mode (`MPI_MODE_RDONLY`, `MPI_MODE_RDWR`, `MPI_MODE_WRONLY`, etc.)
*   Info argument (usually `MPI_INFO_NULL`)

#### C example (Read/Write contiguous data)

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

#### F90 example (Not included in original text, needs to be added)


#### Summary: MPI-IO

MPI-IO is part of the standard MPI-2 library and is widely installed. It's straightforward to implement but writes raw data, making it non-portable, difficult to append variables, and lacking data descriptions.

### NetCDF

NetCDF addresses the limitations of MPI-IO. It uses MPI-IO internally but simplifies data storage. Data is stored as binary, is self-describing (metadata in the header), and is portable.  Optional compression is available.  Supports various visualization packages (e.g., Paraview).

#### example in C (Not included in original text, needs to be added)

### HDF5

HDF5 is similar to NetCDF, supporting self-describing file formats, using MPI-IO internally, and offering optional compression. It's more general than NetCDF, with an object-oriented description of datasets, groups, attributes, etc.

## References

*   [Link 1](https://www.nhr.kit.edu/userdocs/horeka/parallel_IO/)
*   [Link 2](https://hpc-forge.cineca.it/files/CoursesDev/public/2017/Parallel_IO_and_management_of_large_scientific_data/Roma/MPI-IO_2017.pdf)
*   [Link 3](https://janth.home.xs4all.nl/MPIcourse/PDF/08_MPI_IO.pdf)
*   [Link 4](https://events.prace-ri.eu/event/176/contributions/59/attachments/170/326/Advanced_MPI_II.pdf)
*   [Link 5](https://www.cscs.ch/fileadmin/user_upload/contents_publications/tutorials/fast_parallel_IO/MPI-IO_NS.pdf)

