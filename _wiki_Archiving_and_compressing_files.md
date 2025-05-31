# Archiving and Compressing Files

Archiving means creating one file that contains a number of smaller files within it. Reducing the number of files by creating an archive can improve the efficiency of file storage and help you stay within quota limits. Archiving can also improve the efficiency of file transfers. It is faster for the secure copy protocol (`scp`), for example, to transfer one archive file of a reasonable size than thousands of small files of equal total size.

Compressing means encoding a file such that the same information is contained in fewer bytes of storage. The advantage for long-term data storage should be obvious. For data transfers, the time spent for compressing data must be balanced against the time saved moving fewer bytes as described in this discussion of [data compression and transfer](link_to_external_resource_here) from the US National Center for Supercomputing Applications.  *(Note:  Please replace `link_to_external_resource_here` with the actual URL)*


The best-known tool for archiving files in the Linux community is `tar`. Here is [a tutorial on 'tar'](link_to_tar_tutorial_here). *(Note: Please replace `link_to_tar_tutorial_here` with the actual URL)*

A replacement for `tar` called `dar` offers some advantages in functionality. Here is [a tutorial on 'dar'](link_to_dar_tutorial_here). *(Note: Please replace `link_to_dar_tutorial_here` with the actual URL)*  Both `tar` and `dar` can compress files as well as archive.

The `zip` utility, more commonly used in the Windows community but available on our clusters, also provides both archiving and compression.

Compression tools `gzip`, `bzip2`, and `xz` can be used in conjunction with `tar`, or by themselves.
