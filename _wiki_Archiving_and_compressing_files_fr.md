# Archiving and Compressing Files

This page is a translated version of the page [Archiving and compressing files](https://docs.alliancecan.ca/mediawiki/index.php?title=Archiving_and_compressing_files&oldid=74185) and the translation is 100% complete.

Other languages:

*   [English](https://docs.alliancecan.ca/mediawiki/index.php?title=Archiving_and_compressing_files&oldid=74185)
*   français

## Archiving

Archiving means creating a file that contains multiple smaller files. Creating an archive file can improve storage efficiency and help you meet quotas. Archiving can also make transferring files more efficient. For example, the `scp` (secure copy protocol) transfers a reasonably sized archive file faster than thousands of small files totaling the same size.

## Compressing

Compressing means modifying a file's code to reduce the number of bits. The advantages are obvious regarding long-term data storage. In the case of data transfer, one must compare the compression time to the time required to move a smaller amount of bits; see [this text](<link_to_text_needed>) from the National Center for Supercomputing Applications.


## Tools

Under Linux, `tar` is a well-known archiving and compression tool; see the [tar tutorial](<link_to_tutorial_needed>).

Also for archiving and compression, `dar` offers some advantageous functions; see the [dar tutorial](<link_to_tutorial_needed>).

The `zip` utility is well known for archiving and compression in the Windows environment, but it is available with Compute Canada clusters.

The compression tools `gzip`, `bzip2`, and `xz` can be used by themselves or with `tar`.


**(Note:  The placeholders `<link_to_text_needed>` and `<link_to_tutorial_needed>` should be replaced with the actual URLs to the referenced text and tutorials.)**
