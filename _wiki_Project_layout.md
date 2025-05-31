# Project Layout

The project filesystem on Béluga, Cedar, Graham, and Narval is organized by groups.  Access is typically via symbolic links in your home directory, formatted as `$HOME/projects/group_name`.

The group space is owned by the principal investigator (PI), with group members having read and write permissions.  However, newly created files are only readable by group members by default.  To create writable files, create a special directory:

```bash
mkdir $HOME/projects/def-profname/group_writable
setfacl -d -m g::rwx $HOME/projects/def-profname/group_writable
```

For more on data sharing, file ownership, and ACLs, see [Sharing data](link-to-sharing-data-page).


## Quotas and Storage

The project space has a default quota of 1 TB and 500,000 files per group, expandable to 10 TB upon request to [Technical support](link-to-technical-support).  Some groups may have larger quotas from the annual [Resource Allocation Competition](link-to-rac).  This storage is cluster-specific and not transferable.

Check usage with:

```bash
diskusage_report
```

To ensure files inherit the correct group membership (and count against the group quota), set the `setgid` bit on the directory:

```bash
chmod g+s <directory name>
```

For existing subdirectories:

```bash
find <directory name> -type d -print0 | xargs -0 chmod g+s
```

More on `setgid` is available [here](link-to-setgid-page).


## Modifying Group Membership

Use `newgrp` to temporarily change your default group:

```bash
newgrp rrg-profname-ab
```

This only affects the current session.


## Disk Quota Exceeded Errors

`disk quota exceeded` errors (see [Disk quota exceeded error on /project filesystems](link-to-disk-quota-error-page)) often result from files associated with the wrong group (your personal group, with a 2 MB quota).  Fix this with:

```bash
find <directory name> -group $USER -print0 | xargs -0 chgrp -h <group>
```

Replace `<group>` with the correct group (e.g., `def-profname`).


## Example: Sharing Data

Imagine PI "Sue" and sponsored user "Bob".  Their directory structures might look like this:

* `/home/sue/scratch` (symbolic link to `/scratch/sue`)
* `/home/sue/projects` (directory)
* `/home/bob/scratch` (symbolic link to `/scratch/bob`)
* `/home/bob/projects` (directory)

If Bob's only role is sponsored by Sue, both `/home/sue/projects/def-sue` and `/home/bob/projects/def-sue` point to the same location (`/project/<some random number>`), ideal for shared data.

A RAC award for Sue would add a directory like `~/projects/rrg-sue-ab` for RAC-related data.

Sharing with someone without a sponsored role (e.g., Heather) requires changing file permissions (see [Sharing data](link-to-sharing-data-page)).  ACLs are recommended.  Heather needs read and execute permissions on all directories between `~/projects/def-sue` and the shared file(s).  Sharing with non-Alliance users is possible via a Globus shared endpoint.  Sponsoring a role for Heather simplifies collaboration.


## Summary

*   **scratch:** Private temporary files.
*   **home:** Small amounts of relatively private data.
*   **project:** Shared data for a research group (persistent, backed up, up to 10 TB or more with a RAC).

