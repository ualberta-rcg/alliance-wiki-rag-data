# Scratch Purging Policy

The scratch filesystem on our clusters is intended as temporary, fast storage for data used during job execution. Data needed for long-term storage and reference should be kept in `/project` or other archival storage areas. To ensure adequate space on scratch, files older than 60 days are periodically deleted according to the policy outlined below. Note that purging is based on file age, not location; moving a file within scratch won't prevent it from being purged.

On Graham, the scratch filesystem doesn't explicitly expire but uses quota to enforce transient use.


## Expiration Procedure

The scratch filesystem is checked at the end of each month for files to be expired on the 15th of the following month. On the first of the month, a login message and notification email are sent to users with files slated for purging.  These emails contain a file listing all candidates for purging. You then have two weeks to copy data to your project space or another location.

On the 12th, a final notification email is sent with an updated assessment, giving you 72 hours to move files. On the 15th, any remaining files with `ctime` and `atime` older than 60 days will be deleted.  The email reminders and login notices are a courtesy; users are ultimately responsible for managing files older than 60 days.

Simply copying or using `rsync` to move files updates the `atime`, making them ineligible for deletion. After moving data, delete the originals from scratch.


## How/Where to Check Files Slated for Purging

*   **Cedar, Beluga, and Narval clusters:** `/scratch/to_delete/` (look for a file with your name)
*   **Niagara:** `/scratch/t/to_delete/` (symlink to `/scratch/t/todelete/current`)

The file lists filenames with full paths, possibly including `atime`, `ctime`, and size. It's updated on the 1st and 12th of each month.  A file with your name indicates you have candidates for purging.

*   Accessing/reading/moving/deleting candidates between the 1st and 11th doesn't change the assessment until the 12th.
*   If an assessment file exists until the 11th but not the 12th, you no longer have anything to be purged.
*   If you access/read/move/delete candidates after the 12th, you must check yourself to confirm files won't be purged on the 15th.


## How Do I Check the Age of a File?

A file's age is the most recent of its access time (`atime`) and change time (`ctime`).

*   **`ctime`:** `ls -lc <filename>`
*   **`atime`:** `ls -lu <filename>`

We don't use the modify time (`mtime`) as it can be modified by users or programs, leading to incorrect information.  Normally, `atime` would suffice, as it's updated with `ctime`. However, userspace programs can alter `atime`, potentially to past times, causing early expiration. Using `ctime` as a fallback prevents this.


## Abuse

Periodically running a recursive `touch` command to prevent expiration is considered abuse.  Our staff detect this, and users employing such techniques will be contacted and asked to move data from scratch to a more appropriate location.


## How to Safely Copy a Directory with Symlinks

`cp` or `rsync` usually suffice, but copying symlinks causes problems as they still point to scratch. Use `tar` to archive and extract:

```bash
cd /scratch/.../your_data
mkdir /project/.../your_data
tar cf - ./* | (cd /project/.../your_data && tar xf -)
```
