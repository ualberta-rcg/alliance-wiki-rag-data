# Tuning Lustre

Lustre is a high-performance distributed file system that allows you to perform input/output operations in parallel with a high throughput. However, there are some precautions to take if you want to achieve maximum performance. The advice presented here is for experienced users and should be followed with caution. Be sure to perform tests to verify the scientific validity of the results obtained and to ensure that the modifications result in a real improvement in performance.


## Parameters `stripe_count` and `stripe_size`

For each file or directory, it is possible to modify these parameters.

`stripe_count` is the number of disks on which the data is distributed;
`stripe_size` is the size of the smallest data block allocated in the file system.

It is possible to know the value of these parameters for a given file or directory with the command:

```bash
[name@server ~]$ lfs getstripe /path/to/file
```

Similarly, it is possible to modify these parameters for a given directory with the command:

```bash
[name@server ~]$ lfs setstripe -c count /path/to/dir
```

For example, if `count` = 8, the file will be distributed across eight RAID disks and each MB will be written sequentially to up to 8 servers.

```bash
[name@server ~]$ lfs setstripe -c 8 /home/user/newdir
```

Changing these parameters will not modify an existing file; to change them, you must migrate the file or copy it (and not move it) to a directory with different parameters. To create an empty file with particular values for `stripe_count` and `stripe_size` without modifying the directory parameters, you can run `lfs setstripe` on the name of the file you want to create: the file will be created empty and will have the specified parameters.


### Example of an unsegmented directory with the file `example_file` (`lmm_stripe_count` is equal to 1 and there is only one object)

```bash
$ lfs getstripe striping_example/
striping_example/
stripe_count:  1 stripe_size:   1048576 pattern:       raid0 stripe_offset: -1
striping_example//example_file
lmm_stripe_count:  1
lmm_stripe_size:   1048576
lmm_pattern:       raid0
lmm_layout_gen:    0
lmm_stripe_offset: 2
	obdidx		 objid		 objid		 group
	     2	       3714477	     0x38adad	   0x300000400
```

We can modify the segmentation of this directory to use 2 disks and create a new file.

```bash
$ lfs setstripe -c 2 striping_example
$ dd if=/dev/urandom of=striping_example/new_file bs=1M count=10
$ lfs getstripe striping_example/
striping_example/
stripe_count:  2 stripe_size:   1048576 pattern:       raid0 stripe_offset: -1
striping_example//example_file
lmm_stripe_count:  1
lmm_stripe_size:   1048576
lmm_pattern:       raid0
lmm_layout_gen:    0
lmm_stripe_offset: 2
	obdidx		 objid		 objid		 group
	     2	       3714477	     0x38adad	   0x300000400
striping_example//new_file
lmm_stripe_count:  2
lmm_stripe_size:   1048576
lmm_pattern:       raid0
lmm_layout_gen:    0
lmm_stripe_offset: 3
	obdidx		 objid		 objid		 group
	     3	       3714601	     0x38ae29	   0x400000400
	     0	       3714618	     0x38ae3a	   0x2c0000400
```

Only the file `new_file` uses `count=2` ( `lmm_stripe_count` ) by default and 2 objects are allocated.

To re-segment the old file, we use `lfs migrate`

```bash
$ lfs migrate -c 2 striping_example/example_file
$ lfs getstripe striping_example/example_file
striping_example/example_file
lmm_stripe_count:  2
lmm_stripe_size:   1048576
lmm_pattern:       raid0
lmm_layout_gen:    2
lmm_stripe_offset: 10
	obdidx		 objid		 objid		 group
	    10	       3685344	     0x383be0	   0x500000400
	    11	       3685328	     0x383bd0	   0x540000400
```

`lmm_stripe_count` is now 2 and two objects are allocated.


Increasing the number of disks can increase performance, but also makes the file more vulnerable to hardware failures.


When a parallel program needs to read a small file (< 1MB), such as a configuration file, it is more efficient to place this file on a single disk (`stripe count=1`), read it with the master process (`master rank`), and then send it to the other processes using `MPI_Broadcast` or `MPI_Scatter`.


When handling large data files, it is preferable to use as many disks as the number of MPI processes. The size will usually be the same as the size of the data buffer that is read or written by each process; for example, if each process reads 1MB of data at a time, then 1MB will probably be ideal. If you have no good reason to change this size, we recommend leaving it at its default value, which has been optimized for large files.


Note that the size must always be an integer multiple of 1MB.


In general, it is necessary to minimize the opening and closing of files. It will therefore be preferable to aggregate all the data into a single file rather than writing a multitude of small files. It will also be much better to open the file only once at the beginning of the execution and close it at the end, rather than opening and closing it within the same execution each time you want to add new data.


## For more information

[Archiving and compressing files](link-to-archiving-page)
