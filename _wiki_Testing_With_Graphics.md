# Testing With Graphics

This is a draft, a work in progress that is intended to be published into an article, which may or may not be ready for inclusion in the main wiki. It should not necessarily be considered factual or authoritative.

If you need to use graphics while testing your code, e.g., when using a debugger such as DDT or DDD, you have the following options:

## Use the `debugjob` command

You can use the `debugjob` command which automatically provides X-forwarding support.

```bash
ssh niagara.scinet.utoronto.ca -X
```

```
USER@nia-login07:~$ debugjob
debugjob: Requesting 1 nodes for 60 minutes
xalloc: Granted job allocation 189857
xalloc: Waiting for resource configuration
xalloc: Nodes nia0030 are ready for job
```

```
[USER@nia1265 ~]$
```

## Use the regular queue

If `debugjob` is not suitable for your case due to the limitations either on time or resources (see above #Testing), then you have to follow these steps:

You will need two terminals in order to achieve this:

**In the 1st terminal:**

1. `ssh` to `niagara.scinet.utoronto.ca` and issue your `salloc` command.
2. Wait until your resources are allocated and you are assigned the nodes.
3. Take note of the node where you are logged to, i.e., the head node, let's say `niaWXYZ`.

```bash
ssh niagara.scinet.utoronto.ca
USER@nia-login07:~$ salloc --nodes 5 --time=2:00:00
```

```
.salloc: Granted job allocation 141862
.salloc: Waiting for resource configuration
.salloc: Nodes nia1265 are ready for job
```

```
[USER@nia1265 ~]$
```

**On the second terminal:**

1. `ssh` into `niagara.scinet.utoronto.ca` now using the `-X` flag in the `ssh` command.
2. After that, `ssh -X niaWXYZ`, i.e., you will `ssh` carrying on the `-X` flag into the head node of the job.
3. In `niaWXYZ` you should be able to use graphics and should be redirected by x-forwarding to your local terminal.

```bash
ssh niagara.scinet.utoronto.ca -X
USER@nia-login07:~$ ssh -X nia1265
```

```
[USER@nia1265 ~]$ xclock  ## just an example to test the graphics, a clock should pop up, close it to exit
[USER@nia1265 ~]$ module load ddt  ## load corresponding modules, eg. for DDT
[USER@nia1265 ~]$ ddt  ## launch DDT, the GUI should appear in your screen
```

**Observations:**

*   If you are using `ssh` from a Windows machine, you need to have an X-server; a good option is to use MobaXterm, which already includes an X-server.
*   If you are in Mac OS, substitute `-X` by `-Y`.
*   Instead of using two terminals, you could just use `screen` to request the resources and then detach the session and `ssh` into the head node directly.


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Testing\_With\_Graphics&oldid=66473](https://docs.alliancecan.ca/mediawiki/index.php?title=Testing_With_Graphics&oldid=66473)"
