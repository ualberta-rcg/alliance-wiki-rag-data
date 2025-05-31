# Ask Support

Before contacting us, please check the [system status page](link-to-system-status-page) and the [known issues page](link-to-known-issues-page) to see if your problem has already been reported. If you can't find the answer on our wiki, send an email to the address below that best matches your needs.

Ensure the email address you use is registered with your account. This allows our ticketing system to recognize you automatically.

A well-written question (or problem description) will result in faster, more accurate assistance. (See the [support request example](#support-request-example) below).

Emails with vague subjects like "Something is wrong" or "Nothing works" will take longer to resolve, as we'll need to request missing information (see [Information required](#information-required)).

In the subject line, include the system/cluster name and a brief description of the problem. For example: "Job 123456 fails to run on the Cedar cluster".  A good subject line helps us identify issues quickly.

Do not request help on a different topic as a follow-up to an old email thread. Start a new email to avoid reopening an old ticket.


## Email Addresses

Choose the address that best corresponds to your question or issue:

*   `accounts@tech.alliancecan.ca` -- Questions about accounts
*   `renewals@tech.alliancecan.ca` -- Questions about account renewals
*   `globus@tech.alliancecan.ca` -- Questions about Globus file transfer services
*   `cloud@tech.alliancecan.ca` -- Questions about using Cloud resources
*   `allocations@tech.alliancecan.ca` -- Questions about the Resource Allocation Competition (RAC)
*   `support@tech.alliancecan.ca` -- For any other question or issue


## Information Required

To help us assist you better, please include the following information in your support request:

*   Cluster name
*   Job ID
*   Job submission script: Provide the full path of the script on the cluster, copy and paste the script, or attach the script file.
*   File(s) containing error message(s): Provide the full path, copy and paste the file(s), or attach the error message file(s).
*   Commands you were executing
*   Avoid sending screenshots or large image attachments unless necessary. Plain text (commands, job scripts, etc.) is usually more helpful.  See [Copy and paste](link-to-copy-paste-instructions) if you have trouble with this.
*   Software (name and version) you were using
*   When the problem occurred

If you want us to access, copy, edit your files, or inspect/modify your account, explicitly state this in your email. For example, instead of attaching files, indicate their location and grant us permission to access them. If you've already granted permission via the CCDB interface, you don't need to do so again.


## Things to Beware

*   Never send a password!
*   Maximum attachment size is 40 MB.


## Support Request Example

To: `support@tech.alliancecan.ca`
Subject: Job 123456 gives errors on the CC Cedar cluster

Hello:

My name is Alice, user asmith. Today at 10:00 am MST, I submitted job 123456 on the Cedar cluster. The job script is located at `/my/job/script/path`. I haven't changed it since submission. Since it's short, I've included it below:

```bash
#!/bin/bash
#SBATCH --account=def-asmith-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=00:05:00
{ time mpiexec -n 1 ./sample1 ; } 2>out.time
```

The following modules were loaded:

```
[asmith@cedar5]$ module list
Currently Loaded Modules:
1) nixpkgs/16.09 (S) 5) intel/2016.4 (t)
2) icc/.2016.4.258 (H) 6) imkl/11.3.4.258 (math)
3) gcccore/.5.4.0 (H) 7) openmpi/2.1.1 (m)
4) ifort/.2016.4.258 (H) 8) StdEnv/2016.4 (S)
```

The job ran quickly, creating `myjob-123456.out` and `myjob-123456.err` files.  `myjob-123456.out` was empty, but `myjob-123456.err` contained this message:

```
[asmith@cedar5 scheduling]$ cat myjob-123456.err
slurmstepd: error: *** JOB 123456 ON cdr692 CANCELLED AT 2018-09-06T15:19:16 DUE TO TIME LIMIT ***
```

Can you help me fix this?


## Access to Your Account

If you want us to access, copy, or edit your files, or inspect your account and possibly make changes, explicitly state this in your email (unless you've already granted consent via CCDB). For example, instead of attaching files, indicate their location and give us written permission to access them.
