# Managing Slurm Accounts

Each task submitted to the Slurm task scheduler is associated with a scientific allocation project (RAP for Resource Allocation Project). The RAP is selected with the `--account` option of `sbatch`.

The task's priority will be determined by the target share of the account compared to recent usage, as described in the [Task Scheduling Policy](link-to-policy-here).  (Note:  Replace `link-to-policy-here` with the actual link).

Several members of a research group can use the same RAP account to submit tasks. Resource usage is billed to the same account; the usage made by each member of the group thus affects the priority of all members' tasks. It may therefore be advantageous to coordinate task submission to improve project performance.


## Why Manage Usage in a RAP Account?

When a task is submitted by a user who has not already used many resources, it may happen that the priority of this task is very low because other members of the group have recently done a lot of work. In such a case, the tasks of all users of the RAP account will have to wait for the fair share (LevelFS) of the account to drop to a competitive value. Since the fair share principle applies both *within* a group and *between* groups, the tasks of users who have consumed fewer resources will have priority once the account's fair share is restored.

This may not occur if different users submit tasks with radically different requirements. If a user runs several small tasks that can slip into scheduling holes (backfilling, cycle scavenging), the execution speed might be appreciable, even if the priority for the group's tasks remains low. Other users of the RAP account will have difficulty running more resource-intensive tasks.


## Strategies

We invite you to discuss the following strategies during your work meetings:

* **Different Deadlines:** If several users have different deadlines for projects requiring burst tasks, it is advantageous to schedule these tasks to avoid their priorities interfering at critical times.

* **Different Clusters:** Our general-purpose clusters have almost identical capacity, and usage by each RAP account is counted separately on each. User X of group Y on Graham has no effect on the priority of tasks submitted by user Z of account Y on Cedar.

* **Multiple Accounts:** Tasks can use resources allocated through competition and resources offered by default. Tasks executed in one account do not affect the fair share of the other account.  In a collaborative context between several groups, each principal investigator can obtain their own account, and users can be distributed across different RAPs for greater efficiency.

* **Contact Support:** If these strategies are not effective, contact [technical support](link-to-support-here) and ask the analyst to consult the internal documentation entitled "A group in conflict with itself". (Note: Replace `link-to-support-here` with the actual link).


**(Retrieved from "https://docs.alliancecan.ca/mediawiki/index.php?title=Managing_Slurm_accounts/fr&oldid=153769")**
