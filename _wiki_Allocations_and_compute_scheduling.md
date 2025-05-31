# Allocations and Compute Scheduling

## Allocations for High-Performance Computing

An allocation is a resource amount a research group can use for a period (usually a year).  This amount is either a maximum (like storage) or an average usage over the period (like shared compute cores). Allocations are usually in core-years, GPU-years, or storage space. Storage allocations are straightforward: research groups get a maximum amount of exclusive storage. Core-year and GPU-year allocations are more complex because they represent average usage over the allocation period (typically a year) across shared resources.

The allocation period's length is a reference value for calculating the average applied to the actual resource availability period.  If the allocation period is a year and clusters are down for a week, the research group doesn't get an extra week.  Similarly, extending the allocation period doesn't reduce resource access.

For core-year and GPU-year allocations (targeting average usage on shared resources), even usage throughout the allocation period is more likely to meet or exceed targets than burst usage or delayed usage.


### From Compute Allocations to Job Scheduling

Compute resources (granted by core-year and GPU-year allocations) require research groups to submit *jobs* to a *scheduler*. A job combines a computer program (application) and a resource list the application needs. The scheduler prioritizes jobs based on their priority and available resources.

The scheduler uses prioritization algorithms to meet all groups' allocation targets. It considers a research group's recent system usage compared to its allocated usage.  Recent usage (or lack thereof) is weighted more heavily. This allows groups matching actual and allocated usage to operate consistently at that level, smoothing resource usage over time and theoretically enabling all groups to meet their allocation targets.


### Consequences of Overusing a CPU or GPU Allocation

If jobs are waiting and competing demand is low, the scheduler might allow more jobs to run than your target level. The only consequence is that your subsequent jobs might have lower priority temporarily while the scheduler prioritizes other groups below their target. You aren't prevented from submitting or running new jobs, and your average usage should remain close to your target (allocation).

You might even finish a month or year having run more work than your allocation allows, though this is unlikely given resource demand.


## Reference GPU Units (RGUs)

GPU performance has dramatically increased and continues to do so. Until RAC 2023, all GPUs were treated as equivalent for allocation, causing problems in allocation and job running.  In RAC 2024, the *reference GPU unit* (RGU) was introduced to rank GPU models and alleviate these problems. RAC 2025 will address complexities involving multi-instance GPU technology.

Because roughly half our users primarily use single-precision floating-point operations (FP32), the other half use half-precision floating-point operations (FP16), and a significant portion are constrained by GPU memory, we use these evaluation criteria and weights to rank GPU models:

| Evaluation Criterion                                      | Weight |
|----------------------------------------------------------|--------|
| FP32 score (with dense matrices on regular GPU cores)    | 40%    |
| FP16 score (with dense matrices on Tensor cores)         | 40%    |
| GPU memory score                                        | 20%    |

The NVIDIA A100-40gb GPU is the reference model (RGU value of 4.0 for historical reasons). Its FP16 performance, FP32 performance, and memory size are each defined as 1.0. Multiplying the percentages by 4.0 yields these coefficients and RGU values for other models:

### RGU Scores for Whole GPU Models

| FP32 score | FP16 score | Memory score | Combined score | Available | Allocatable | RAC 2025 |
|-------------|-------------|---------------|-----------------|------------|-------------|-----------|
| Coefficient: | 1.6         | 1.6           | 0.8             | (RGU)      | Now         |           |
| H100-80gb   | 3.44        | 3.17          | 2.0             | 12.2        | No          | Yes       |
| A100-80gb   | 1.00        | 1.00          | 2.0             | 4.8         | ?           | No        |
| A100-40gb   | 1.00        | 1.00          | 1.0             | 4.0         | Yes         | Yes       |
| V100-32gb   | 0.81        | 0.40          | 0.8             | 2.6         | ?           | No        |
| V100-16gb   | 0.81        | 0.40          | 0.4             | 2.2         | ?           | No        |
| T4-16gb     | 0.42        | 0.21          | 0.4             | 1.3         | ?           | No        |
| P100-16gb   | 0.48        | 0.03          | 0.4             | 1.1         | No          | No        |
| P100-12gb   | 0.48        | 0.03          | 0.3             | 1.0         | No          | No        |


With the 2025 infrastructure renewal, scheduling GPU fractions using *multi-instance GPU* technology will be possible.  Different jobs (potentially from different users) can run on the same GPU simultaneously.  Following NVIDIA's terminology, a GPU fraction allocated to a single job is a *GPU instance* (or *MIG instance*).

### GPU Models and Instances Available for RAC 2025

| Model or instance | Fraction of GPU | RGU |
|--------------------|-----------------|-----|
| A100-40gb         | Whole GPU ⇒ 100% | 4.0 |
| A100-3g.20gb      | max(3g/7g, 20GB/40GB) ⇒ 50% | 2.0 |
| A100-4g.20gb      | max(4g/7g, 20GB/40GB) ⇒ 57% | 2.3 |
| H100-80gb         | Whole GPU ⇒ 100% | 12.2|
| H100-1g.10gb      | max(1g/7g, 40GB/80GB) ⇒ 14% | 1.7 |
| H100-2g.20gb      | max(2g/7g, 40GB/80GB) ⇒ 28% | 3.5 |
| H100-3g.40gb      | max(3g/7g, 40GB/80GB) ⇒ 50% | 6.1 |
| H100-4g.40gb      | max(4g/7g, 40GB/80GB) ⇒ 57% | 7.0 |

Note: a 1g GPU instance is 1/7 of an A100 or H100 GPU.  The 3g case considers the extra memory per g.


### Choosing GPU Models for Your Project

The relative scores above should guide model selection.  Here's an example with extremes:

*   If your applications primarily use FP32 operations, an A100-40gb GPU is expected to be twice as fast as a P100-12gb GPU, but recorded usage will be 4 times the resources.  For equal RGUs, P100-12gb GPUs should allow double the computations.
*   If your applications (typically AI-related) primarily use FP16 operations (including mixed precision or other floating-point formats), using an A100-40gb will be evaluated as using 4x the resources of a P100-12gb, but it can compute ~30x the calculations in the same time, allowing ~7.5x the computations.


### RAC Awards Hold RGU Values Constant

During the Resource Allocation Competition (RAC), GPU proposals must specify the preferred GPU model.  The CCDB form automatically calculates the reference GPU units (RGUs) from the requested gpu-years per project year.

For example, requesting 13 gpu-years of the A100-40gb model on the `narval-gpu` resource results in 13 \* 4.0 = 52 RGUs. The RAC committee allocates up to 52 RGUs based on the proposal score. If the allocation must move to a different cluster, the committee allocates gpu-years to maintain the same RGU amount.


## Detailed Effect of Resource Usage on Priority

The priority calculation principle is that compute-based jobs are considered based on resources prevented from use by others, not resources actually used.

A common example is a job requesting multiple cores but using fewer.  The priority-affecting usage is the number of cores requested, not used, because the unused cores were unavailable to others.

Another case is a job requesting memory beyond the associated cores.  If a cluster has 4GB of memory per core and a job requests one core but 8GB of memory, it's deemed to have used two cores because the second core was unavailable due to lack of memory.


### Cores Equivalent Used by the Scheduler

A core equivalent is a bundle of a single core and an associated memory amount.  It's a core plus the memory amount considered associated with each core on a given system.

Cedar and Graham provide 4GB per core (based on the most common node type), making a core equivalent a 4GB-per-core core-memory bundle. Niagara provides 4.8GB per core, making its core equivalent a 4.8GB-per-core core-memory bundle. Jobs are charged based on core equivalent usage at 4 or 4.8 GB per core.

Allocation target tracking is straightforward when resource requests are in whole core equivalents. It becomes more complex when jobs request portions of a core equivalent, as many points can count against a research group's allocation even when using only portions of core equivalents. The Alliance's usage accounting method addresses fairness issues but isn't initially intuitive.

Research groups are charged for the maximum number of core equivalents they take.  Assuming a core equivalent of 1 core and 4GB of memory:

*   Research groups using more cores than memory (above the 1 core/4GB ratio) are charged by cores.  For example, two cores and 2GB per core (4GB total) is counted as 2 core equivalents.
*   Research groups using more memory than the 1 core/4GB ratio are charged by memory. For example, two cores and 5GB per core (10GB total) is counted as 2.5 core equivalents.


### Reference GPU Unit Equivalent Used by the Scheduler

GPU usage follows the same principles as core equivalents, adding an RGU to the bundle with cores and memory. GPU-based allocation target accounting must include the RGU.  A point system expresses RGU equivalence.

Research groups are charged for the maximum number of RGU-core-memory bundles they use.  Assuming a fictive bundle of 1 RGU, 3 cores, and 4GB of memory:

*   Research groups using more RGUs than cores or memory per bundle are charged by RGU.
*   Research groups using more cores than RGUs or memory per bundle are charged by core.
*   Research groups using more memory than RGUs or cores per bundle are charged by memory.


On the same fictive cluster:

*   A bundle with one V100-32gb GPU, 7.8 CPU cores, and 10.4 GB of memory is worth 2.6 RGU equivalents.
*   A bundle with one A100-40gb GPU, 12 CPU cores, and 16 GB of memory is worth 4.0 RGU equivalents.


### Ratios in Bundles

Alliance systems have these RGU-core-memory and GPU-core-memory bundle characteristics:

| Cluster   | Model or instance | RGU per GPU | Bundle per RGU           | Bundle per GPU              |
|-----------|--------------------|-------------|--------------------------|-----------------------------|
| Béluga    | V100-16gb          | 2.2          | 4.5 cores, 21 GB          | 10 cores, 46.5 GB           |
| Cedar     | P100-12gb          | 1.0          | 3.1 cores, 25 GB          | 3.1 cores, 25 GB           |
|           | P100-16gb          | 1.1          | 3.4 cores, 27 GB          |                            |
|           | V100-32gb          | 2.6          | 8.0 cores, 65 GB          |                            |
| Fir       | H100-80gb          | 12.2         | 0.98 core, 20.5 GB        | 12 cores, 250 GB            |
|           | H100-1g.10gb       | 1.7          | 1.6 cores, 34.8 GB        |                            |
|           | H100-2g.20gb       | 3.5          | 3.4 cores, 71.7 GB        |                            |
|           | H100-3g.40gb       | 6.1          | 6 cores, 125 GB           |                            |
|           | H100-4g.40gb       | 7.0          | 6.9 cores, 143 GB         |                            |
| Graham    | P100-12gb          | 1.0          | 9.7 cores, 43 GB          | 9.7 cores, 43 GB           |
|           | T4-16gb            | 1.3          | 12.6 cores, 56 GB         |                            |
|           | V100-16gb          | 2.2          | 21.3 cores, 95 GB         |                            |
|           | V100-32gb          | 2.6          | 25.2 cores, 112 GB        |                            |
|           | A100-80gb          | 4.8          | 46.6 cores, 206 GB        |                            |
| Nibi      | H100-80gb          | 12.2         | 1.3 cores, 15.3 GB        | 12 cores, 187 GB            |
|           | H100-1g.10gb       | 1.7          | 1.6 cores, 26 GB         |                            |
|           | H100-2g.20gb       | 3.5          | 3.4 cores, 53.5 GB        |                            |
|           | H100-3g.40gb       | 6.1          | 6 cores, 93.5 GB          |                            |
|           | H100-4g.40gb       | 7.0          | 6.9 cores, 107 GB         |                            |
| Narval    | A100-40gb         | 4.0          | 3.0 cores, 31 GB          | 12 cores, 124.5 GB          |
|           | A100-3g.20gb      | 2.0          | 6 cores, 62.3 GB          |                            |
|           | A100-4g.20gb      | 2.3          | 6.9 cores, 71.5 GB         |                            |
| Rorqual   | H100-80gb          | 12.2         | 1.3 cores, 10.2 GB        | 16 cores, 124.5 GB          |
|           | H100-1g.10gb       | 1.7          | 2.2 cores, 17.4 GB        |                            |
|           | H100-2g.20gb       | 3.5          | 4.5 cores, 35.8 GB        |                            |
|           | H100-3g.40gb       | 6.1          | 8 cores, 62.3 GB          |                            |
|           | H100-4g.40gb       | 7.0          | 9.1 cores, 71.5 GB         |                            |
| Trillium  | H100-80gb          | 12.2         | 1.97 cores, 15.4 GB       | 24 cores, 188 GB            |

(*) All GPU resources on this cluster are not allocatable through the RAC process.

Note: While the scheduler computes priority based on usage calculated with the above bundles, users requesting multiple GPUs per node must also consider physical ratios.


## Viewing Resource Usage in the Portal

`portal.alliancecan.ca/slurm` provides an interface for exploring time-series data about jobs on national clusters.  The page contains a figure displaying several usage metrics.  Upon login, it displays CPU days on the Cedar cluster for all accessible project accounts. If there's no usage on Cedar, it shows "No Data or usage too small to have a meaningful plot".  The data is modifiable via control panels on the left margin:

*   **Select system and dates:** Selects the cluster and time interval.
*   **Parameters:**  Selects metrics, summation type, and whether to include running jobs.
*   **SLURM account:** Selects the SLURM account to display.


### Displaying a Specified Account

If you have access to multiple Slurm accounts, the "Select user's account" pull-down menu lets you select the project account to display.  If left empty, it displays all your usage across accounts on the specified cluster during the selected time period. The menu is populated with accounts having job records on the selected cluster during the selected time interval.  Other accessible accounts without usage are grayed out. Selecting a single project account updates the figure and populates the "Allocation Information" summary panel.  Bar height corresponds to the daily metric (e.g., CPU-equivalent days) across all users in the account. The top seven users are displayed in unique colors, stacked on top of the summed metric for all other users (gray).  Plotly tools (zoom, pan, etc.) are available via icons in the top-right when hovering over the figure window.  The legend on the right manipulates the figure: single-clicking toggles an item's presence, double-clicking toggles all other items.


### Options in the Figure Legend

The legend provides display options.  Additional variables can be enabled or disabled.  Beyond user color affiliation, it provides access to displaying:

*   **SLURM Raw Usage:** Obtained from polling `sshare` for each account.
*   **SLURM Raw Shares:** Obtained from polling `sshare` for each account.
*   **CCDB allocation:** The CCDB account profile representation of SLURM Raw Shares.
*   **Queued jobs:** Resources in pending jobs, represented by narrow gray bars.
*   **Total:** Daily total metric across users.

Single-clicking toggles a specific item; double-clicking toggles all other items.


### Displaying the Allocation Target and Queued Resources

When a single account is selected, SLURM Raw Shares are shown as a horizontal red line.  This can be toggled via "Display allocation target by default" in the Parameters panel or by clicking "SLURM Raw Shares" in the legend.  The Queued jobs metric (sum of resources in pending jobs) can be toggled by clicking "Queued jobs" in the legend.


### Mouse Hover Over the Figure Window

Native Plotly interactive figure options (download, zoom, pan, box select, lasso select, zoom in/out, autoscale, reset axes) are available in the top-right when hovering over the figure.  Hovering over bars shows the user name, day, and usage quantity for that specific user (not the daily sum across users).


### Default SLURM Raw Shares and the SLURM Raw Usage

The SLURM Raw Shares of an `rrg-*` or `rpp-*` account is a straight line corresponding to the account's resource allocation. For default accounts, SLURM Raw Shares are dynamic based on the number of active accounts.  Plotting SLURM Raw Shares for a default account easily determines the expected usage achievable on a given cluster.

SLURM Raw Usage is a metric the scheduling software uses to determine account priority. It's the cumulative sum of account usage in billing units plus a half-life decay period.  Plotting it assesses how past usage influences priority.  A good rule of thumb is that if SLURM Raw Usage is 10 times SLURM Raw Shares, the account's usage is at its target share (the usage rate the scheduler tries to maintain).


### Selecting a Specific Cluster and Time Interval

The figure shows usage for a single cluster over a specified time interval. The System pull-down menu lists active national clusters using Slurm.  The "Start date (incl.)" and "End date (incl.)" fields change the displayed time interval.  It includes all jobs in a running (R) or pending (PD) state during the interval, including both the start and end dates. Selecting a future end date projects currently running and pending jobs for their requested duration.


### Displaying Usage Over an Extended Time Period into the Future

Selecting a future end time overlays a transparent red area labeled "Projection".  In this projection, each job is assumed to run to its requested time limit. For queued resources, the projection assumes each pending job starts immediately and runs until its requested time limit. This isn't a forecast of actual future events.


### Metrics, Summation, and Running Jobs

The Metric pull-down control selects from CPU, CPU-equivalent, RGU, RGU-equivalent, Memory, Billing, gpu, and all specific GPU models on the selected cluster.

The Summation pull-down switches between daily Total and Running total.  Total shows daily usage; Running total shows the sum of that day's usage and all previous days.  The Allocation Target is similarly adjusted.

Setting "Include Running jobs" to No shows only completed job data; Yes includes running jobs.


### Display of the Running Total of Account Usage

When displaying the running total of usage for a single account with the Allocation target, the usage histogram shows how an account deviates from its target share. Values are the cumulative sum across days from the "total" summation view for both usage and allocation target.  When an account submits jobs requesting more than its target share, the usage cumulative sum should oscillate above and below the target share cumulative sum if the scheduler manages fair share properly.  Because the scheduler uses a decay period for past usage, a good interval to inspect scheduler performance is the past 30 days.


## Viewing Resource Usage in CCDB

Information on compute resource usage is found by logging into the CCDB and navigating to My Account > View Group Usage.

CPU and GPU core-year values are calculated based on resources allocated to jobs.  The summarized values don't represent core-equivalent measures; for large memory jobs, usage values won't match the cluster scheduler's representation.

The first tab bar offers these options:

*   **By Compute Resource:** Cluster where jobs are submitted.
*   **By Resource Allocation Project:** Projects where jobs are submitted.
*   **By Submitter:** User submitting the jobs.

Storage usage is discussed in Storage and file management.


### Usage by Compute Resource

This view shows compute resource usage per cluster for groups you own or are a member of for the current allocation year (starting April 1st).  Tables contain total usage to date and projected usage to the end of the allocation period.  From the Extra Info column, "Show monthly usage" displays a monthly breakdown; "Show submitter usage" displays a similar breakdown for users submitting jobs on the cluster.


### Usage by Resource Allocation Project

This tab displays RAPIs (Resource Allocation Project Identifiers) for the selected allocation year.  Tables contain detailed information for each allocation project and resources used on all clusters.  The top summarizes account name (e.g., def-, rrg-, rpp-*), project title and ownership, and allocation and usage summaries.


### GPU Usage and Reference GPU Units (RGUs)

For projects with GPU usage, the table breaks down usage on various GPU models measured in RGUs.


### Usage by Submitter

Usage can be displayed grouped by users submitting jobs from within resource allocation projects (group accounts).  The view shows usage for each user aggregated across systems.  Selecting a user displays their usage broken down by cluster.  These user summaries can be broken down monthly via the "Show monthly usage" link in the Extra Info column.
