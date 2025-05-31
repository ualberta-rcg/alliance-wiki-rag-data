# Portail

This article is a draft. This is not a complete article.  This is a draft, a work in progress intended for publication as an article. It may or may not be ready for inclusion in the main wiki and should not necessarily be considered factual or authoritative.


## Aperçu

The portal is a website for Alliance users. It uses information collected from computing nodes and management servers to interactively generate data allowing users to monitor their resource usage (CPU, GPU, memory, file system) in real time.

* **Béluga:** https://portail.beluga.calculquebec.ca
* **Narval:** https://portail.narval.calculquebec.ca


### Performance des systèmes de fichiers

Bandwidth and metadata operation graphs are displayed, with visualization options for the last week, last day, and last hour.


### Noeuds de connexions

Usage statistics for CPUs, memory, system load, and network are presented, with visualization options for the last week, last day, and last hour.


### Ordonnanceur

This tab presents statistics on allocated cluster cores and GPUs, with visualization options for the last week, last day, and last hour.


### Logiciels scientifiques

The most frequently used software with CPU cores and GPUs is presented in graph form.


### Noeuds de transfert de données

Bandwidth statistics for data transfer nodes are presented in this tab.


## Sommaire utilisateur

The user summary tab displays your quotas for different file systems, followed by your 10 most recent tasks. You can select a task by its number to access the detailed page.  Clicking "(Plus de détails)" redirects you to the "Statistiques des tâches" tab, where you can find all your tasks.


## Statistiques des tâches

The first block displays your current usage (CPU cores, memory, and GPUs). These statistics represent the average resources used by all currently running tasks. You can easily compare your allocated resources to your actual usage.

Next, you have access to a daily average, presented as a graph.

Following this is a representation of your file system activity. On the left, the graph shows the number of disk write commands you've performed (input/output operations per second (IOPS)). On the right, you see the amount of data transferred to the servers over a given period (bandwidth).

The next section presents all tasks you've launched, those currently running, and those pending.  In the upper left, you can filter tasks by status (OOM, completed, running, etc.). In the upper right, you can search by task number (Job ID) or name. Finally, in the lower right, an option allows you to quickly navigate between pages using multiple jumps.


### Page d'une tâche CPU

At the top, you'll find the task name, number, your username, and the status. Details of your submission script are displayed by clicking "Voir le script de la tâche". If the task was launched in interactive mode, the submission script will not be available.

The working directory and submission command are accessible by clicking "Voir la commande de soumission".

The next section is dedicated to scheduler information. You can access your CPU account monitoring page by clicking on your account number.

In the "Ressources" section, you can get an initial overview of your task's resource usage by comparing the "Alloués" and "Utilisés" columns for the different parameters listed.

The "CPU" graph allows you to visualize, over time, the CPU cores you requested. On the right, you can select/deselect different cores as needed. Note that for very short tasks, this graph is not available.

The "Mémoire" graph allows you to visualize, over time, the memory usage you requested.

The "Process and threads" graph allows you to observe different parameters related to processes and threads. Ideally, for a multithreaded task, the sum of "Running threads" and "Sleeping threads" should not exceed twice the number of cores requested. However, it's perfectly normal to have some processes in "sleeping" mode ("Sleeping threads") for certain types of programs (Java, Matlab, commercial software, or complex programs). You also have parameters showing the applications of the program executed over time.

The following graphs represent the file system usage for the current task, not the entire node. On the left, a representation of the number of input/output operations per second (IOPS) is displayed. On the right, the graph illustrates the data transfer rate between the task and the file system over time. This graph helps identify periods of intense activity or low file system usage.

For the complete node resource statistics, note that they may be imprecise if the node is shared among multiple users. The left graph shows the evolution of the bandwidth used by the task over time, in relation to software, licenses, etc. The right graph represents the evolution of the network bandwidth used by a task or set of tasks via the Infiniband network over time.  You can observe periods of massive data transfer (e.g., reading/writing to a file system (Lustre), MPI communication between nodes).

The left graph illustrates the evolution of the number of input/output operations per second (IOPS) performed on the local disk over time. The right graph shows the evolution of the bandwidth used on the local disk over time, i.e., the amount of data read or written per second.

Graphical representation of local disk space usage.

Graphical representation of power used.


### Page d'une tâche CPU (vecteur de tâches, job array)

The page for a CPU task in a job array is identical to that of a regular CPU task, except for the "Other jobs in the array" section. The table lists the other task numbers belonging to the same job array, along with information on their status, name, start time, and end time.


### Page d'une tâche GPU

At the top of the page, you have the task name, number, your username, and the status. Details of your submission script are displayed by clicking "Voir le script de la tâche". If you launched an interactive task, the submission script is not available.

The directory and submission command are accessible by clicking "Voir la commande de soumission".

The next section is reserved for scheduler information. You can access your GPU account page by clicking on your account number.

In the "Ressources" section, you can get a first overview of your task's resource usage by comparing the "Alloués" and "Utilisés" columns for the different parameters listed.

The "CPU" graph allows you to visualize the use of the requested CPU cores over time. On the right, you can select/deselect different cores as needed. Note that for very short tasks, this graph is not available.

The "Mémoire" graph allows you to visualize the memory usage you requested for the CPUs over time.

The "Process and threads" graph allows you to observe different parameters related to processes and threads.

The following graphs represent the file system usage for the current task, not the entire node. On the left, a representation of the number of input/output operations per second (IOPS) is displayed. On the right, the graph illustrates the data transfer rate between the task and the file system over time. This graph helps identify periods of intense activity or low file system usage.

The GPU graph represents your GPU usage. The "Streaming Multiprocessors" (SM) active parameter indicates the percentage of time the GPU spends executing a warp (a group of consecutive threads) in the last sampling window. This value should ideally be around 80%. For "SM occupancy" (defined as the ratio between the number of warps assigned to an SM and the maximum number of warps an SM can handle), a value around 50% is generally expected. Regarding the "Tensor" parameter, the value should be as high as possible. Ideally, your code should exploit this part of the GPU, optimized for multiplications and convolutions of multidimensional matrices. Finally, for floating-point operations (Floating Point) FP64, FP32, and FP16, you should observe significant activity on only one of these types, depending on the precision used by your code.

On the left, you have a graph showing the memory used by the GPU. On the right, a graph of GPU memory access cycles, representing the percentage of cycles during which the device's memory interface is active to send or receive data.

The GPU power graph displays the evolution of the GPU's power consumption (in watts) over time.

On the left, the GPU bandwidth on the PCIe bus (or PCI Express, for Peripheral Component Interconnect Express). On the right, GPU bandwidth on the NVLink bus. The NVLink bus is a technology developed by NVIDIA to allow ultra-fast communication between multiple GPUs.

For the complete node resource statistics, note that they may be imprecise if the node is shared among multiple users. The left graph shows the evolution of the bandwidth used by the task over time, in relation to software, licenses, etc. The right graph represents the evolution of the network bandwidth used by a task or set of tasks via the Infiniband network over time. You can observe periods of massive data transfer (e.g., reading/writing to a file system (Lustre), MPI communication between nodes).

The left graph illustrates the evolution of the number of input/output operations per second (IOPS) performed on the local disk over time. The right graph shows the evolution of the bandwidth used on the local disk over time, i.e., the amount of data read or written per second.

Graphical representation of local disk space usage.

Graphical representation of power used.


## Statistiques d'un compte

The "Statistique d'un compte" section groups your group's usage into two subsections: CPU and GPU.


### Statistiques d'un compte CPU

You will find the sum of your group's requests for CPU cores, as well as their corresponding usage over the past months. You can also track the evolution of your priority, which varies depending on your usage.

This graph shows the most commonly used applications.

You can view the resource usage of each user in your group here.

This graph shows the evolution over time of wasted CPU cores by each user in the group.

You can view the memory usage of each user in your group here.

This graph represents the wasted memory by each user.

Next, you have a representation of your activity on the file systems. On the left, the graph shows the number of disk write commands you have performed (input/output operations per second (IOPS)). On the right, you see the amount of data transferred to the servers over a given period (bandwidth).

You have a list of the latest tasks that have been performed for the entire group.


### Statistiques d'un compte GPU

Here you find the sum of your group's GPU requests, as well as the corresponding usage over the past months. You can also track the evolution of your priority, which varies depending on your usage.

This graph represents the most commonly used applications.

You can view the resource usage of each user in your group here.

The following graph represents, over time, the amount of wasted GPUs per user.

Next, you have the CPU cores allocated and used in your GPU tasks.

This figure illustrates the waste of CPUs in the context of your GPU tasks.

You can visualize the memory usage for each user in your group here.

This graph illustrates the wasted memory by each user.

Next, you have a representation of your activity on the file systems. On the left, the graph shows the number of disk write commands you have performed (input/output operations per second (IOPS)). On the right, you see the amount of data transferred to the servers over a given period (bandwidth).

Here is the list of the latest tasks performed at the group level.


## Statistiques du cloud

The first table, "Vos instances," presents all virtual machines associated with an account. The "Saveur" column refers to the type of virtual machine. The "UUID" column corresponds to a unique identifier assigned to each virtual machine.

Then, each virtual machine has its own usage statistics (CPU cores, Memory, Disk bandwidth, Disk IOPS, and Network bandwidth) displayable for the last month, last week, last day, or last hour.


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Portail&oldid=177508](https://docs.alliancecan.ca/mediawiki/index.php?title=Portail&oldid=177508)"
