# Automating VM Creation

To automate the creation of cloud VMs, volumes, etc., you can use the OpenStack CLI, Heat, Terraform, or the OpenStack Python API.  Both the OpenStack CLI and Terraform are command-line tools. Heat is used through the OpenStack web dashboard, Horizon. To install and configure settings and software within the VM, use cloud-init.

In addition to these tools for creating and provisioning your VMs, you can also access the Compute Canada software stack (CVMFS) available on our general-purpose computing clusters within your VM. See the [Enabling CVMFS](#enabling-cvmfs-on-your-vm) section below.


## Enabling CVMFS on your VM

CVMFS is an HTTP-based file system providing a scalable, reliable, and low-maintenance research software distribution service.  At the client end, users only need to mount CVMFS and then use the software or libraries directly without worrying about compiling, building, or patching. All software is pre-compiled for common OS flavors and even modularized so users can simply load software as a module.

CVMFS is already installed on Compute Canada cluster systems such as Cedar, Graham, and Beluga. On cloud systems, users need to enable it manually, following these [cloud instructions](link-to-cloud-instructions).

For more information, see the [Compute Canada CVMFS documentation](link-to-compute-canada-cvmfs-docs) and [CERN CVMFS documentation](link-to-cern-cvmfs-docs).


## Using cloud-init

Cloud-init files initialize a particular VM and run within it. They automate tasks you would perform at the command line while logged into your VM.  They can update the operating system, install and configure applications, create files, run commands, and create users and groups. Cloud-init can be used to set up other provisioning tools such as Ansible or Puppet to continue with software and VM configuration if desired.

Cloud-init configuration is specified using plain text in the YAML format. For creating cloud-init files, see the official [cloud-init documentation](link-to-cloud-init-docs). Cloud-init files can be used with the Horizon dashboard (OpenStack's web GUI), Terraform, the CLI, or the Python API.  This section describes using a cloud-init file with Horizon.


### Specifying a cloud-init File

Start as normal when launching an instance by clicking under *Project* -> *Compute* -> *Instances* and specifying your VM's configuration as described in [Launching a VM](link-to-launching-vm-docs).

Before clicking *Launch*, select the *Post-Creation* tab and specify your *Customization Script Source*, in this case a Cloud-init YAML file. You can either copy and paste into a text box (*Direct Input* method) or upload from a file on your desktop computer (*File* method). Older versions of OpenStack, particularly IceHouse, only provide a text box to copy and paste your CloudInit file into.

Once the usual selections for your VM (as described in [Launching a VM](link-to-launching-vm-docs)) have been made and the Cloud-init YAML file is included, click *Launch* to create the VM.  It may take some time for CloudInit to complete, depending on the Cloud-init YAML file's contents.


### Checking Cloud-init Progress

To check Cloud-init's progress on a VM, check the VM's console log by:

1. Selecting *Project* -> *Compute* -> *Instances* in the left-hand menu.
2. Clicking the *Instance Name* of the VM. This provides more information about the VM.
3. Selecting the *Log* tab and looking for lines containing 'cloud-init' for information about the various phases of CloudInit.

When Cloud-init finishes, the following line appears near or at the end of the log:

```
Cloud-init v. 0.7.5 finished at Wed, 22 Jun 2016 17:52:29 +0000. Datasource DataSourceOpenStack [net,ver=2].  Up 44.33 seconds
```

The log must be refreshed manually by clicking the *Go* button.


## Using Heat Templates

Heat templates are more powerful; they automate tasks performed in the OpenStack dashboard, such as creating multiple VMs at once, configuring security groups, creating and configuring networks, and creating and attaching volumes to VMs. Heat templates can be used with cloud-init files. Once Heat creates the VM, it can pass a cloud-init file to that VM to perform setup tasks and even include information about other resources dynamically in the cloud-init files (e.g., floating IPs of other VMs).

Creating Heat Orchestration Template (HOT) files is not covered here; see the official [documentation](link-to-heat-docs). HOT files are also written in the YAML format. Heat automates operations performed in the OpenStack dashboard (Horizon) and allows passing information into embedded CloudInit files, such as another server's IP.  Before using a Heat template, there's usually no need to create resources in advance. In fact, it's often good practice to remove any unused resources beforehand, as using a Heat template consumes resources towards your quota and will fail if it tries to exceed your quota.


To create a stack using a HOT file:

1. Select *Project* -> *Orchestration* -> *Stacks* and click the *Launch Stack* button.
2. Provide a HOT file by entering the URL, filename, or using *Direct Input*.  This example uses a HOT file from one of the links in the [Available Setups](link-to-available-setups) section below.
3. In the *Template Source* box, select *URL* from the drop-down list.
4. Paste the selected URL into the *Template URL* box.
5. Click *Next* to begin setting stack parameters; these vary depending on the template, but all stacks have the following parameters by default:
    *   *Stack Name*: Choose a meaningful name.
    *   *Creation Timeout*: Indicates how long after stack creation before OpenStack gives up trying to create the stack if it hasn't finished; the default value is usually sufficient.
    *   *Password for user*: Sets the password required for later stack changes. This is seldom used as many of the stacks mentioned in the next section are not designed to be updated.
6. Click *Launch* to begin creating your stack.

To graphically see the progress of your stack creation, click the *Stack Name* and select the *Topology* tab. Gray nodes indicate creation is in progress, green nodes have finished, and red nodes indicate failures. Once the stack completes successfully, click the *Overview* tab to see any information the stack may provide (e.g., a URL to access a service or website).


**(Remember to replace the bracketed placeholders like `[link-to-cloud-instructions]` with actual links.)**
