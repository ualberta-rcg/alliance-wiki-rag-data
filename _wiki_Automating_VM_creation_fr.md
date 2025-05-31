# Automating VM Creation

To automate the creation of cloud instances, volumes, etc., you can use the OpenStack CLI, Heat, Terraform, or the OpenStack Python API. OpenStack CLI and Terraform are command-line tools, while Heat is used via the OpenStack Horizon web dashboard. To install and configure settings and software in the instance, use `cloud-init`.

In addition to these tools, you can also access Compute Canada's software stack (CVMFS), which is available on our general-purpose clusters; see [Using CVMFS](#using-cvmfs) below.


## Using CVMFS

CVMFS is an HTTP file system that provides a scalable and reliable service for distributing research software. On the client side, users only need to mount CVMFS and use the software and libraries directly, without worrying about compiling or adapting the code. The software is pre-compiled for frequently used operating systems and can be loaded via modules (see [Using Modules](<add_link_here_if_available>)).

CVMFS is installed on the Cedar, Graham, and Beluga clusters; installation on a cloud is done by following [these instructions](<add_link_here>).

For more information, see our [wiki page Accessing CVMFS](<add_link_here>) and the [CERN documentation](<add_link_here>).


## Using cloud-init

`cloud-init` files are used to initialize a particular instance and are executed within that instance.  It's a way to automate tasks you would do on the command line when connected to your instance. These files can be used, for example, to update the operating system, install and configure applications, create files, execute commands, and create users and groups. `cloud-init` can also configure other tools like Ansible or Puppet.

The `cloud-init` configuration is specified in plain text in YAML format. To learn how to create files, see the [cloud-init documentation](<add_link_here>). These files can be used with Terraform, the CLI, the Python API, and the Horizon dashboard, the OpenStack web interface. We describe here the use of `cloud-init` with Horizon.


### Specifying the cloud-init file

Start an instance as usual via `Project->Compute->Instances`, by clicking on `Start Instance`. Configure the instance as described in the [Launching an Instance](<add_link_here>) section.

Before clicking on `Start`, select the `Post-creation` tab and enter a `cloud-init` YAML file in the `Customization script source` field, either by copying and pasting the file (direct entry method) or by uploading the file from your computer (file method). In older versions of OpenStack, and particularly IceHouse, the `cloud-init` file is copied into a text area. Return to the `Details` tab.

Once all fields are filled, click `Start` to create the instance.

The duration of the operation can be long since it depends on the content of the YAML file.


### Monitoring CloudInit

To follow the progress, examine the instance's console log.

In the `Instance Name` column, click on the instance to get information about it.

Under the `Log` tab, lines containing 'cloud-init' describe the phases.

When `cloud-init` stops, the following line is found towards the end of the log:

```
Cloud-init v. 0.7.5 finished at Wed, 22 Jun 2016 17:52:29 +0000. Datasource DataSourceOpenStack [net,ver=2].  Up 44.33 seconds
```

The log is refreshed by clicking the `Go` button at the top of the page.


## Using Heat Templates

Heat templates are even more powerful: they can be used to automate tasks done in the OpenStack dashboard, for example, creating multiple instances simultaneously, configuring security groups, creating and configuring networks, or creating volumes and attaching them to instances. Heat templates can be used with `cloud-init` files; once Heat has created an instance, it can pass a `cloud-init` file to that instance to perform configuration tasks or even dynamically include information about other resources in the `cloud-init` file (e.g., floating IPs of other instances).

We do not discuss here the creation of HOT files (Heat Orchestration Template); for this, consult the [official documentation](<add_link_here>). Heat can automate tasks performed from the OpenStack dashboard (Horizon) and pass to the included CloudInit files information about the IP addresses of other servers. Using a Heat template usually does not require the prior creation of resources. It is good practice to delete unused resources since Heat templates consume resources and will stop if the quota is exceeded.

To create a stack with a HOT file:

Select `Project->Orchestration->Stacks` and click `Launch Stack` to create a new stack.

In the `Select a template` window, you can enter a URL, a file name, or make a direct entry. We will use a HOT file from those listed in the [Available Configurations](#available-configurations) section below.

For `Template source`, select `URL`.

Paste the URL into the `Template URL` field.

Click `Next` to configure the parameters; these may vary depending on the template used, but by default, all stacks have:

*   **Stack Name**: Enter the chosen name.
*   **Stack creation timeout (minutes)**: Number of minutes allocated to stack creation; usually, the default value is sufficient.
*   **Password for user \[name]**: This password is required for subsequent modifications to the stack; it is rarely used since the stacks in the following section are not designed to be modified.

Click `Start` to create the stack.

For a progress image, click on the stack name and select the `Topology` tab. Gray nodes indicate that creation is in progress; green nodes indicate that the stack is created; red nodes indicate that creation has failed. Once the stack is created, click on the `Overview` tab for information about the stack, i.e., the URL to access a site or service.


<!-- Add a section for Available Configurations if needed -->
<!-- ### Available Configurations -->
<!--  Add content here -->

