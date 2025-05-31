# OpenStack Command Line Clients

This page is a translated version of the page [OpenStack command line clients](https://docs.alliancecan.ca/mediawiki/index.php?title=OpenStack_command_line_clients&oldid=162518) and the translation is 100% complete.

Other languages: [English](https://docs.alliancecan.ca/mediawiki/index.php?title=OpenStack_command_line_clients&oldid=162518), français

## OpenStackClient

`OpenStackClient` allows you to use several functions of the OpenStack dashboard, as well as other functions that are not available through the graphical interface. To use it on any type of machine, virtual or otherwise, simply install the client and have an internet connection. The examples on this page are for Linux.

## Contents

1. Installation
2. Connecting the command line client to OpenStack
3. Running commands
4. Command groups
    * 4.1 Server commands
    * 4.2 Volume commands
    * 4.3 Console commands
    * 4.4 Flavor commands
    * 4.5 Image commands
    * 4.6 IP commands
    * 4.7 Keypair commands
    * 4.8 Network commands
    * 4.9 Snapshot commands
    * 4.10 Security group commands
    * 4.11 Limits commands
5. Other interfaces


## Installation

The OpenStack command-line tools are for Python and work on a personal computer or a cloud instance. Different Linux distributions may offer pre-compiled packages; for details, see the [installation documentation](link_to_installation_documentation_needed). If you have administrator permissions, you can quickly install Python and the OpenStack command-line tools.

**Ubuntu:**

```bash
sudo apt-get install python python-dev python-pip
sudo pip install python-openstackclient
```

**CentOS 7:**

(Run as root)

```bash
yum install epel-release
yum install gcc python python-dev python2-pip
pip install python-openstackclient
```

**Fedora:**

```bash
sudo dnf install python-openstackclient
```

**Note:** If you do not have administrator permissions, you must install Python and `pip` differently. Once the installation is complete, you can install the command-line tools in your home space as follows:

```bash
pip install --user python-openstackclient
```

The installation destination is probably included in the `$PATH`; however, you can check if `~/.bashrc` or `~/.bash_profile` includes the line:

```bash
PATH=$PATH:$HOME/.local/bin:$HOME/bin
```

**SDK:** To explore the [Python APIs](link_to_python_apis_needed), add:

```bash
export PYTHONPATH=${HOME}/.local/lib/python2.7/site-packages/:${PYTHONPATH}
```

and modify `python2.7` according to the installed Python version.


## Connecting the command line client to OpenStack

You must tell the client where to find the OpenStack project in our cloud environment. The easiest way is to download a configuration file via the OpenStack dashboard, as follows:

Project -> API Access -> Download OpenStack RC file.

Then run the command:

```bash
[nom@serveur ~]$ source <project name>-openrc.sh
```

When you need to enter the OpenStack password, enter your password for our CCDB database. To test the configuration, enter:

```bash
[nom@serveur ~]$ openstack image list
```

If you use multiple RC files, be aware of environment variables that remain from the last RC file used, as they may prevent the execution of OpenStack client commands. You can work around this problem in two ways: by destroying the variables with `unset <variable-name>` or by starting a new session without defined variables.


## Running commands

The command-line client can be used interactively by entering:

```bash
[nom@serveur ~]$ openstack
```

Then enter the commands at the prompt. Each command can be entered individually by preceding it with `openstack`, for example:

```bash
[nom@serveur ~]$ openstack server list
```

In interactive mode, display the list of available commands by entering `help` at the OpenStack prompt. The available commands are grouped; the most common are presented below. To get the list of commands belonging to a particular group, enter `help <command group>`. To get the options and arguments related to a command, enter `help <command group> <command>`. Note that several commands are only available to users with administrator permissions, and otherwise an error message will be displayed. The following commands are available to all users.


## Command groups

### Server commands

`add security group`, `migrate`, `resume`, `unlock`, `add volume`, `pause`, `set`, `unpause`, `create`, `reboot`, `shelve`, `unrescue`, `delete`, `rebuild`, `show`, `unset`, `dump create`, `remove security group`, `ssh`, `unshelve`, `image create`, `remove volume`, `start`, `list`, `rescue`, `stop`, `lock`, `resize`, `suspend`

### Volume commands

`create`, `set`, `delete`, `show`, `list`, `unset`

### Console commands

`log show`, `url show`

### Flavor commands

`list`, `show`

### Image commands

`create`, `save`, `delete`, `set`, `list`, `show`

### IP commands

`fixed add`, `floating list`, `fixed remove`, `floating pool list`, `floating add`, `floating remove`, `floating create`, `floating show`, `floating delete`

### Keypair commands

`create`, `list`, `delete`, `show`

### Network commands

`create`, `set`, `delete`, `show`, `list`

### Snapshot commands

`create`, `set`, `delete`, `show`, `list`, `unset`

### Security group commands

`create`, `rule list`, `delete`, `rule show`, `list`, `set`, `rule create`, `show`, `rule delete`

### Limits commands

`show`


## Other interfaces

In addition to the `openstack` command (described above) which incorporates most of the features into a single command, there are also separate commands for the various OpenStack components that add other features. These commands are installed at the same time as the `openstack` command and no other installation is necessary. These commands are:

* `nova` for working with servers;
* `glance` for working with images;
* `cinder` for working with volumes;
* `heat` for working with orchestration.

**(Retrieved from "https://docs.alliancecan.ca/mediawiki/index.php?title=OpenStack_command_line_clients/fr&oldid=162521")**
