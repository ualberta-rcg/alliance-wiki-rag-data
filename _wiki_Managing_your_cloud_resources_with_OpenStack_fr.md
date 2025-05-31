# Managing Your Cloud Resources with OpenStack

This page is a translated version of the page [Managing your cloud resources with OpenStack](https://docs.alliancecan.ca/mediawiki/index.php?title=Managing_your_cloud_resources_with_OpenStack&oldid=140323) and the translation is 100% complete.

Other languages:

* [English](https://docs.alliancecan.ca/mediawiki/index.php?title=Managing_your_cloud_resources_with_OpenStack&oldid=140323)
* français

## Cloud Computing Service

Our cloud computing service uses the OpenStack software suite to control resources such as computers, storage space, and networking hardware. OpenStack allows you to create and manage instances (or VMs for virtual machines) that function as separate machines through emulation. You have complete control over the development environment, from choosing the operating system to installing and configuring the software. OpenStack serves various uses, including web hosting and the creation of virtual clusters. You can find additional information on the [OpenStack website](https://www.openstack.org/).

We address several aspects of working with OpenStack here, assuming you have read the [Cloud: Getting Started Guide](link_to_getting_started_guide) and are familiar with the basic operations for launching an instance and connecting to it. You can work from the OpenStack dashboard (see screenshots below), a command-line client, or using a tool like Terraform.

However, some tasks can only be performed via the command line, for example, sharing an image with another project.

## Contents

1. Dashboard
2. Projects
3. Volumes
4. Images
5. Instances
6. Availability Zones
7. Security Groups
    * Default Security Group
    * Managing Security Groups
    * CIDR Rules
8. cloudInit
    * Adding Users with cloud-init During Instance Creation

## Dashboard

In our documentation, the dashboard is the name given to the web interface that manages your cloud resources. The dashboard is an OpenStack sub-project sometimes called Horizon. You can consult the [documentation describing how Horizon works](link_to_horizon_documentation).

## Projects

OpenStack projects group instances and grant a quota for the creation of instances and associated resources. An OpenStack project is located in a particular cloud. All member accounts of a project have the same permissions and can create or delete an instance for that project. To find out which projects you are a member of, log in to the OpenStack dashboard in the cloud(s) you have access to; for the list of URLs, see [Cloud Resources](link_to_cloud_resources). The name of the active project is displayed in the dropdown menu to the right of the cloud name; if you are a member of several projects in the same cloud, this menu allows you to select another active project.

Depending on the resources allocated to you, your project may be limited to certain instance templates. For example, compute allocations generally only have type 'c' templates, while persistent allocations generally only have type 'p' templates.

Principal investigators are considered the owners of the projects and are the only ones who can request the creation of a new project or the adjustment of a quota. Also, access rights to a project can only be granted by a principal investigator.

## Volumes

To learn how to create and manage storage with volumes, see [Working with Volumes](link_to_volumes_documentation).

## Images

To learn how to create and manage disk image files, see [Working with Images](link_to_images_documentation).

## Instances

To learn how to manage certain characteristics of your instances, see [Working with Instances](link_to_instances_documentation).

## Availability Zones

The availability zone indicates the group of hardware resources used for instance execution. With the Béluga and Graham clouds, the only available zone is `nova`. However, with Arbutus, three zones are available: `Compute` for running compute templates and `Persistent_01` and `Persistent_02` for running persistent templates (see [Instance Templates](link_to_instance_templates)). Having two persistent zones can be useful when, for example, two instances of a website are located in two different zones; this ensures that the site remains available in case of a problem in one of the zones.

## Security Groups

A security group is a set of rules for managing input and output from instances. To define the rules, select `Project -> Network -> Security Groups`; the displayed page shows the list of existing security groups. If no group has yet been defined, only the default security group appears in the list.

To add or remove a rule from a group, click the `Manage Rules` button corresponding to the group. When the group description is displayed, click the `+ Add Rule` button in the upper right corner. To delete a rule, click the corresponding `Delete Rule` button.

### Default Security Group

[Rules of the default security group (click to enlarge)](link_to_default_security_group_image)

The rules of the default security group allow an instance to access the internet to, for example, download operating system updates or install packages. These rules prevent other computers from accessing the instance, except for other instances that belong to the same security group. We recommend that you do not delete these rules to avoid problems when creating a new instance. The rules are:

* Two egress rules so that the instance has unlimited access outside the network. There is one rule for IPv4 and another for IPv6;
* Two ingress rules so that all instances belonging to the security group can communicate. There is one rule for IPv4 and another for IPv6.

It is possible to add rules to safely connect to an instance under Linux for SSH and RDP (Windows tab, under Firewall and rules allowing the RDP protocol); see the [Cloud: Getting Started Guide](link_to_getting_started_guide).

### Managing Security Groups

Several security groups can be defined, and an instance can belong to more than one group. When defining your groups and rules, carefully consider what needs to be accessed and who will need access. Try to define a minimum of IP addresses and ports in your egress rules; if, for example, you always use the same computer with the same IP address to connect to your instance via SSH, it is logical to allow SSH access only from this IP address. To define the IP address(es) that can access, use the CIDR box in the rule addition window; this web tool allows you to [convert a set of IP addresses into CIDR rules](link_to_cidr_converter).  Furthermore, if you always connect from outside to the same instance via SSH and this connection serves as a gateway to other instances in the cloud, it is logical that the SSH rule is in a separate security group and that this group is associated only with the instance serving as a gateway. However, you must ensure that your SSH keys are configured to allow you to use them for multiple instances (see [SSH Keys](link_to_ssh_keys)). In addition to CIDR rules, other security rules may apply in the case of a project that uses groups. For example, you can configure a security rule for an instance of a project using a MySQL database so that this instance can be accessed by other instances of the default security group.

The security groups to which an instance belongs can be defined at two times:

* During group creation, via `Project -> Compute -> Access and Security`, `Security Groups` tab;
* When the instance is active, via `Project -> Compute -> Instances`, dropdown list in the `Actions` column, `Edit Security Groups` option.

### CIDR Rules

CIDR (for Classless Inter-Domain Routing) is a standard way to define sets of IP addresses (see also the Wikipedia page [CIDR](https://en.wikipedia.org/wiki/Classless_Inter-Domain_Routing)).

An example of a CIDR rule is `192.168.1.1/24`. This looks like a normal IP address to which `/24` has been added. IP addresses are composed of four numbers between 0 and 255, one byte (8 bits) each. In our example, the `/24` ending means that this rule will compare the 24 leftmost bits (3 bytes) to other IP addresses; thus, all addresses starting with `192.168.1` will respect this rule. With the `/32` ending, it is the 32 bits of the IP address that must match exactly, and with the ending, no bit must match, and thus, all IP addresses will respect the rule.

## cloudInit

The first time your instance is launched, you can customize it with cloudInit. This can be done via the OpenStack command-line interface, or by pasting your cloudInit script into the `Customization Script` field of the OpenStack dashboard (`Project -> Compute -> Instances -> Launch Instance` button, `Configuration` option).

### Adding Users with cloud-init During Instance Creation

[Adding multiple users with cloud-init (click to enlarge)](link_to_cloudinit_image)

The following cloud-init script adds users `gretzky` with sudo permissions and `lemieux` without sudo permissions.

```yaml
#cloud-config
users:
  - name: gretzky
    shell: /bin/bash
    sudo: ALL=(ALL) NOPASSWD:ALL
    ssh_authorized_keys:
      - <Gretzky's public key goes here>
  - name: lemieux
    shell: /bin/bash
    ssh_authorized_keys:
      - <Lemieux's public key goes here>
```

For information on the YAML format used by cloud-init, see [YAML Preview](link_to_yaml_preview).

Spaces are important in the YAML format; be careful to leave a space between the initial hyphen and the public key. This configuration replaces the default user added when there is no cloud-init script; the users defined in the script will therefore be the only users of the new instance, which is why it is necessary to ensure that at least one user has sudo permissions. To add other users, simply insert other sections `- name: username` into the script.

To keep the default user created by the distribution (users `debian`, `centos`, `ubuntu`, etc.), use the following script instead:

```yaml
#cloud-config
users:
  - default
  - name: gretzky
    shell: /bin/bash
    sudo: ALL=(ALL) NOPASSWD:ALL
    ssh_authorized_keys:
      - <Gretzky's public key goes here>
  - name: lemieux
    shell: /bin/bash
    ssh_authorized_keys:
      - <Lemieux's public key goes here>
```

Once the instance has started, examine the operation log to verify that the keys are correctly associated with the users. To view the log, select `Project -> Compute -> Instances` and click on the instance name. The `Log` tab displays the instance's console log, which looks like this:

```
ci-info: ++++++++Authorized keys from /home/gretzky/.ssh/authorized_keys for user gretzky++++++++
ci-info: +---------+-------------------------------------------------+---------+------------------+
ci-info: | Keytype |                Fingerprint (md5)                | Options |     Comment      |
ci-info: +---------+-------------------------------------------------+---------+------------------+
ci-info: | ssh-rsa | ad:a6:35:fc:2a:17:c9:02:cd:59:38:c9:18:dd:15:19 |    -    | rsa-key-20160229 |
ci-info: +---------+-------------------------------------------------+---------+------------------+
ci-info: ++++++++++++Authorized keys from /home/lemieux/.ssh/authorized_keys for user lemieux++++++++++++
ci-info: +---------+-------------------------------------------------+---------+------------------+
ci-info: | Keytype |                Fingerprint (md5)                | Options |     Comment      |
ci-info: +---------+-------------------------------------------------+---------+------------------+
ci-info: | ssh-rsa | ad:a6:35:fc:2a:17:c9:02:cd:59:38:c9:18:dd:15:19 |    -    | rsa-key-20160229 |
ci-info: +---------+-------------------------------------------------+---------+------------------+
```

Users can now connect to the instance using their private key (see [SSH Keys](link_to_ssh_keys)).


**(Remember to replace the bracketed placeholders like `link_to_getting_started_guide` with the actual links.)**
