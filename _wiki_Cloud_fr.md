# Cloud

This page is a translated version of the page [Cloud](https://docs.alliancecan.ca/mediawiki/index.php?title=Cloud&oldid=153792) and the translation is 100% complete.

Other languages: [English](https://docs.alliancecan.ca/mediawiki/index.php?title=Cloud&oldid=153792), français

We offer an IaaS infrastructure for the creation and operation of virtual environments.

In a cloud, one or more instances (or VMs for virtual machine) are usually created. With administrator privileges, you can install and run all the programs necessary for your project, whether it's analyzing data in particle physics or operating a web service for research in literature or the humanities. The advantage is that you have total control over the installed applications (the software stack). The disadvantage, however, is that you need some experience in managing a computer and installing applications.

It is easy to create snapshots of your instances to make copies; this allows you to have versions with different functionalities or to restart the same instance in case of a power failure, for example.

If your tasks integrate well into a batch processing environment managed by a scheduler on a supercomputer, it would be preferable to use the other resources which are more available and whose software is already configured for common needs. In addition, some tools such as Apptainer can easily be used to run custom software stacks in containers on our computing clusters.

If Apptainer or batch processing does not meet your requirements, choose cloud computing.


## Contents

1. [Getting a project in the cloud environment](#getting-a-project-in-the-cloud-environment)
    1. [Preparing your request](#preparing-your-request)
2. [Creating a virtual machine](#creating-a-virtual-machine)
3. [Your responsibilities](#your-responsibilities)
4. [Advanced topics](#advanced-topics)
5. [Use cases](#use-cases)
6. [Cloud resources](#cloud-resources)
7. [Technical support](#technical-support)


## Getting a project in the cloud environment

Make sure you understand your responsibilities regarding the protection of your project and that of a shared infrastructure.

If you do not have an account, see [these instructions](LINK_TO_INSTRUCTIONS).

A project is an allocation of resources that allows you to create cloud instances.

If you are a principal investigator and have a cloud resource allocation (see the [Competition for resource allocation](LINK_TO_COMPETITION) page), you should already have a project; see the information below. If this is not the case or if you have any doubts, contact [technical support](#technical-support).

Otherwise, fill out the [Cloud projects and allocations by the rapid access service](LINK_TO_FORM) form to:

* obtain access to an existing project; for information on what you will need to provide, see below,
* as a principal investigator, request the creation of a new project and a resource allocation by the [rapid access service](LINK_TO_RAPID_ACCESS),
* request an increase in the resource quota for an existing project.

Requests are generally processed within 48 business hours.


### Preparing your request

To access a computing project or a persistent project, you must know the name of the project and the cloud where it is located; see [how to find the project name](LINK_TO_FIND_PROJECT_NAME) and the [list of our cloud resources](#cloud-resources). The principal investigator must confirm their right to access the project.

If you are requesting the creation of a new project or an increase in the resource quota for an existing project, you must:

* explain why you are requesting cloud resources,
* explain why the CHP clusters are not suitable for your project,
* describe the maintenance and security methods that will be put in place (see [this page](LINK_TO_MAINTENANCE_SECURITY)).

A principal investigator can own a maximum of three projects and the sum of the quotas must respect the established limits (see the limits on [this page](LINK_TO_LIMITS)). They can own allocations for both computing projects and persistent projects.


## Creating a virtual machine

[How to manually create your first virtual machine](LINK_TO_MANUAL_VM_CREATION)

[Technical glossary](LINK_TO_GLOSSARY)

[Storage type selection](LINK_TO_STORAGE_TYPE)

[Troubleshooting common problems](LINK_TO_TROUBLESHOOTING)


## Your responsibilities

For each of your projects, you are responsible for:

* Creating and managing your instances
* Ensuring the security and updating the software of your instances
* Defining security groups for access to your network
* Creating user accounts
* Applying best practices
* Ensuring the security of your instances
* Making backups of your instances


## Advanced topics

If you have more experience, you can:

* Automatically create your instances
* Code your infrastructure with Terraform


## Use cases

* Configure a data server or a web server
* Use vGPUs
* Use GPUs
* Use a graphical interface
* Use IPv6 in the Arbutus cloud


## Cloud resources

* [Béluga Cloud](LINK_TO_BELUGA)
* [Arbutus Cloud](LINK_TO_ARBUTUS); see the documentation
* [Graham Cloud](LINK_TO_GRAHAM)
* [Cedar Cloud](LINK_TO_CEDAR)

Information on hardware and OpenStack versions can be found on the [cloud resources](LINK_TO_CLOUD_RESOURCES) page.

The status of resources and planned maintenance and update activities are described on the wiki page [Resource Status](LINK_TO_RESOURCE_STATUS).


## Technical support

For questions about our cloud service, contact [technical support](LINK_TO_TECHNICAL_SUPPORT).

Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Cloud/fr&oldid=153792](https://docs.alliancecan.ca/mediawiki/index.php?title=Cloud/fr&oldid=153792)"
