# Cloud Infrastructure Documentation

## We Offer Infrastructure as a Service

Our Infrastructure as Service (IaaS) supports virtualization.  A user typically creates or "spins up" one or more virtual machines (VMs or instances). They then log into the VM with administrative privileges, install desired software, and run applications. These applications can range from CPU-intensive tasks (e.g., particle physics data analysis) to web services.  The advantage is complete control over the software stack. The disadvantage is that the user needs some experience in software installation and computer management.

Virtual machines are easily replicated.  A snapshot of a VM can be started elsewhere, simplifying replication, scaling, and recovery from interruptions.


If your work fits the HPC batch submission workflow (see [What is a scheduler?](#what-is-a-scheduler)), it's preferable to work outside the cloud due to greater HPC resources and pre-configured software. Tools like [Apptainer](https://apptainer.org/) enable running custom software stacks in containers within our HPC clusters.  If your needs aren't met by Apptainer or HPC batch, the cloud is the solution.


## Contents

1. [Getting a cloud project](#getting-a-cloud-project)
    1. [Preparing your request](#preparing-your-request)
2. [Creating a virtual machine on the cloud infrastructure](#creating-a-virtual-machine-on-the-cloud-infrastructure)
3. [User responsibilities](#user-responsibilities)
4. [Advanced topics](#advanced-topics)
5. [Use cases](#use-cases)
6. [Cloud systems](#cloud-systems)
7. [Support](#support)


## Getting a cloud project

Review the important role you'll play in safeguarding your research and the shared cloud infrastructure.

If you don't have an account, create one using [these instructions](link-to-instructions-needed).

A *project* is a resource allocation for creating VMs within a cloud.

If you're a primary investigator (PI) with an active cloud resource allocation (see [RAC](#rac)), you should already have a project. See the sections below on cloud usage. If not, or if unsure, contact [technical support](#support).

Otherwise, go to the [Alliance cloud project and RAS request form](link-to-form-needed) to request access to an existing project (see below for required information).  As a PI, you may also request a new project with our Rapid Access Service ([RAS](#ras)) or request an increase in an existing project's quota.

Requests are typically processed within two business days.


### Preparing your request

When requesting access to an existing project, you'll need the project name and cloud. See the section on [projects](#projects) for finding the project name and the section on [cloud systems](#cloud-systems) for a list of our clouds.  PI confirmation is required for access requests.

When requesting a new project or increased quota, provide a brief justification (a few sentences):

* Why you need cloud resources
* Why an HPC cluster isn't suitable
* Your plans for efficient resource usage
* Your plans for maintenance and security ([refer to this page](link-to-security-page-needed))

A PI may own up to 3 projects, but the total project quota must be within the [RAS](#ras) allocation limits. A PI may have both compute and persistent cloud RAS allocations.


## Creating a virtual machine on the cloud infrastructure

The [cloud quick start guide](link-to-quickstart-guide-needed) describes how to manually create your first VM.

Review the [glossary](link-to-glossary-needed) for definitions.

Consider [storage options](link-to-storage-options-needed) best suited to your use case.

See the [troubleshooting guide](link-to-troubleshooting-guide-needed) for common cloud computing issues.


## User responsibilities

For each cloud project, you're responsible for:

* Creating and managing your virtual machines
* Securing and patching software on your VM
* Defining security groups to allow network access
* Creating user accounts
* Following best practices
* Considering security issues
* Backing up your VMs


## Advanced topics

More experienced users can:

* Automatically create VMs
* Describe their VM infrastructure as code using [Terraform](https://www.terraform.io/).


## Use cases

More detailed instructions are available for common cloud use cases, including:

* Configure a data or web server
* Using vGPUs (standard shared GPU allocation) in the cloud
* Using PCI-e passthrough GPUs in the cloud
* Setting up a GUI Desktop on a VM
* Using IPv6 in Arbutus cloud


## Cloud systems

Your project will be on one of the following clouds:

* Béluga
* Arbutus
* Graham
* Cedar

Details on underlying hardware and OpenStack versions are on the [cloud resources](link-to-cloud-resources-needed) page. The [System status](link-to-system-status-needed) wiki page contains information about current cloud status, planned maintenance, and upgrades.


## Support

For questions about our cloud service, contact [technical support](link-to-support-needed).


**(Retrieved from "https://docs.alliancecan.ca/mediawiki/index.php?title=Cloud/en&oldid=148344")**
