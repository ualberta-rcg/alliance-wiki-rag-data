# Terraform Tutorial for OpenStack Clouds

Terraform is a tool for defining and provisioning data center infrastructure, including virtual machines. It's seeing growing use within the Alliance Federation. Its infrastructure-as-code model allows you to maintain OpenStack resources as a collection of definitions that can be easily updated using your favorite text editor, shared among group members, and stored in version control.

This page serves as a tutorial introducing Terraform and demonstrating its use on our OpenStack clouds. We'll set up a local workspace for Terraform and create a VM with a floating IP and attached volume.


## 1 Preparation

Before starting with Terraform, you need:

*   Access to an OpenStack tenant with available resources.
*   Terraform itself.
*   A few things configured on your workstation or laptop.

### 1.1 Access to OpenStack

For cloud access, see [Getting a Cloud project](wiki_link_here) on our wiki. If you've never used OpenStack before, familiarize yourself with it by creating a VM, attaching a volume, associating a floating IP, and ensuring you can log in to the VM afterward. This tutorial assumes you already have an SSH key pair created and the public key stored with OpenStack.

If you don't know how to do these things, the [Cloud Quick Start](wiki_link_here) guide will help. Creating these resources using the web interface will lay a foundation for understanding what Terraform does and where it's valuable.

### 1.2 Terraform

See the [Terraform downloads page](terraform_download_link_here) for the latest client. This guide is based on Terraform 0.12.

### 1.3 Credentials

There are two ways to provide your OpenStack credentials in a command-line environment: via environment variables or in a configuration file. We'll use one of these methods with Terraform, described in the next section.  Regardless of your preferred method, the OpenStack web interface offers a simple way to download credentials: once logged in, click on `API Access` in the navigation bar. On that page, there's a dropdown menu titled "Download OpenStack RC File". From here, you can download a `clouds.yaml` file or an RC file that can be sourced from your shell session.

The RC file is a series of shell commands that export environment variables to your current shell session. It's not a standalone script and must be sourced in the context of the current session, like so:

```bash
$ source openrc.sh
```

It will then prompt you for your OpenStack password, which, along with necessary information about you, your tenant, and the cloud you're connecting to, will be stored in environment variables prefixed by `OS_`, such as `$OS_AUTH_URL` and so on.

The other method is to create a configuration in `$HOME/.config/openstack/clouds.yaml`. If you don't have such a file, you can download `clouds.yaml` as described above and move it into place. We recommend changing the name given to the cloud in the downloaded file to something meaningful, especially if you use more than one OpenStack cloud. Then, to use the CLI tools described below, simply create an environment variable `$OS_CLOUD` with the name of the cloud you want to use:

```bash
$ export OS_CLOUD=arbutus
```

Whichever method you choose, you'll use it to configure Terraform.

### 1.4 OpenStack Session

It's helpful to have a terminal window open running the OpenStack CLI. This provides a handy reference for the specifications you'll be building, as you'll be looking up flavor and image IDs, and it's useful for verifying the actions performed by Terraform. Horizon can be used for looking up images and for verifying that Terraform is having the intended effects, but it's not possible to directly look up flavor IDs.

The OpenStack CLI (referred to as "OSC") is a Python client best installed through Python Pip and is available for multiple OSes and distributions.  (link to installation instructions here)

### 1.5 Terraform Workspace

Finally, create a directory for your Terraform configuration and state files and consider this your home base for this guide. This is where we will start.


## 2 Defining OpenStack Provider

First, describe the `provider`: this is where you tell Terraform to use OpenStack and how. On initialization, the most recent version of the OpenStack provider plugin will be installed in the working directory, and on subsequent Terraform operations, the included credentials will be used to connect to the specified cloud.

Your connection and credential information for OpenStack can be provided to Terraform in the specification, in the environment, or partially in the specification with the rest in the environment.

The following is an example of a provider specification with connection and credential information:

```terraform
provider "openstack" {
  tenant_name      = "some_tenant"
  tenant_id        = "1a2b3c45678901234d567890fa1b2cd3"
  auth_url         = "https://cloud.example.org:5000/v3"
  user_name        = "joe"
  password         = "sharethiswithyourfriends!"
  user_domain_name = "CentralID"
}
```

For some OpenStack instances, the above would specify the complete set of information necessary to connect to the instance and manage resources in the given project ("tenant"). However, Terraform supports partial credentials, in which you could leave some values out of the Terraform configuration and supply them a different way. This would allow us, for example, to leave the password out of the configuration file, in which case it would need to be specified in the environment with `$OS_PASSWORD`.

Alternatively, if you prefer to use `clouds.yaml`, specify `cloud` in the provider stanza:

```terraform
provider "openstack" {
  cloud = "my_cloud"
}
```

It's acceptable to leave the provider definition completely empty:

```terraform
provider "openstack" {
}
```

In this case, either `$OS_CLOUD` or the variables set by the appropriate RC file would need to be in the executing environment for Terraform to proceed.

The [configuration reference of the OpenStack Provider](openstack_provider_config_link_here) describes the available options in detail.

### 2.1 What Should You Use?

It may be tempting to leave some details in the environment to make the Terraform configuration more portable or reusable, but as we'll see later, the Terraform configuration will and must contain details specific to each cloud, such as flavor and image UUIDs, network names, and tenants.

The most important consideration is security. You probably want to avoid storing your credentials in the Terraform configuration, even if you're not sharing it with anyone.  Always be concerned about hacking!

### 2.2 Initializing Terraform

To ensure we have the provider set up correctly, initialize Terraform and check the configuration. With the provider definition in a file called, for example, `nodes.tf`, run `terraform init`:

```bash
$ terraform init
```

This shows success in initializing Terraform and downloading the OpenStack provider plugin so the OpenStack stanzas will be handled correctly. This does not test out the credentials because this operation doesn’t actually try to connect to the defined provider.


## 3 Defining a VM

Let's look at defining a basic VM.

**Important:** It's good practice to *always* specify flavors and images using their IDs, even when Terraform supports using the name. Although the name is more readable, the ID is what actually defines the state of the resource, and the ID of a given image or flavor will never change.  The *name*, however, can change. If a flavor or image is retired, for example, and replaced with another of the same name, the next time you run Terraform, the updated ID will be detected, and Terraform will determine that you want to rebuild or resize the associated resource. This is a destructive (and reconstructive) operation.

A minimal OpenStack VM can be defined as follows in Terraform:

```terraform
resource "openstack_compute_instance_v2" "myvm" {
  name            = "myvm"
  image_id        = "80ceebef-f9aa-462e-a793-d3c1cf96123b"
  flavor_id       = "0351ddb0-00d0-4269-80d3-913029d1a111"
  key_pair        = "Aluminum"
  security_groups = ["default"]
}
```

This will create a VM with the given name, image, and flavor, and associate with it a key pair and the default security group.

**Note:** If you're following along, use your own values for `image_id`, `flavor_id`, and `key_pair`, or this will probably fail!  Use `flavor list` or `image list` from the OpenStack CLI to find these IDs.  See the appendix for information on finding these IDs in Horizon.

Note that no volumes are supplied. A compute instance on our clouds will already have an associated volume, but a persistent instance will probably fail unless there is sufficient empty space in the image itself. It is recommended that a boot volume be created for VMs using persistent flavors.

### 3.1 Trying it Out

The command `terraform plan` compiles the Terraform definition and attempts to determine how to reconcile the resulting state with the actual state on the cloud, and produces a plan of what it would do if the changes were applied.

```bash
$ terraform plan
```

Read through this output carefully before applying changes to ensure there are no surprises. If you get an error about incomplete credentials, you may have forgotten to define `OS_CLOUD` or source the RC file, or your `clouds.yaml` file may be missing.

### 3.2 Side Note: What Happens to Existing OpenStack Resources?

You may have VMs already defined in your OpenStack project and wonder whether Terraform will affect those resources. It will not. Terraform has no knowledge of resources already defined in the project and does not attempt to determine existing state. Terraform bases its actions on the given configuration and previously determined state relevant to that configuration. Any existing resources are not represented in either and are invisible to Terraform.

It is possible to import previously defined OpenStack resources into Terraform, but it's not trivial and is outside the scope of this tutorial. The important thing here is that any existing resources in your OpenStack project are safe from inadvertent mangling from Terraform—but just to be on the safe side, why don’t you make sure you read the output plans carefully?

### 3.3 Applying the Configuration

Now, use `terraform apply` to actually effect the changes described in the plan.

```bash
$ terraform apply
```

This example will likely fail because OpenStack projects have at least two networks defined: one private and one public. Terraform needs to know which one to use.


## 4 Adding a Network

The name of the private network differs from project to project and the naming convention can differ from cloud to cloud, but typically they are on a 192.168.X.Y network, and can be found in the CLI using `network list` or on Horizon under `Network -> Networks`. If your project's private network is `my-tenant-net`, you will add a `network` resource sub-block to your VM definition similar to the following:

```terraform
resource "openstack_compute_instance_v2" "myvm" {
  name            = "myvm"
  image_id        = "80ceebef-f9aa-462e-a793-d3c1cf96123b"
  flavor_id       = "0351ddb0-00d0-4269-80d3-913029d1a111"
  key_pair        = "Aluminum"
  security_groups = ["default"]

  network {
    name = "my-tenant-net"
  }
}
```

Try `terraform apply` again.  You now have a working VM created by Terraform.


### 4.1 Recap

Note there is now a file in your workspace called `terraform.tfstate`. This was created by Terraform during the application of the new configuration and confirmation of its success. The state file contains details about the managed resources Terraform uses to determine how to arrive at a new state described by configuration updates. In general, you will not need to look at this file, but know that without it, Terraform cannot properly manage resources, and if you delete it, you will need to restore it or recreate it, or manage those resources without Terraform.

You now have a working VM that has been successfully initialized and is on the private network. You can’t log in and check it out, however, because you haven’t assigned a floating IP to this host, so it’s not directly accessible from outside the tenant.


## 5 Adding a Floating IP

Floating IPs are not created directly on a VM in OpenStack: they are allocated to the project from a pool and associated with the VM’s private network interface.

Assuming you do not already have a floating IP allocated for this use, declare a floating IP resource like the following example. The only thing you need is to know the pool from which to allocate the floating IP; on our clouds, this is the external network (`ext_net` in this example).

```terraform
resource "openstack_networking_floatingip_v2" "myvm_fip" {
  pool = "ext_net"
}
```

You may either apply this change immediately or just use `terraform plan` to show what would happen.  Then, associate the floating IP with your VM.


## 6 Adding a Volume

Next, add a root volume to the VM. Since this will replace its boot disk, this is a destructive operation. This is something you need to watch out for in Terraform, and one of the chief reasons for reading your plans carefully before applying. It’s unlikely you’re going to accidentally cause critical issues in creating new resources, but it can be deceptively easy to accidentally create configuration changes that require rebuilding existing VMs.

Since this is a root volume, create it as part of the compute instance, as another subblock along with the network subblock:

```terraform
block_device {
  uuid               = "80ceebef-f9aa-462e-a793-d3c1cf96123b"
  source_type       = "image"
  destination_type  = "volume"
  volume_size       = 10
  boot_index        = 0
  delete_on_termination = true
}
```

Set the `uuid` attribute to the UUID of the image you want to use and remove `image_id` from the outer block definition. The other attributes are self-explanatory, except for `destination_type`, which is here set to `volume` to indicate this is to be stored with an OpenStack-provided volume rather than using disk on the hypervisor. `delete_on_termination` is important—for testing, you will probably want this to be `true` so you don’t have to remember to constantly clean up leftover volumes, but for real use you should consider setting it to `false` as a last defense against accidental deletion of resources.

Do *not* leave the `image_id` attribute defined in the outer compute instance definition! This will work, but Terraform will see a change from "boot from volume" to "boot directly from image" on every run and so will always attempt to rebuild your instance. (This is probably a flaw in the OpenStack provider.)


## 7 The Full Example

```terraform
provider "openstack" {
}

resource "openstack_compute_instance_v2" "myvm" {
  name            = "myvm"
  flavor_id       = "0351ddb0-00d0-4269-80d3-913029d1a111"
  key_pair        = "Aluminum"
  security_groups = ["default"]

  network {
    name = "my-tenant-net"
  }

  block_device {
    uuid               = "80ceebef-f9aa-462e-a793-d3c1cf96123b"
    source_type       = "image"
    destination_type  = "volume"
    volume_size       = 10
    boot_index        = 0
    delete_on_termination = true
  }
}

resource "openstack_networking_floatingip_v2" "myvm_fip" {
  pool = "provider-199-2"
}

resource "openstack_compute_floatingip_associate_v2" "myvm_fip" {
  floating_ip = openstack_networking_floatingip_v2.myvm_fip.address
  instance_id = openstack_compute_instance_v2.myvm.id
}
```


## 8 Appendix

### 8.1 References

*   [Introduction to Terraform](terraform_intro_link_here)
*   [OpenStack provider](openstack_provider_link_here)
*   [OpenStack compute instance resource](openstack_compute_link_here)
*   [Our cloud documentation](our_cloud_docs_link_here) and the [Cloud Quick Start](our_cloud_quickstart_link_here) guide

### 8.2 Examples

*   The [Magic Castle](magic_castle_link_here) project
*   [diodonfrost/terraform-openstack-examples](github_link_here) on GitHub

### 8.3 Finding Image and Flavor UUIDs in Horizon

For those more comfortable using the OpenStack web interface, here's how to find flavor and image UUIDs in Horizon.  You'll need to log into the web interface of the cloud for this information.  (Screenshots or detailed instructions would go here)


Remember to replace placeholder links (e.g., `wiki_link_here`) with the actual links.  Also, consider adding screenshots to the Appendix section to improve clarity.
