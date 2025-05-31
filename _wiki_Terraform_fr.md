# Terraform

Terraform is a tool for defining and provisioning data center infrastructure, including virtual machines.  Terraform is increasingly used within the Alliance. Its infrastructure-as-code model allows maintaining OpenStack resources as a collection of definitions that can be easily updated using text editors, shared among group members, and stored in a version control system.

This page is written as a tutorial where we introduce Terraform and demonstrate its use on our OpenStack clouds. We configure our local workspace for Terraform and create a virtual machine (VM) with a floating IP and an attached volume.


## Contents

1. Preparation
    1. Accessing OpenStack
    2. Installing Terraform
    3. Authenticating to OpenStack
    4. OpenStack Session
    5. Workspace
2. Defining the OpenStack Provider
    1. What You Should Use
    2. Initializing Terraform
3. Defining a VM
    1. Testing
    2. What Happens with Existing OpenStack Resources?
    3. Applying the Configuration
4. Adding a Network
    1. Recap
5. Adding a Floating IP Address
6. Adding a Volume
7. Complete Example
8. Appendix
    1. References
    2. Examples
    3. Image and Flavor UUIDs Under Horizon


## Preparation

Before starting with Terraform, you need access to an OpenStack project with available resources, install the `terraform` binary, and perform some configurations on your workstation or laptop.

### Accessing OpenStack

To access the cloud, see [Getting a Project in the Cloud Environment](link-to-getting-a-project). If you have never used OpenStack before, you should first familiarize yourself with this system by creating a virtual machine, attaching a volume, associating a floating IP, and ensuring you can connect to the virtual machine afterward. This tutorial also assumes you have already created an SSH key pair and the public key has been imported into OpenStack.

If you don't know how to do this yet, the [Getting Started Guide](link-to-getting-started-guide) is a good introduction. Creating these resources using the OpenStack web interface will allow you to understand how Terraform works and its usefulness.

### Installing Terraform

Consult the [Terraform download page](link-to-terraform-download) to get the latest version of the binary. We use Terraform 0.12 here.

### Authenticating to OpenStack

There are two ways to provide your OpenStack credentials in a command-line environment: via environment variables or in a configuration file. We will use one of the methods described in the [following section](#defining-the-openstack-provider).  Regardless of your preferred method, OpenStack offers a straightforward way to download credentials. Once logged in, click on `API Access` in the navigation bar; on that page is a dropdown menu titled `Download OpenStack RC File`. From there, you can download a `clouds.yaml` file or an RC file that can be sourced from your shell session.

The RC file contains a list of shell commands that export environment variables in your active session. It is not a standalone script and needs to be sourced via `$ source openrc.sh`. You will then be prompted for your OpenStack password, which will be stored in an environment variable prefixed by `OS_`. Other environment variables will be created with some information about you, your project, and the cloud you want to connect to, for example, `OS_PROJECT_ID`, `OS_AUTH_URL`, `OS_USERNAME`, `OS_PASSWORD`, `OS_USER_DOMAIN_NAME`, `OS_PROJECT_DOMAIN_NAME`.

The other method is to create a configuration in `$HOME/.config/openstack/clouds.yaml`. If you don't already have such a file, you can download `clouds.yaml` as described above and copy it to the desired location. We recommend changing the name given to the cloud in the downloaded file to a meaningful name, especially if you are using multiple OpenStack clouds. Then, to use the CLI tools described below, simply create an environment variable `$OS_CLOUD` with the name of the cloud you wish to use.

```bash
export OS_CLOUD=arbutus
```

No matter what you choose, you will use this to configure Terraform.

### OpenStack Session

It is helpful to have a terminal window open that is running the OpenStack command-line interface. This provides a handy reference for the specifications you are going to create, as you will need flavor and image IDs to verify the actions performed by Terraform. Horizon can be used to look up images and generally verify that Terraform produces the expected effects, but it is not possible to directly look up flavor IDs.

OpenStack CLI (also called `OSC`) is a Python client that can be installed with Python Pip and is [available for several distributions and operating systems](link-to-osc-availability).

### Workspace

Finally, create a directory for your configuration and state files that will serve as your starting point.


## Defining the OpenStack Provider

First, describe the provider: this is where you tell Terraform to use OpenStack and how to use it. Upon initialization, the latest version of the OpenStack provider plugin will be installed in the working directory, and on subsequent Terraform operations, the included credentials will be used to connect to the specified cloud.

Your OpenStack connection and authentication information can be provided to Terraform in the specification, in the environment, or partially in the specification with the rest in the environment.

Here is an example of specifying the provider with connection and authentication information:

```terraform
provider "openstack" {
  tenant_name     = "some_tenant"
  tenant_id       = "1a2b3c45678901234d567890fa1b2cd3"
  auth_url        = "https://cloud.example.org:5000/v3"
  user_name       = "joe"
  password        = "sharethiswithyourfriends!"
  user_domain_name = "CentralID"
}
```

For some OpenStack instances, the above would specify the complete set of information needed to connect to the instance and manage resources in the given tenant project. However, Terraform supports partial credentials where you can leave some values out of the Terraform configuration and provide them another way. This would allow us, for example, to leave the password out of the configuration file, in which case it should be specified in the environment with `$OS_PASSWORD`.

Alternatively, you can use `clouds.yaml` and specify `cloud`.

```terraform
provider "openstack" {
  cloud = "my_cloud"
}
```

It is not necessary to enter a definition for `provider`.

```terraform
provider "openstack" {
}
```

In this case, either `$OS_CLOUD` or the variables set by the appropriate RC file must be in the execution environment for Terraform to proceed.

The available options are described in detail on [this Terraform page](link-to-terraform-openstack-provider-docs).

### What You Should Use

It might be tempting to leave some details in the environment so that the Terraform configuration is more portable or reusable, but as we will see later, the Terraform configuration will and should contain cloud-specific details such as flavor and image UUIDs, network names, and tenants.

The most important thing for your configuration is security. You will likely want to avoid storing your credentials in the Terraform configuration, even if you don't share it with anyone, even if it resides on your own workstation and no one else has access to it. Even if you are not afraid of hacking, it is certainly not a good practice to store passwords and such in configuration files that may end up being copied and moved around your filesystem as you try things out. But also, never forget good practices to counter hacking.

### Initializing Terraform

To ensure the provider is correctly configured, initialize Terraform and check the configuration so far. With the provider definition in a file called, for example, `nodes.tf`, run `terraform init`.

```bash
terraform init
```

The output should show successful initialization and plugin download, indicating that the OpenStack code is handled correctly.  This does *not* test the credentials, as this operation does not actually try to connect to the defined provider.


## Defining a VM

Let's now define a basic VM.

**Important:** It is recommended to *always* specify flavors and images using their IDs, even when Terraform supports using the name. While the name is more readable, the ID is what actually defines the state of the resource, and the ID of a given image or flavor will *never* change. However, the `name` might. If a flavor or image is retired, for example, and replaced by another one with the same name, the next time you run Terraform, the updated ID will be detected, and Terraform will determine that you want to *rebuild or resize the associated resource*. This is a destroy (and rebuild) operation.

A minimal OpenStack virtual machine can be defined as follows in Terraform:

```terraform
resource "openstack_compute_instance_v2" "myvm" {
  name            = "myvm"
  image_id        = "80ceebef-f9aa-462e-a793-d3c1cf96123b"
  flavor_id       = "0351ddb0-00d0-4269-80d3-913029d1a111"
  key_pair        = "Aluminum"
  security_groups = ["default"]
}
```

This creates a VM with the given name, image, and flavor and associates the VM with a key pair and the default security group.

**Note:** If you've followed the tutorial so far (which would be good to do), use your own values for `image_id`, `flavor_id`, and `key_pair`, otherwise this will likely fail.

The values for `image_id` and `flavor_id` are one of the reasons I like to have an open terminal session running the OpenStack command-line interface, connected to the cloud I'm targeting with Terraform: in the terminal I can use `flavor list` or `image list` to list names and IDs.

If you use Horizon (the OpenStack web interface), this is partially possible; see [Image and Flavor UUIDs Under Horizon](#image-and-flavor-uuids-under-horizon) in the appendix.

Note that no volume is provided. A compute instance on our clouds will already have a volume associated with it, but a persistent instance will likely fail unless there is enough free space in the image itself. It is recommended to [create a boot volume](link-to-creating-a-boot-volume) for virtual machines that use persistent versions.

### Testing

The command `terraform plan` compiles the Terraform definition, attempts to determine how to reconcile the resulting state with the current state of the cloud, and produces a plan of the changes that would be applied.

```bash
terraform plan
```

Take note of the output. Even if it contains a lot of information, it is necessary to know its content before accepting the changes to avoid unpleasant surprises.

If you get an error about incomplete credentials, you may have forgotten to set `OS_CLOUD` or source the RC file, or your `clouds.yaml` file might be missing.

These values are those of the resources as they would be defined in OpenStack. Anything marked as `known after apply` will be determined from the state of the newly created resources queried from OpenStack. The other values are defined based on what we have defined or determined by Terraform and the OpenStack plugin as computed or default values.

If you are short on time and it's not a big deal to accidentally destroy resources or rebuild them, at least take the time to check the last line of the plan.

```
Plan: 1 to add, 0 to change, 0 to destroy.
```

In this case, we know we are adding one resource, so this seems correct. If the other values were different from zero, we'd better re-examine our configuration, state, and what is actually defined in OpenStack, and then make the necessary corrections.

### What Happens with Existing OpenStack Resources?

If VMs are already defined in your OpenStack project, you might wonder if Terraform will affect these resources.

Well, no. Terraform does not know about resources that are already defined for the project and does not try to determine their state. Terraform's actions are based on the provided configuration and the previously determined state in the configuration. Existing resources are not represented in Terraform and are invisible to it.

It is possible to import previously defined OpenStack resources into Terraform, but this is [not a trivial task](link-to-importing-openstack-resources) and is outside the scope of this tutorial. The important thing here is that any existing resources in your OpenStack project are protected from unintentional manipulation via Terraform, but why not carefully read the output plans for your peace of mind? :)

### Applying the Configuration

Use `terraform apply` to perform the change described in the plan.

```bash
terraform apply
```

In our example, this fails. There are at least two networks that are defined for an OpenStack project: a private one and a public one. Terraform needs to know which one to use.


## Adding a Network

The name of the private network differs from project to project, and the naming convention may differ from cloud to cloud, but they are usually on a 192.168.X.Y network and can be found in the CLI using `network list` or on Horizon under `Network -> Networks`. If your project's private network is `my-tenant-net`, you will add a `network` resource sub-block to your VM definition similar to the following:

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

Try again.


You now have a virtual machine created by Terraform. You should see your new virtual machine on Horizon or in the output of `server list` in your OpenStack terminal window.

```bash
(openstack) server list -c ID -c Name -c Status
```

In this output, three previously created instances survived untouched by Terraform.

### Recap

Note that there is now a file in your workspace called `terraform.tfstate`. It was created by Terraform when applying the new configuration and confirming its success. The state file contains details about the managed resources that Terraform uses to determine how to arrive at a new state described by the configuration updates. In general, you won't need to consult this file, but know that without it, Terraform cannot properly manage resources, and if you delete it, you will need to restore or recreate it, or manage these resources without Terraform.

You now have a working virtual machine that has been successfully initialized and resides on the private network. However, you cannot connect to and access it because you have not assigned a floating IP address to this host. Therefore, it is not directly accessible from outside the tenant.

If you had another host in this tenant with a floating IP address, you could use it as a bastion host for the new virtual machine, as they will both be on the same private network. This is a good strategy to use for nodes that don't need to be directly accessible from the internet, such as a database server, or simply to preserve the limited resource that is floating IP addresses.

For now, add a floating IP to your new VM.


## Adding a Floating IP Address

A floating IP is not created directly on an OpenStack VM but is allocated to the project from a pool of IPs and associated with the private network interface of the IP.

Assuming you don't already have a floating IP allocated for this use, declare a floating IP resource as in the following example. The only thing you need is to know the pool from which to allocate the floating IP; on our clouds, this is the external network (`ext_net` in this example).

```terraform
resource "openstack_networking_floatingip_v2" "myvm_fip" {
  pool = "ext_net"
}
```

Accept this change now or use `terraform plan` to see what would happen.

This floating IP is now *allocated*, but not yet associated with your instance. Add the following definition:

```terraform
resource "openstack_compute_floatingip_associate_v2" "myvm_fip" {
  floating_ip = openstack_networking_floatingip_v2.myvm_fip.address
  instance_id = openstack_compute_instance_v2.myvm.id
}
```

The attributes of this new resource are defined by references to other resources and their attributes.

**Note:** The current OpenStack provider documentation uses a different syntax than presented here because it has not yet been updated for changes made to Terraform v.12.

References like this are generally `<resource type>.<resource name>.<attribute>`. Other references you might see soon include `var.<variable name>`. In any case, this resource forms an association between the previously created resource and the floating IP allocated in the next step.

If there is a floating IP, you can probably connect to the instance via SSH now.

```bash
ssh centos@X.Y.Z.W
```

Otherwise, you may need to add your computer's IP address to the project's default security group.


## Adding a Volume

Next, add a root volume to the virtual machine. Since this will replace its boot disk, this is a *destructive operation*. This is something you need to pay attention to in Terraform and one of the main reasons why you should carefully read your plans before applying them. It's unlikely you'll accidentally cause critical problems when creating new resources, but it can be incredibly easy to accidentally create configuration changes that require rebuilding existing virtual machines.

Since this is a root volume, create it in the compute instance as a sub-block with the network sub-block.

```terraform
block_device {
  uuid              = "80ceebef-f9aa-462e-a793-d3c1cf96123b"
  source_type       = "image"
  destination_type = "volume"
  volume_size       = 10
  boot_index        = 0
  delete_on_termination = true
}
```

Define the `uuid` attribute as the UUID of the image you want to use and remove `image_id` from the outer block definition. The other attributes are self-explanatory, except for `destination_type`, which is here set to `volume` to indicate that the storage should be done with a volume provided by OpenStack rather than using a disk on the hypervisor.

`delete_on_termination` is important: for testing, you'll probably want this set to `true` so you don't have to constantly remember to clean up leftover volumes, but for real use, you should consider setting it to `false` as a last defense against accidental resource deletion.

Do not leave the `image_id` attribute defined in the outer compute instance definition. This will work, but Terraform will switch from *booting from volume* to *booting directly from image* on each run, and will therefore always try to rebuild your instance. (This is likely a bug in the OpenStack provider.)

Note that there are several warnings informing you of what will be modified, not to mention

```
Plan: 2 to add, 0 to change, 2 to destroy.
```

Your VM will be created with a new SSH key. If you have already connected, you will therefore need to remove the SSH key from your `known_hosts` file (or equivalent). After that, the first thing to do is to connect and apply all available updates.

```bash
[centos@myvm ~]$ sudo yum update -y
...
[goes for ages]
```

You now have a working, terraformed VM, a way to access it, and a place to store data and the latest operating system updates.


## Complete Example

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
    uuid              = "80ceebef-f9aa-462e-a793-d3c1cf96123b"
    source_type       = "image"
    destination_type = "volume"
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


## Appendix

### References

The following might interest those who want to explore deeper and expand on the work done in this tutorial. Note that at the time of writing, the OpenStack provider documentation uses v0.11 syntax, but this should work without problems under v0.12.

* [What is Terraform?](link-to-what-is-terraform)
* [OpenStack Provider](link-to-openstack-provider)
* [OpenStack compute instance v2](link-to-openstack-compute-instance-v2): several use cases for creating instances in OpenStack with Terraform
* [The wiki page on our cloud service](link-to-alliancecan-cloud-service-wiki) and the [Cloud: Getting Started Guide](link-to-alliancecan-cloud-getting-started)

### Examples

* Project [Magic Castle](link-to-magic-castle-project)
* [diodonfrost/terraform-openstack-examples](link-to-diodonfrost-terraform-examples) on GitHub

### Image and Flavor UUIDs Under Horizon

If you are more comfortable with the OpenStack web interface, here is a quick reminder of how to find the UUIDs of flavors and images in Horizon. You will need to log into the cloud's web interface to obtain this information.

To find the UUID of an image, look for the `Images` menu item under `Compute` (1). Locate and select an image. (2) ...and you have the ID.

It's a bit more complicated for flavors. For this, you need to simulate launching an instance, but this doesn't even give you the flavor ID. But you will at least know the name of the flavor you want. On the instance page, select `Flavor`. You should now have a list of flavors and see which ones match your quotas. However, all you have here is the name. To get the ID, there are two options:

1. Use the name for the first Terraform run, then retrieve the ID from the output or state file, and finally, change your configuration to use the ID instead. This shouldn't attempt to recreate the VM, but check before accepting `terraform apply`.
2. Switch to using the OpenStack command-line interface. (Recommended.)


**(Remember to replace the bracketed placeholders like `[link-to-getting-a-project]` with actual links.)**
