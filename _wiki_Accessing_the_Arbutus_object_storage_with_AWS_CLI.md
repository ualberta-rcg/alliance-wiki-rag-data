# Accessing the Arbutus Object Storage with AWS CLI

This page contains instructions on how to set up and access Arbutus object storage with the AWS Command Line Interface (CLI), one of the object storage clients available for this storage type.

Compared to other object storage clients, AWS CLI has better support for large (>5GB) files and the helpful `sync` command. However, not all features have been thoroughly tested.

## Installing AWS CLI

```bash
pip install awscli awscli-plugin-endpoint
```

## Configuring AWS CLI

1. **Generate an access key ID and secret key:**

   ```bash
   openstack ec2 credentials create
   ```

2. **Edit or create `~/.aws/credentials` and add the credentials generated above:**

   ```ini
   [default]
   aws_access_key_id = <access_key>
   aws_secret_access_key = <secret_key>
   ```

3. **Edit `~/.aws/config` and add the following configuration:**

   ```ini
   [plugins]
   endpoint = awscli_plugin_endpoint

   [profile default]
   s3 =
     endpoint_url = https://object-arbutus.cloud.computecanada.ca
     signature_version = s3v4
   s3api =
     endpoint_url = https://object-arbutus.cloud.computecanada.ca
   ```

## Using AWS CLI

1. **Set the AWS profile:**

   ```bash
   export AWS_PROFILE=default
   ```

2. **List objects in a container:**

   ```bash
   aws s3 ls <container-name>
   ```

3. **Synchronize a local directory with a remote container:**

   ```bash
   aws s3 sync local_directory s3://<container-name>/<prefix>
   ```

More examples of using the AWS CLI can be found on [this external site](link_to_external_site_here).  *(Note:  Please replace `link_to_external_site_here` with the actual link if available.)*
