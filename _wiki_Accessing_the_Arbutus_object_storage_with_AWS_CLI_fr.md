# Accessing the Arbutus Object Storage with AWS CLI

This page contains information on configuring and accessing the Arbutus object storage with AWS CLI, a client for object storage.

Compared to other object storage clients, AWS CLI offers better support for large files (>5GB) in addition to the `sync` command which is very useful. Note however that we have not tested all functionalities.


## Installation

```bash
pip install awscli awscli-plugin-endpoint
```

## Configuration

Generate the access key ID and secret key.

```bash
openstack ec2 credentials create
```

Modify or create `~/.aws/credentials` and add the newly generated information.

```ini
[default]
aws_access_key_id = <access_key>
aws_secret_access_key = <secret_key>
```

Modify `~/.aws/config` and add the following configuration:

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

## Usage

```bash
export AWS_PROFILE=default
aws s3 ls <container-name>
aws s3 sync local_directory s3://container-name/prefix
```

You can find other examples of using AWS CLI on [this external site](link_to_external_site_here).  *(Note:  The original document did not provide a link.  Please insert the correct link here.)*
