# Accessing Object Storage with s3cmd

This page provides information on configuring and accessing object storage on Arbutus using `s3cmd`, a client for this type of storage.

## Installation

Depending on your Linux distribution, the `s3cmd` command is installed with `yum` (RHEL, CentOS) or `apt` (Debian, Ubuntu).

```bash
$ sudo yum install s3cmd
$ sudo apt install s3cmd
```

## Configuration

To configure the `s3cmd` tool, run the command:

```bash
$ s3cmd --configure
```

Perform the following configurations using the keys provided to you or created with the command `openstack ec2 credentials create`:

```
Enter new values or accept defaults in brackets with Enter.
Refer to user manual for detailed description of all options.

Access key and Secret key are your identifiers for Amazon S3. Leave them empty for using the env variables.
Access Key []: 20_DIGIT_ACCESS_KEY
Secret Key []: 40_DIGIT_SECRET_KEY
Default Region [US]:

Use "s3.amazonaws.com" for S3 Endpoint and not modify it to the target Amazon S3.
S3 Endpoint []: object-arbutus.cloud.computecanada.ca

Use "%(bucket)s.s3.amazonaws.com" to the target Amazon S3. "%(bucket)s" and "%(location)s" vars can be used
if the target S3 system supports dns based buckets.
DNS-style bucket+hostname:port template for accessing a bucket []: object-arbutus.cloud.computecanada.ca

Encryption password is used to protect your files from reading
by unauthorized persons while in transfer to S3
Encryption password []:
Path to GPG program [/usr/bin/gpg]: 

When using secure HTTPS protocol all communication with Amazon S3
servers is protected from 3rd party eavesdropping. This method is
slower than plain HTTP, and can only be proxied with Python 2.7 or newer
Use HTTPS protocol []: Yes

On some networks all internet access must go through a HTTP proxy.
Try setting it here if you can't connect to S3 directly
HTTP Proxy server name:
```

This should produce a configuration file similar to the one below, where you will specify the values of your own keys. Use the other configuration options according to your specific case.

```ini
[default]
access_key = <redacted>
check_ssl_certificate = True
check_ssl_hostname = True
host_base = object-arbutus.cloud.computecanada.ca
host_bucket = object-arbutus.cloud.computecanada.ca
secret_key = <redacted>
use_https = True
```

## Creating Buckets

Buckets contain files, and a bucket name must be unique across the entire Arbutus object storage solution.  Therefore, you must create a bucket with a unique name to avoid conflicts with other users. For example, buckets `s3://test/` and `s3://data/` likely already exist. Instead, use names related to your project, such as `s3://def-test-bucket1` or `s3://atlas_project_bucket`. Valid characters for a bucket name are uppercase letters, lowercase letters, numbers, period, hyphen, and underscore (A-Z, a-z, 0-9, ., -, and _).

To create a bucket, use the `mb` (make bucket) command:

```bash
$ s3cmd mb s3://BUCKET_NAME/
```

To check the status of a bucket, run the command:

```bash
$ s3cmd info s3://BUCKET_NAME/
```

The result will be similar to this:

```
s3://BUCKET_NAME/ (bucket):
   Location:  default
   Payer:     BucketOwner
   Expiration Rule: none
   Policy:    none
   CORS:      none
   ACL:       *anon*: READ
   ACL:       USER: FULL_CONTROL
   URL:       http://object-arbutus.cloud.computecanada.ca/BUCKET_NAME/
```

## Uploading Files

To upload a file to a bucket, run:

```bash
$ s3cmd put --guess-mime-type FILE_NAME.dat s3://BUCKET_NAME/FILE_NAME.dat
```

where the bucket name and file name are indicated. The MIME (Multipurpose Internet Mail Extensions) mechanism manages files according to their type. The `--guess-mime-type` parameter detects the MIME type based on the file extension. By default, the MIME type is `binary/octet-stream`.

## Deleting a File

To delete a file in a bucket, run:

```bash
$ s3cmd rm s3://BUCKET_NAME/FILE_NAME.dat
```

## Access Control Lists (ACLs) and Policies

It is possible to associate Access Control Lists (ACLs) and policies with a bucket to indicate who can access a particular resource in the object storage space. These features are very advanced. Here are two simple examples of using ACLs with the `setacl` command.

```bash
$ s3cmd setacl --acl-public -r s3://BUCKET_NAME/
```

With this command, the public can access the bucket and, recursively (-r), each file in the bucket. Access to files can be done with URLs like `https://object-arbutus.cloud.computecanada.ca/BUCKET_NAME/FILE_NAME.dat`

With the next command, the bucket is accessible only by the owner.

```bash
$ s3cmd setacl --acl-private s3://BUCKET_NAME/
```

To see the current configuration of a bucket, use the command:

```bash
$ s3cmd info s3://testbucket
```

For other more advanced examples, see the [s3cmd help site](link_to_s3cmd_help_site) or the `s3cmd(1)` man page.

See the [Object Storage on Arbutus](link_to_arbutus_object_storage) page for examples and guidelines on managing container policies.


**(Note:  Please replace `link_to_s3cmd_help_site` and `link_to_arbutus_object_storage` with the actual links.)**
