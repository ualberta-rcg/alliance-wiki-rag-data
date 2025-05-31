# Accessing Arbutus Object Storage with s3cmd

This page provides instructions on setting up and accessing Arbutus object storage using `s3cmd`, an object storage client.

## Installing s3cmd

Depending on your Linux distribution, install `s3cmd` using `yum` (RHEL, CentOS) or `apt` (Debian, Ubuntu):

```bash
$ sudo yum install s3cmd
$ sudo apt install s3cmd
```

## Configuring s3cmd

Configure `s3cmd` using:

```bash
$ s3cmd --configure
```

Provide the following configurations using keys obtained via the `openstack ec2 credentials create` command:

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

This will create an `s3cmd` configuration file.  The example below redacts keys; replace `<redacted>` with your actual key values:

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

## Create Buckets

Create a bucket using the `mb` (make bucket) command. Bucket names must be unique:

```bash
$ s3cmd mb s3://BUCKET_NAME/
```

Check bucket status using the `info` command:

```bash
$ s3cmd info s3://BUCKET_NAME/
```

Example output:

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

## Upload Files

Upload a file using the `put` command:

```bash
$ s3cmd put --guess-mime-type FILE_NAME.dat s3://BUCKET_NAME/FILE_NAME.dat
```

`--guess-mime-type` guesses the MIME type based on the file extension. The default is `binary/octet-stream`.

## Delete Files

Delete a file using the `rm` command:

```bash
$ s3cmd rm s3://BUCKET_NAME/FILE_NAME.dat
```

## Access Control Lists (ACLs) and Policies

Manage access using ACLs and policies.  Here are two examples using `setacl`:

Make the bucket and its contents publicly accessible:

```bash
$ s3cmd setacl --acl-public -r s3://BUCKET_NAME/
```

Files can be accessed via URLs like `https://object-arbutus.cloud.computecanada.ca/BUCKET_NAME/FILE_NAME.dat`.

Limit access to the bucket owner only:

```bash
$ s3cmd setacl --acl-private s3://BUCKET_NAME/
```

View bucket configuration:

```bash
$ s3cmd info s3://testbucket
```

For more advanced examples, refer to the `s3cmd` help site or `s3cmd(1)` man page.  See the main [object storage](link-to-object-storage-page) page for instructions on managing bucket policies.
