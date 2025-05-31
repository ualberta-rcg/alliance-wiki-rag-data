# Arbutus Object Storage

This page is a translated version of the page [Arbutus object storage](https://docs.alliancecan.ca/mediawiki/index.php?title=Arbutus_object_storage&oldid=172838) and the translation is 100% complete.

Other languages:

*   [English](https://docs.alliancecan.ca/mediawiki/index.php?title=Arbutus_object_storage&oldid=172838)
*   français

## Introduction

Object storage is a simpler storage setup than a normal hierarchical file system, but it avoids certain performance bottlenecks. Objects can be created, replaced, or deleted, but they cannot be modified in place, as is the case with traditional storage. This type of storage has become very popular due to its ability to manage multiple files and large files, as well as the existence of many compatible tools.

An object is a file in a flat namespace: an object can be created or uploaded as a whole, but you cannot modify the bytes it contains. An object uses the bucket:tag nomenclature without being nested further. Since operations on buckets concern the entirety of a file, the provider can use a simpler internal representation. The flat namespace allows the provider to avoid metadata bottlenecks; it can be said to be a kind of key-value storage.

The best way to use object storage is to store and export items that are not named in a hierarchical structure; that are primarily accessed in a total and read-only manner; and for which access and control rules are simple. We recommend its use with platforms or software that are designed to work with data that *lives* in an object storage space.

On Arbutus, each project has 1TB of object storage by default. If this is insufficient, you can either use our [fast access service](link-to-fast-access-service-needed). If you need more than 10TB, submit a request to the next [resource allocation competition](link-to-resource-allocation-competition-needed).

Unlike a cluster computing environment, the system administration functions for a user's object storage are the responsibility of that user, meaning that operations like backups must be performed by the user. For more information, see [Cloud Storage Options](link-to-cloud-storage-options-needed).

We offer two different protocols for accessing Object Store in OpenStack: Swift and Amazon Simple Storage Service (S3).

These protocols are very similar and are interchangeable in most cases. It is not necessary to always stick to the same protocol since containers or buckets and objects are accessible via both Swift and S3 protocols. However, some differences exist in the context of object storage on Arbutus.

Swift is the default protocol and is the simplest to use; you don't have to manage credentials since access is done with your Arbutus account. However, Swift does not offer all the features of S3. The main use case is that you must use S3 to manage your containers with access policies because Swift does not support these policies. In addition, S3 allows you to create and manage your own keys, which may be necessary if, for example, you want to create a read-only account for a particular application. See the [OpenStack S3/Swift compatibility list](link-to-compatibility-list-needed).


## Access and Management of the Object Store

To manage the Object Store, you need your own identifier and secret key to access the storage. Generate them with your S3 access ID and secret key for the protocol using the [OpenStack command-line client](link-to-openstack-client-needed).

```bash
openstack ec2 credentials create
```

## Accessing the Object Store

Access policies cannot be done via a web browser, but through a [SWIFT or S3 compatible client](link-to-compatible-client-needed). Access to data containers can be done in several ways:

*   via an S3-compatible client (e.g., `s3cmd`);
*   via Globus;
*   via an HTTPS endpoint in a browser, provided that your policies are configured as public and not by default.

```
https://object-arbutus.cloud.computecanada.ca:443/DATA_CONTAINER/FILENAME
```

## Managing Object Storage on Arbutus

The recommended way to manage containers and objects in Arbutus's Object Storage is to use the `s3cmd` tool, which is available on Linux.

Our documentation provides specific instructions on [configuring and managing access](link-to-s3cmd-config-needed) with the `s3cmd` client.

It is also possible to use other [S3-compatible clients](link-to-s3-compatible-clients-needed) that are also compatible with Arbutus object storage.

In addition, we can perform certain management tasks for our object storage using the Containers section under the Object Storage tab in the [Arbutus OpenStack Dashboard](link-to-openstack-dashboard-needed).

This interface refers to data containers, also called buckets in other object storage systems.

Using the dashboard, we can create new data containers, upload files, and create folders. We can also create data containers using an [S3-compatible client](link-to-s3-compatible-clients-needed).

Please note that data containers belong to the user who creates them and cannot be manipulated by other users.  Therefore, you are responsible for managing your data containers and their contents within your cloud project.

If you create a new *public* container, anyone on the internet can read its contents by simply browsing to the following address:

```
https://object-arbutus.cloud.computecanada.ca/<YOUR_CONTAINER_NAME>/<YOUR_OBJECT_NAME>
```

with your container and object names inserted in place.

It is important to keep in mind that each data container in Arbutus Object Storage must have a *unique name compared to other users*. To ensure this uniqueness, we might want to prefix our data container names with the name of our project to avoid conflicts with other users. A useful rule of thumb is to avoid using generic names like `test` for data containers. It is better to use more specific and unique names like `def-myname-test`.

To make a data container publicly accessible, we can modify its policy to allow public access. This can be useful if we need to share files with a wider audience. We can manage container policies with JSON files, allowing us to specify various access controls for our containers and objects.


## Managing Data Container (Bucket) Policies for Object Storage on Arbutus

**Caution:** Be careful with policies, as a poorly designed policy can prevent you from accessing your data container.

Currently, Arbutus Object Storage only implements a subset of the AWS specification for data container policies. The following example shows how to create, apply, and view a policy. The first step is to create a policy JSON file.

```json
{
  "Version": "2012-10-17",
  "Id": "S3PolicyId1",
  "Statement": [
    {
      "Sid": "IPAllow",
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:*",
      "Resource": [
        "arn:aws:s3:::testbucket",
        "arn:aws:s3:::testbucket/*"
      ],
      "Condition": {
        "NotIpAddress": {
          "aws:SourceIp": "206.12.0.0/16",
          "aws:SourceIp": "142.104.0.0/16"
        }
      }
    }
  ]
}
```

This example denies access except from the specified source IP address ranges in CIDR (Classless Inter-Domain Routing) notation. In this example, `s3://testbucket` is limited to the public IP address range (206.12.0.0/16) used by the Arbutus cloud and the public IP address range (142.104.0.0/16) used by the University of Victoria.

Once you have your policy file, you can apply it to your data container:

```bash
s3cmd setpolicy testbucket.policy s3://testbucket
```

To view the policy, you can use the following command:

```bash
s3cmd info s3://testbucket
```

## Subset

As of September 2023, we support the following actions:

*   `s3:AbortMultipartUpload`
*   `s3:CreateBucket`
*   `s3:DeleteBucketPolicy`
*   `s3:DeleteBucket`
*   `s3:DeleteBucketWebsite`
*   `s3:DeleteObject`
*   `s3:DeleteObjectVersion`
*   `s3:DeleteReplicationConfiguration`
*   `s3:GetAccelerateConfiguration`
*   `s3:GetBucketAcl`
*   `s3:GetBucketCORS`
*   `s3:GetBucketLocation`
*   `s3:GetBucketLogging`
*   `s3:GetBucketNotification`
*   `s3:GetBucketPolicy`
*   `s3:GetBucketRequestPayment`
*   `s3:GetBucketTagging`
*   `s3:GetBucketVersioning`
*   `s3:GetBucketWebsite`
*   `s3:GetLifecycleConfiguration`
*   `s3:GetObjectAcl`
*   `s3:GetObject`
*   `s3:GetObjectTorrent`
*   `s3:GetObjectVersionAcl`
*   `s3:GetObjectVersion`
*   `s3:GetObjectVersionTorrent`
*   `s3:GetReplicationConfiguration`
*   `s3:IPAddress`
*   `s3:NotIpAddress`
*   `s3:ListAllMyBuckets`
*   `s3:ListBucketMultipartUploads`
*   `s3:ListBucket`
*   `s3:ListBucketVersions`
*   `s3:ListMultipartUploadParts`
*   `s3:PutAccelerateConfiguration`
*   `s3:PutBucketAcl`
*   `s3:PutBucketCORS`
*   `s3:PutBucketLogging`
*   `s3:PutBucketNotification`
*   `s3:PutBucketPolicy`
*   `s3:PutBucketRequestPayment`
*   `s3:PutBucketTagging`
*   `s3:PutBucketVersioning`
*   `s3:PutBucketWebsite`
*   `s3:PutLifecycleConfiguration`
*   `s3:PutObjectAcl`
*   `s3:PutObject`
*   `s3:PutObjectVersionAcl`
*   `s3:PutReplicationConfiguration`
*   `s3:RestoreObject`


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Arbutus_object_storage/fr&oldid=172838](https://docs.alliancecan.ca/mediawiki/index.php?title=Arbutus_object_storage/fr&oldid=172838)"
