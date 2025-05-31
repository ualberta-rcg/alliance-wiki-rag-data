# Accessing Arbutus Object Storage with WinSCP

This page provides instructions for setting up and accessing Arbutus object storage using WinSCP, one of the available object storage clients.

## Installing WinSCP

WinSCP can be installed from [https://winscp.net/](https://winscp.net/).

## Configuring WinSCP

Under "New Session", configure the following settings:

* **File protocol:** Amazon S3
* **Host name:** `object-arbutus.cloud.computecanada.ca`
* **Port number:** 443
* **Access key ID:** 20_DIGIT_ACCESS_KEY

Save these settings.  *(Image: WinSCP configuration screen)*

Next, click "Edit", then "Advanced...", navigate to "Environment" -> "S3" -> "Protocol options" -> "URL style:".  Change this setting from "Virtual Host" to "Path". *(Image: WinSCP Path Configuration)*

The "Path" setting is crucial; otherwise, WinSCP will not function correctly, resulting in hostname resolution errors like this: *(Image: WinSCP resolve error)*


## Using WinSCP

Click the "Login" button and use the WinSCP GUI to create buckets and transfer files. *(Image: WinSCP file transfer screen)*


## Access Control Lists (ACLs) and Policies

Right-clicking on a file allows you to set its ACL. *(Image: WinSCP ACL screen)*
