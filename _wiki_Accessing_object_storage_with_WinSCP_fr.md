# Accessing Object Storage with WinSCP

This page contains information on configuring and accessing object storage on Arbutus with WinSCP, a client for this type of storage.

## Installation

Install WinSCP from [https://winscp.net/](https://winscp.net/).

## Configuration

Under **New Session**, enter:

*   **File protocol:** Amazon S3
*   **Host name:** object-arbutus.cloud.computecanada.ca
*   **Port number:** 443
*   **Access key ID:** 20_DIGIT_ACCESS_KEY

Then click the **Save** button.

**Configuration Window**

Next, click the **Edit** button and then **Advanced...**. Under **Environment**, select **S3**. In the protocol options, select **Path** in the **URL style** field.

**Path Configuration Window**

Choosing **Path** is important for WinSCP to function and avoid errors like "WinSCP resolve error".

## Utilisation

Click the **Login** button and use the WinSCP interface to create buckets and transfer files.

**File Transfer Window**

## Access Control Lists (ACLs) and Policies

Right-click on the file name to get the access list, for example:

**WinSCP ACL screen**
