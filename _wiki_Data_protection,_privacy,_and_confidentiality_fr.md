# Data Protection, Privacy, and Confidentiality

## Processing Personal, Private, or Sensitive Data (e.g., clinical research data on humans)

Currently, none of our resources are assigned to the processing of sensitive data.  All our resources are managed according to best practices for academic research, and we make considerable efforts to ensure data integrity, confidentiality, and availability. However, none of our resources are formally certified to meet the security and confidentiality requirements that may apply to certain datasets.  Generally, our resources are shared among several people, including networks, nodes, and memory spaces. The data stored there may not be encrypted. We offer standard Linux features for file system segregation and access control to files and directories, and our system administrators have access to all this material as needed or with the authorization of the owners.

The protection of data confidentiality is the responsibility of the researchers.  As such, we invite you to review our policies.

You can contact our technical support team for assistance with managing your sensitive data, as well as for advice on access control, encryption, storage, and data transmission.


## Hardware Failure

Our basic principle is to have a certain level of duplication for most file systems, depending on the risk level presented by the hardware. For example:

* There is no form of duplication of file systems stored locally on the compute nodes.
* There are no backups of the `/scratch` file systems, but they are configured to be protected against multiple disk failures.
* There are periodic backups of the `/project` and `/home` file systems, which are also protected against multiple disk failures.
* A copy is made to tape of the file systems of the `/nearline` storage space.


## Unauthorized Access

Unauthorized access could occur mainly through hardware or software.

Regarding hardware, the physical infrastructure is only accessible by authorized personnel. Any storage equipment that needs to be removed due to a failure is either destroyed, encrypted, or erased before being returned to the supplier for replacement.

Software access to the file systems of our clusters is protected by standard POSIX permissions and ACLs. Each file has an associated owner and group. The group associated with a file is either a user or a research project. The default permissions are such that newly created files are writable by the owner and readable by the group. The default group associated with a file may depend on where the file is located in the file system. The file owner must ensure that the file belongs to the correct group and that the appropriate access permissions are set.

If the access permissions to a file are correctly defined, unauthorized access can only occur through privilege escalation (hacking). To counter this, our technical team monitors CVE (Common Vulnerabilities and Exposures) mailing lists and applies the necessary patches. We also examine abnormal behavior that may indicate an intrusion, in addition to imposing stricter security measures for the accounts of our personnel who have privileged access compared to regular users.

It should be noted that our clusters are part of a shared infrastructure. Even though we take every precaution to reduce the risk of unauthorized access, the possibility always remains. If your data requires a high level of security, it is advisable to encrypt it.
