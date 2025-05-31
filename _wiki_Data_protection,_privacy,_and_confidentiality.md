# Data Protection, Privacy, and Confidentiality

## Handling Sensitive Data

We don't currently have resources specifically designated for sensitive data.  All our resources are managed using best practices for academic research systems, prioritizing data integrity, confidentiality, and availability. However, no resource is formally certified to meet specific security or privacy assurance levels required for certain datasets.  We primarily provide shared resources (networks, nodes, memory), and data isn't guaranteed to be encrypted at rest. Standard Linux filesystem segregation and access control are offered, with sysadmins accessing data only when necessary or authorized by owners.

**Responsibility for data protection and privacy ultimately rests with the researcher.**  See our [policies](link-to-policies-page) for details.  Support staff can advise on handling sensitive data (access control, encryption, storage, transmission). Contact [technical support](link-to-technical-support) for assistance.


## Data Protection Against Hardware Failure

Our strategy involves duplication for most filesystems, varying by risk:

*   **Local storage on compute nodes:** No duplication.
*   **Scratch filesystems:** High reliability against multiple simultaneous disk failures, but no backup.
*   **Project and home filesystems:** High reliability against multiple simultaneous disk failures, with periodic backups.
*   **Nearline storage:** Tape copies of data.


## Data Protection Against Unauthorized Access

Unauthorized access can occur through hardware or software vulnerabilities.

**Hardware:** Only approved personnel have physical access.  Storage devices removed due to hardware failure are destroyed or erased/encrypted before vendor return.

**Software:** Our clusters use standard POSIX and ACL permissions. Each file has an owner and group.  Default permissions allow owner write and group read access. The default group depends on file location (project or user).  File owners are responsible for ensuring correct group and permissions.

Unauthorized access, assuming correct permissions, requires privilege escalation. We mitigate this by:

*   Monitoring Common Vulnerabilities and Exposures (CVE) and applying patches.
*   System monitoring for anomalous behavior.
*   Stricter security for privileged accounts.

Our clusters are shared infrastructure. While we minimize risk, unauthorized access remains possible.  Consider encrypting data requiring higher protection.
