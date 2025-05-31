# SSH Tunnelling

## What is SSH tunnelling?

SSH tunnelling uses a gateway computer to connect two computers that can't connect directly.  On the Alliance network, this is necessary because compute nodes on Niagara, Béluga, and Graham lack direct internet access and can't be contacted directly from the internet.  SSH tunnels are required for:

* Running commercial software on a compute node needing to contact a license server over the internet.
* Running visualization software on a compute node that needs to be contacted by client software on a user's local computer.
* Running a Jupyter Notebook on a compute node that needs to be contacted by a web browser on a user's local computer.
* Connecting to the Cedar database server from outside the Cedar head node (e.g., your desktop).

In the first case, the license server is external to the compute cluster and rarely under user control. In the other cases, the server is on the compute node, but the challenge is connecting to it externally.  While not strictly required, familiarity with SSH key pairs is helpful.


## Contacting a license server from a compute node

**What's a port?**

A port is a number distinguishing communication streams, analogous to a radio frequency or channel. Many port numbers are reserved for specific traffic types. See [List of TCP and UDP port numbers](link_to_port_list_here) for more information.

Commercially licensed programs often connect to a license server via a specific port. If the compute node lacks internet access, a gateway server with internet access forwards communications on that port from the compute node to the license server. This requires an SSH tunnel (also called port forwarding).

Creating an SSH tunnel in a batch job usually requires two or three commands in your job script. You need:

* The IP address or name of the license server (here: `LICSERVER`).
* The port number of the license service (here: `LICPORT`).  Obtain this from the license server maintainer.  For Niagara, the outgoing IP address will be either 142.1.174.227 or 142.1.174.228.  The server must allow connections from the login nodes. For Graham, consider requesting a firewall exception for `LICSERVER` and port `LICPORT` as an alternative.

The gateway server on Niagara is `nia-gw`. On Graham, use a login node (e.g., `gra-login1`, `gra-login2`, etc.). Let's call the gateway node `GATEWAY`. Choose a port number on the compute node (here: `COMPUTEPORT`).

The SSH command for the job script is:

```bash
ssh GATEWAY -L COMPUTEPORT:LICSERVER:LICPORT -N -f
```

The `-L` parameter specifies port forwarding. `-N` prevents SSH from opening a shell on `GATEWAY`. `-f` and `-N` run SSH in the background, allowing the job script to continue.

Inform the software that the license server is on port `COMPUTEPORT` on `localhost`.  `localhost` refers to the computer itself and shouldn't be replaced. How to do this depends on the application and license server; often, it involves setting an environment variable like:

```bash
export MLM_LICENSE_FILE=COMPUTEPORT@localhost
```

### Example job script

This script sets up an SSH tunnel to `licenseserver.institution.ca` on port 9999:

```bash
#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 40
#SBATCH --time 3:00:00
REMOTEHOST=licenseserver.institution.ca
REMOTEPORT=9999
LOCALHOST=localhost
for (( i=0; i<10; ++i )); do
  LOCALPORT=$(shuf -i 1024-65535 -n 1)
  ssh nia-gw -L $LOCALPORT:$REMOTEHOST:$REMOTEPORT -N -f && break
done || {
  echo "Giving up forwarding license port after $i attempts..."
  exit 1
}
export MLM_LICENSE_FILE=$LOCALPORT@$LOCALHOST
module load thesoftware/2.0
mpirun thesoftware
.....
```


## Connecting to a program running on a compute node

SSH tunnelling lets a user's computer connect to a compute node via an encrypted tunnel routed through the cluster's login node. This allows graphical output (e.g., from Jupyter Notebook or visualization software) to be displayed on the user's workstation, even though the application runs on the cluster's compute node.  It's also useful for database servers accessible only through the head node.

Graham and Cedar use Network Address Translation (NAT), allowing internet access from compute nodes. However, Graham blocks access by default; contact technical support to open specific ports, providing the IP address(es) allowed to use that port.


### From Linux or MacOS X

Use the `sshuttle` Python package. Open a terminal and run:

```bash
sshuttle --dns -Nr userid@machine_name
```

Then, paste the application's URL into your browser (e.g., a Jupyter Notebook URL with a token).


### From Windows

Use MobaXTerm:

1. Open two MobaXTerm sessions.
2. Session 1: Connect to the cluster and start your job (e.g., Jupyter Notebook). Note the URL (e.g., `cdr544.int.cedar.computecanada.ca:8888`).
3. Session 2: Set up the SSH tunnel:

```bash
ssh -L 8888:cdr544.int.cedar.computecanada.ca:8888 someuser@cedar.computecanada.ca
```

This forwards connections from local port 8888 to port 8888 on `cdr544.int.cedar.computecanada.ca`. The local and remote port numbers don't need to match, but it's conventional.

Modify the URL from Session 1, replacing the hostname with `localhost` (e.g., `http://localhost:8888/?token=...`).


### Example for connecting to a database server on Cedar from your desktop

Use these commands for PostgreSQL or MySQL respectively:

```bash
ssh -L PORT:cedar-pgsql-vm.int.cedar.computecanada.ca:5432 someuser@cedar.computecanada.ca
ssh -L PORT:cedar-mysql-vm.int.cedar.computecanada.ca:3306 someuser@cedar.computecanada.ca
```

`PORT` should be less than 32768.  `someuser` is your username.  Then, connect to the database server from your desktop:

```bash
psql -h 127.0.0.1 -p PORT -U <your username> -d <your database>
mysql -h 127.0.0.1 -P PORT -u <your username> --protocol=TCP -p
```

MySQL requires a password (from your `.my.cnf` file on Cedar). The database connection stays open as long as the SSH connection is open.
