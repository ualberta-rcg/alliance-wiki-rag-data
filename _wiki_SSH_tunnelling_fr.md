# SSH Tunneling

This page is a translated version of the page [SSH tunneling](https://docs.alliancecan.ca/mediawiki/index.php?title=SSH_tunnelling&oldid=164546) and the translation is 100% complete.

Other languages: [English](https://docs.alliancecan.ca/mediawiki/index.php?title=SSH_tunnelling&oldid=164546), français

## Description

An SSH tunnel allows you to use a gateway computer to connect two computers that cannot be directly connected to each other.  In some cases, setting up a tunnel is necessary because the compute nodes of Niagara, Béluga, and Graham do not have direct access to the internet and cannot be contacted directly via the internet.

A tunnel will be required in the following cases:

* To use a commercial application that needs to contact a license server via the internet.
* To use a visualization application on a compute node that a client application on a user's local computer needs to contact.
* To use Jupyter Notebook on a compute node that a web browser on a user's local computer needs to contact.
* To access the Cedar database server from a computer other than the Cedar login node, for example, your personal computer.


In the first case, the license server is located outside the cluster and is rarely controlled by the user, while in the other cases, the server is located on the compute node and the difficulty is connecting to it from the outside. We address both situations here.

While not essential to using tunnels, you might want to familiarize yourself with [SSH keys](LINK_TO_SSH_KEYS_PAGE_IF_AVAILABLE).


## Contacting a License Server from a Compute Node

### What is a port?

A software port (identified by its number) allows you to distinguish different interlocutors. It is a kind of radio frequency or channel. By obligation or convention, several of these numbers are reserved for particular types of communication; for more information, consult this [list of software ports](LINK_TO_SOFTWARE_PORTS_LIST_IF_AVAILABLE).

Some commercial applications need to connect to a license server somewhere on the internet from a predetermined port. When the node where an application is running does not have internet access, a gateway server with access is used to route communications through this port, from the compute node to the license server.  An SSH tunnel is therefore set up to perform port forwarding.

Often, setting up a tunnel for a serial task only requires two or three commands in the task script. You will need:

* The IP address or name of the license server (here LICSERVER);
* The port number of the license server (here LICPORT).

This information can be obtained from the person who manages the license server. The server must also allow connection from the login nodes; in the case of Niagara, the outward IP address is 142.1.174.227 or 142.1.174.228.

The tunnel can now be set up. On Graham, another solution would be to request a firewall exception for LICSERVER and its port LICPORT.

The gateway server for Niagara is `nia-gw`. For Graham, choose a login node (`gra-login1`, `2`, ...). Here, our gateway node is named `GATEWAY`. Also choose the compute node port number you want to use (here `COMPUTEPORT`).

The SSH command in the script is:

```bash
ssh GATEWAY -L COMPUTEPORT:LICSERVER:LICPORT -N -f
```

The elements following the `-L` parameter define the redirection options:

* `-N` ensures that SSH does not open an interpreter on `GATEWAY`;
* `-f` and `-N` prevent SSH from reading input data (which is nevertheless impossible for a computation task);
* `-f` ensures that SSH is run in the background and that the following commands are executed.

Another command should be added so that the application knows that the license server is located on port `COMPUTEPORT` of the `localhost` server. The term `localhost` is the name by which a computer refers to itself; it should not be replaced by the name of your computer. The exact procedure varies depending on the application and the type of license server; however, it is often only a matter of defining an environment variable in the script, for example:

```bash
export MLM_LICENSE_FILE=COMPUTEPORT@localhost
```

### Example Script

The following script sets up an SSH tunnel to contact `licenseserver.institution.ca` on port 9999.

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

## Connecting to an Application Running on a Compute Node

An encrypted tunnel can be set up on a cluster's login node to connect a user's computer to a compute node of that cluster. The user's computer can thus transparently display visualizations and Jupyter Notebook graphs running on a compute node of the cluster. When the only way to access a database server is through a login node, an SSH tunnel can forward an arbitrary network port from a compute node to the cluster's login node and associate it with the database server.

Graham and Cedar perform NAT (network address translation) so that users can access the internet from the compute nodes. However, the Graham firewall blocks access by default. You must request the technical support to open a particular IP port; indicate the IP address(es) that will use this port.

### From Linux or macOS

The Python package `sshuttle` is recommended.

On your computer, open a new terminal window and run the command:

```bash
[name@my_computer $] sshuttle --dns -Nr userid@machine_name
```

Copy and paste the application URL into your browser. If your application is Jupyter Notebook, for example, the URL will include a token:

`http://cdr544.int.cedar.computecanada.ca:8888/?token=7ed7059fad64446f837567e32af8d20efa72e72476eb72ca`

### From Windows

An SSH tunnel can be created with MobaXTerm as follows:

1. Launch two MobaXTerm sessions.
2. Session 1 should be used to connect to the cluster. Launch your task here according to your application's instructions, for example with Jupyter Notebook. You should receive a URL that contains the name and port of the host node, for example `cdr544.int.cedar.computecanada.ca:8888`.
3. Session 2 is a local terminal in which the SSH tunnel will be set up. Run the next command, replacing the node name with the URL obtained in session 1.

```bash
[name@my_computer] $ ssh -L 8888:cdr544.int.cedar.computecanada.ca:8888 someuser@cedar.computecanada.ca
```

This command redirects connections to local port 8888 to port 8888 on `cdr544.int.cedar.computecanada.ca`, the name given to the remote port.  The numbers do not need to be identical, but it is a convention that makes it easy to identify the local port and the remote port.

Modify the URL obtained in session 1 by replacing the node name with `localhost`. Following the example with Jupyter Notebook, the URL to copy into the browser is:

`http://localhost:8888/?token=7ed7059fad64446f837567e32af8d20efa72e72476eb72ca`

## Connecting to a Database Server on Cedar from Your Computer

The following commands create an SSH tunnel from your computer to the PostgreSQL and MySQL servers.

```bash
ssh -L PORT:cedar-pgsql-vm.int.cedar.computecanada.ca:5432 someuser@cedar.computecanada.ca
ssh -L PORT:cedar-mysql-vm.int.cedar.computecanada.ca:3306 someuser@cedar.computecanada.ca
```

These commands connect the PORT number of your local computer to `cedar.computecanada.ca:PORT`; the port number value must be less than 32768 (2^15). In this example, `someuser` is the username associated with your account. By launching one of these commands, you will be connected to Cedar like any other SSH connection. The only difference between this connection and an ordinary SSH connection is that you can then use another terminal to connect directly to the database server from your personal computer. On your personal computer, run the usual command, for example for PostgreSQL and MySQL:

```bash
psql -h 127.0.0.1 -p PORT -U <your username> -d <your database>
mysql -h 127.0.0.1 -P PORT -u <your username> --protocol=TCP -p
```

A password is required for MySQL; it is located in the `.my.cnf` file located in your Cedar home directory.

The database connection is maintained as long as the SSH connection is active.
