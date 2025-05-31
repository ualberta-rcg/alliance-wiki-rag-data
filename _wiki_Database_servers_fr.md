# Database Servers

This page is a translated version of the page Database servers and the translation is 100% complete.

Other languages: [English](link-to-english-page), [français](current-page-url)


## Database Servers for Research

MySQL and Postgres database servers are available on Cedar and Graham.

**NOTE:** As of January 13, 2025, the Graham cluster capacity will be reduced to 25% (see [Capacity Reduction](link-to-capacity-reduction-page)) until the new Nibi system is available. No database servers will be offered during the transition.


| Information          | Cedar, MySQL                                      | Cedar, Postgres                                           | Graham, MySQL                                     |
|----------------------|---------------------------------------------------|--------------------------------------------------------|----------------------------------------------------|
| **Description**       | General-purpose server for configuring and processing SQL tables | General-purpose server for configuring and processing SQL tables; includes the PostGIS extension for spatial data | General-purpose server for configuring and processing SQL tables |
| **Long Name**         | `cedar-mysql-vm.int.cedar.computecanada.ca`         | `cedar-pgsql-vm.int.cedar.computecanada.ca`            | `cc-gra-dbaas1.sharcnet.ca`                        |
| **IP:**               | `199.241.163.99`                                   |                                                        |                                                    |
| **Short Name**        | `cedar-mysql-vm`                                  | `cedar-pgsql-vm`                                      | ---                                                 |
| **Recent Version**    | MariaDB version 11.5                               | PostgreSQL version 16, PostGIS version 3.3 (extension) | MariaDB version 11.5                               |
| **Documentation**     | [MariaDB website](link-to-mariadb-website)         | [Postgres website](link-to-postgres-website), [PostGIS documentation](link-to-postgis-documentation) | [MariaDB website](link-to-mariadb-website)         |


## MySQL Server on Cedar

The MySQL server on Cedar offers MariaDB 10.4, which is compatible with other MySQL versions. For compatibility information, see [MariaDB versus MySQL: Compatibility](link-to-compatibility-page).

The MariaDB server is the instance `cedar-mysql-vm` (long name, `cedar-mysql-vm.int.cedar.computecanada.ca`). If you have an account on the MySQL server, you can access it only from the login node (`cedar.computecanada.ca`), compute nodes, or the [Cedar portal](link-to-cedar-portal).

For security reasons, you cannot connect directly to the database server via SSH.


### Account and Connection

You must have a MySQL account to have the required privilege to create a database. To obtain an account on the Cedar MySQL server, contact [technical support](link-to-technical-support) indicating:

*   Your Alliance account username
*   The required space for your project's database

We will create a MySQL account where the username will be your Alliance account username and a 16-digit random number string as the password. A file named `.my.cnf` will be saved in your `/home` directory containing the username, password, database server name, and other information needed to connect. This file is confidential. Its content cannot be modified, but the file can be read or deleted. Deleting this file will lose your database access.

Launch the `mysql` client to connect to the MySQL server. An older version of the client is available without having to [load a module](link-to-module-loading), but you will not have the latest server features. We recommend loading a newer version of the client with:

```bash
[name@server ~]$ module load mariadb
[name@server ~]$ mariadb --version
```

Test your MySQL account configuration with:

```sql
[name@server ~]$ mysql
MariaDB [(none)]> show databases;
MariaDB [(none)]> quit
```

Do not use the `-p` or `-h` options as arguments when launching `mysql`. The password and server name will be automatically retrieved from the `.my.cnf` file.

You can submit an SQL command from the login node since CPU usage is largely on the database server side. However, if your script contains multiple SQL commands and uses a lot of CPU, it should be part of a job submitted to the scheduler; see [Running Jobs](link-to-running-jobs) for more information.


### Configuration

To create tables and make queries, you must create your own database whose name must start with `username_`, your MySQL username. If your username is `david`, the database name must start with `david_`, and the commands to create `david_db1` would be:

```sql
[name@server ~]$ mysql
MariaDB [(none)]> CREATE DATABASE david_db1;
MariaDB [(none)]> quit
```

You can create multiple databases, but their names must start with `username_`. These databases will only be accessible by you from the login node (`cedar.computecanada.ca`), compute nodes, or the [Cedar portal](link-to-cedar-portal), and you will have all privileges for object creation, such as tables and views.


### Using Your Database

Suppose your account is `david` and you have created the database `david_db1`. To use it, launch:

```sql
[name@server ~]$ mysql
MariaDB [(none)]> -- List available databases. Confirm david_db1 is in the list
MariaDB [(none)]> SHOW DATABASES;
MariaDB [(none)]> -- Get into the database
MariaDB [(none)]> USE david_db1;
MariaDB [(none)]> ... Issue SQL commands. See below for information.
MariaDB [(none)]> quit
```

See the following websites for more information on MariaDB:

*   [Knowledge Base](link-to-mariadb-knowledge-base)
*   [Training & Tutorials](link-to-mariadb-training)
*   [SQL Statements & Structure](link-to-mariadb-sql-statements)
*   [Optimization and Indexes](link-to-mariadb-optimization)


### Sharing Your MySQL Data

If you have a MySQL account on Cedar, you can share your data. To share a table:

1.  Log in to MySQL with `mysql`.
2.  Launch the command `USE database;`  `database` is the name of the database where the table you want to share is located.
3.  Launch the command `GRANT priv_type ON mytable TO 'user'@'172.%';`
    *   `priv_type` is the type of privilege you want to grant.
    *   `mytable` is the table name.
    *   `user` is the username of the person with whom you want to share the table.


#### Example of Sharing

Here, user `david` wants to share the table `mytable` from the database `david_db` with `john` in read-only mode.

```sql
[name@server ~]$ mysql
MariaDB [(none)]> USE david_db;
MariaDB [(none)]> GRANT SELECT on mytable to 'john'@'172.%';
MariaDB [(none)]> quit;
```


## PostgreSQL Server on Cedar

The Postgres server on Cedar offers Postgres and the PostGIS extension.

The PostgreSQL server is the instance `cedar-pgsql-vm` (long name, `cedar-pgsql-vm.int.cedar.computecanada.ca`). If you have an account on the PostgreSQL server, you can access it only from the login node (`cedar.computecanada.ca`), compute nodes, or the [Cedar portal](link-to-cedar-portal).

For security reasons, you cannot connect directly to the database server via SSH.

To obtain an account on the Cedar PostgreSQL server, contact [technical support](link-to-technical-support) indicating:

*   Your username
*   The required space for your project's database
*   If you need the PostGIS extension


### Account and Connection

We will create a PostgreSQL account where the username will be your Alliance account username. You will have access to a database whose name will be `<username>_db`. You cannot create a database, but if you need more than one, write to [technical support](link-to-technical-support).

You do not need a password to access your PostgreSQL account on Cedar. For security reasons, the password for your Alliance account should NEVER be required or used in a script. Users thus have no direct access to other users' databases.

Launch the `psql` client to connect to the PostgreSQL server. An older version of the client is available without having to [load a module](link-to-module-loading), but you will not have the latest features of version 10. We recommend loading a newer version with:

```bash
[name@server ~]$ module load postgresql
[name@server ~]$ psql --version
```


### Using Your Database

Suppose your account is `david` and you have been assigned the database `david_db`. To use it from a login node, launch:

```sql
[name@server ~]$ psql -h cedar-pgsql-vm -d david_db
david_db=> -- List names of tables in your database
david_db=> \d t
david_db=> ... Issue SQL commands. See below for more information.
david_db=> -- Quit
david_db=> \q
```

See the following websites for more information on PostgreSQL:

*   [Tutorial](link-to-postgresql-tutorial)
*   [Manuals](link-to-postgresql-manuals)
*   [Release Notes](link-to-postgresql-release-notes)


### Sharing Your PostgreSQL Data

To share your PostgreSQL database data:

1.  The person with whom you want to share your data must have a PostgreSQL account on the cluster (see above).
2.  Give this person `connect` access to your database.
3.  For each table or view you want to share, also give one or more of the `select`, `update`, `insert`, and `delete` accesses.
4.  Access to a table, view, or database can be revoked.


Here's an example where David shares a table with Kim:

```sql
[name@server ~]$ psql -h cedar-pgsql-vm -d david_db
david_db=> -- Give kim connect access to the database
david_db=> grant connect on database david_db to kim;
david_db=> -- Give kim select-only access to a table called mytable
david_db=> grant select on mytable to kim;
david_db=> -- Quit
david_db=> \q
```

Here, Kim accesses the shared table:

```sql
[name@server ~]$ psql -h cedar-pgsql-vm -d kim_db
kim_db=> -- Connect to the database containing the table to be accessed
kim_db=> \c david_db
david_db=> -- Display the rows in the shared table
david_db=> select * from mytable;
david_db=> -- Quit
david_db=> \q
```

Here, David revokes Kim's access rights:

```sql
[name@server ~]$ psql -h cedar-pgsql-vm -d david_db
david_db=> -- Revoke kim's select-only access to a table called mytable
david_db=> revoke select on mytable from kim;
david_db=> -- Revoke kim's connect access to the database
david_db=> revoke connect on database david_db from kim;
david_db=> -- Quit
david_db=> \q
```


## MySQL Server on Graham

The steps to obtain and use an account on the Graham MySQL server are similar to those described above for Cedar.


## Cloud Servers

### On-Demand Databases (DBaaS)

If you need more than one instance to process your database, you can use MySQL/MariaDB or Postgres on a physical machine.

Backups are performed daily and kept for three months.

To access them, contact [technical support](link-to-technical-support). In your request, indicate the client network or IP address from which you want to access the database.

| Type   | Host Name                                      | TCP Port |
|--------|-------------------------------------------------|----------|
| mysql  | `dbaas101.arbutus.cloud.computecanada.ca`       | 3306     |
| pgsql  | `dbaas101.arbutus.cloud.computecanada.ca`       | 5432     |

Download the Certificate Authority certificate.


### PostgreSQL

Your instance will use an ssl connection to connect to the DBaaS host.

In the following example, the connection is made to the DBaaS host by `user01` and uses the `dbinstance` database via an ssl connection.

```bash
psql --set "sslmode=require" -h dbaas101.arbutus.cloud.computecanada.ca -U user01 -d dbinstance
Password for user user01: 
SSL connection (protocol: TLSv1.3, cipher: TLS_AES_256_GCM_SHA384, bits: 256, compression: off)
dbinstance=> \l dbinstance
```

The ssl connection applies and plaintext connections will fail.


### MariaDB/MySQL

Your instance will use an ssl connection to connect to the DBaaS host.

In the following example, the connection is made to the DBaaS host by `user01` and uses the `dbinstance` database via an ssl connection.

```bash
mysql --ssl -h dbaas101.arbutus.cloud.computecanada.ca -u user01 -p dbinstance
Enter password: 
MariaDB [dbinstance]> show databases;
```

If you try to connect with plaintext, your authentication will fail.

```bash
mysql -h dbaas101.arbutus.cloud.computecanada.ca -u user01 -p dbinstance
Enter password: 
ERROR 1045 (28000): Access denied for user 'user01'@'client.arbutus' (using password: YES)
```

**(Remember to replace bracketed placeholders like `[link-to-...]` with the actual links.)**
