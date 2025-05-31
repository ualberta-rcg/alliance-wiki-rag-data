# Database Servers

## Database Servers Available for Researchers

The Alliance offers access to MySQL and Postgres database servers for researchers on both Cedar and Graham.

**NOTE:** As of January 13, 2025, the Graham cluster will be operating at approximately 25% capacity ([details here](link_to_details)) until Nibi is available. No database server will be provided on Graham during the transition to the new system.


| Database Servers | Information                                                                     |
|-----------------|---------------------------------------------------------------------------------|
| Cedar - MySQL    | General purpose server for setting up SQL tables and issuing SQL commands.       |
| Cedar - Postgres | General purpose server; includes PostGIS extension for geocoding.                |
| Graham - MySQL   | General purpose server for setting up SQL tables and issuing SQL commands.       |


| Server Details                                       | Cedar - MySQL                                      | Cedar - Postgres                                     | Graham - MySQL                                   |
|-------------------------------------------------------|------------------------------------------------------|-----------------------------------------------------|----------------------------------------------------|
| Server name                                          | `cedar-mysql-vm.int.cedar.computecanada.ca`         | `cedar-pgsql-vm.int.cedar.computecanada.ca`        | `cc-gra-dbaas1.sharcnet.ca`                       |
| IP                                                   | 199.241.163.99                                      |                                                     |                                                    |
| Short server name                                    | `cedar-mysql-vm`                                   | `cedar-pgsql-vm`                                  | N/A                                               |
| Latest version                                       | MariaDB version 11.5                               | PostgreSQL version 16, PostGIS version 3.3 extension | MariaDB version 11.5                               |
| Documentation                                        | [MariaDB website](link_to_mariadb_website)         | [Postgres website](link_to_postgres_website), [PostGIS documentation](link_to_postgis_docs) | [MariaDB website](link_to_mariadb_website)         |


## Cedar MySQL Server

The Cedar MySQL server offers MariaDB, compatible with other MySQL flavors. For compatibility information, see [MariaDB versus MySQL Compatibility](link_to_compatibility).

The MariaDB server runs as a VM called "cedar-mysql-vm" (full name: `cedar-mysql-vm.int.cedar.computecanada.ca`). Users can connect only through the Cedar head node (`cedar.computecanada.ca`), Cedar compute nodes, and the Cedar Portal.  For security, direct SSH connections to the database server are not allowed.


### MySQL Account and Connection

To create your own database, you need a MySQL account. Request an account from [Technical support](link_to_technical_support) with:

*   Your Alliance username
*   Required database space

An account with your Alliance username and a 16-digit random password will be created.  Connection details are stored in a `.my.cnf` file in your home directory. This file is confidential; you can read or delete it, but deleting it removes database access.

To connect, load a recent client version:

```bash
module load mariadb
mariadb --version
```

Test your account:

```sql
mysql
SHOW DATABASES;
quit
```

Do *not* use `-p` or `-h` options; the `.my.cnf` file provides necessary information.

Long-running SQL commands from the head node are acceptable (most CPU usage is on the database server).  CPU-intensive scripts should be submitted as jobs to the scheduler ([Running jobs](link_to_running_jobs)).


### Set Up Your MySQL Database

To create a database, the name must start with "username_", where "username" is your MySQL username. For example, for username "david", create a database like "david_db1":

```sql
mysql
CREATE DATABASE david_db1;
quit
```

You can create multiple databases, all starting with "username_". Created databases are accessible only to you from the Cedar head node, Cedar compute nodes, and the Cedar Portal. You have full privileges to create tables, views, etc.


### Work with Your MySQL Database

For account "david" and database "david_db1":

```sql
mysql
SHOW DATABASES; -- List available databases. Confirm david_db1 is in the list
USE david_db1; -- Get into the database
... -- Issue SQL commands. See below for information.
quit
```

MariaDB resources:

*   [MariaDB Knowledgebase](link_to_mariadb_kb)
*   [MariaDB Training and Tutorials](link_to_mariadb_tutorials)
*   [MariaDB SQL Statement Structure](link_to_mariadb_sql_structure)
*   [MariaDB Optimization and Indexes](link_to_mariadb_optimization)


### Share Your MySQL Data

To share a table with other users:

1.  Connect to MySQL using `mysql`.
2.  Use the database: `USE database;`
3.  Grant privileges: `GRANT priv_type ON mytable TO 'user'@'172.%';`  (priv\_type = privilege type, mytable = table name, user = username)


#### MySQL Sharing Example

Username "david" shares "mytable" from "david_db" with "john" (read-only):

```sql
mysql
USE david_db;
GRANT SELECT on mytable to 'john'@'172.%';
quit;
```


## Cedar PostgreSQL Server

The Cedar Postgres server includes Postgres and the PostGIS extension.

The server runs as a VM called "cedar-pgsql-vm" (`cedar-pgsql-vm.int.cedar.computecanada.ca`). Users connect through the Cedar head node (`cedar.computecanada.ca`), Cedar compute nodes, and the Cedar Portal. Direct SSH connections are not allowed.

Request an account and database from [Technical support](link_to_technical_support), providing:

*   Your Alliance username
*   Required database space
*   Whether you need PostGIS


### PostgreSQL Account and Connection

Your account will use your Alliance username.  The database name is typically "<username>\_db". You cannot create databases yourself; contact [Technical support](link_to_technical_support) for more than one.  No password is required; your Alliance password is never used for security reasons.


Load a recent client version:

```bash
module load postgresql
psql --version
```


### Work with Your PostgreSQL Database

For account "david" and database "david_db":

```sql
psql -h cedar-pgsql-vm -d david_db
\d t -- List table names
... -- Issue SQL commands
\q -- Quit
```

PostgreSQL resources:

*   [PostgreSQL tutorials](link_to_postgres_tutorials)
*   [PostgreSQL manuals](link_to_postgres_manuals)
*   [PostgreSQL release notes](link_to_postgres_release_notes)


### Share Your PostgreSQL Data

To share data:

1.  The recipient needs a Postgres account.
2.  Grant `connect` access to your database.
3.  Grant `select`, `update`, `insert`, and/or `delete` access to specific tables/views.
4.  Revoke access as needed.


Example (user 'david' sharing with 'kim'):

```sql
psql -h cedar-pgsql-vm -d david_db
grant connect on database david_db to kim;
grant select on mytable to kim;
\q
```

'kim' accessing the shared table:

```sql
psql -h cedar-pgsql-vm -d kim_db
\c david_db
select * from mytable;
\q
```

Revoking access ('david' revoking 'kim's access):

```sql
psql -h cedar-pgsql-vm -d david_db
revoke select on mytable from kim;
revoke connect on database david_db from kim;
\q
```


## Graham MySQL Server

Instructions are similar to those for the Cedar MySQL server.


## Cloud-based Database Servers

### Database as a Service (DBaaS)

For larger database loads, consider a managed database (MySQL/MariaDB and Postgres).  Daily backups are archived for 3 months.  Request access from [Technical support](link_to_technical_support), providing your client network or IP address.

| Type    | Hostname                                         | TCP port |
|---------|-------------------------------------------------|----------|
| mysql   | `dbaas101.arbutus.cloud.computecanada.ca`       | 3306     |
| pgsql   | `dbaas101.arbutus.cloud.computecanada.ca`       | 5432     |

The CA certificate is available for download [here](link_to_ca_certificate).


### PostgreSQL Database

SSL connections are required. Example (user `user01`, database `dbinstance`):

```bash
psql --set "sslmode=require" -h dbaas101.arbutus.cloud.computecanada.ca -U user01 -d dbinstance
```


### MariaDB/MySQL Database

SSL connections are required. Example (user `user01`, database `dbinstance`):

```bash
mysql --ssl -h dbaas101.arbutus.cloud.computecanada.ca -u user01 -p dbinstance
```

Plain text connections fail.


**(Remember to replace placeholder links like `link_to_details` with actual links.)**
