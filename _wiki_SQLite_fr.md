# SQLite

SQLite is a database engine for building so-called "pocket" databases.  It offers all the functionalities of relational databases without the client-server architecture.  The advantage is that all data resides in a single disk file, which can be copied to another computer. Applications written in several well-known languages can read and write to an SQLite file using standard SQL queries via their database interaction API.

SQLite databases, like all others, should not be used in shared file systems such as `/home`, `/scratch`, and `/project`. At the beginning of a task, you should generally copy the SQLite file to the local `/scratch` space (`$SLURM_TMPDIR`), where you can use the database without problems while benefiting from the best performance. Note that SQLite does not provide for the use of multiple threads or processes writing to the database at the same time; to do this, you should use a client-server solution.


## Using SQLite Directly

You can also directly access an SQLite database using the native client:

```bash
[name@server ~]$ sqlite3 foo.sqlite
```

If the file `foo.sqlite` does not exist, SQLite creates it, and this client starts in an empty database; otherwise, you are connected to the existing database. You can then execute any queries, for example, run `SELECT * FROM tablename;` to display the contents of `tablename` on the screen.


## Accessing SQLite from an Application

The usual way to interact with an SQLite database (or any other) is to use function calls to establish the connection; execute read, write, or update data queries; and close the connection so that the changes are saved to the SQLite disk file. In the simple example shown below, we assume that the database already exists and contains the `employee` table with two columns: the string `name` and the integer `age`.

### Python

```python
#!/usr/bin/env python3
# For Python we can use the module sqlite3, installed in a virtual environment,
# to access an SQLite database
import sqlite3

age = 34
# Connect to the database...
dbase = sqlite3.connect("foo.sqlite")
dbase.execute("INSERT INTO employee(name,age) VALUES(\"John Smith\"," + str(age) + ");")
# Close the database connection
dbase.close()
```

### R

```r
# Using R, the first step is to install the RSQLite package in your R environment,
# after which you can use code like the following to interact with the SQLite database
library(DBI)
age <- 34
# Connect to the database...
dbase <- dbConnect(RSQLite::SQLite(), "foo.sqlite")
# A parameterized query
query <- paste(c("INSERT INTO employee(name,age) VALUES(\"John Smith\",", toString(age), ");"), collapse = '')
dbExecute(dbase, query)
# Close the database connection
dbDisconnect(dbase)
```

### C++

```c++
#include <iostream>
#include <string>
#include <sqlite3.h>

int main(int argc, char **argv) {
  int age = 34;
  std::string query;
  sqlite3 *dbase;
  sqlite3_open("foo.sqlite", &dbase);
  query = "INSERT INTO employee(name,age) VALUES(\"John Smith\"," + std::to_string(age) + ");";
  sqlite3_exec(dbase, query.c_str(), nullptr, nullptr, nullptr);
  sqlite3_close(dbase);
  return 0;
}
```


## Limitations

As its name suggests, SQLite is easy to use and designed for relatively simple databases whose size does not exceed a few hundred GB and whose entity-relationship model is not too complex. As the size and complexity of your database increase, you may notice a decrease in performance; if this is the case, it would be time to find a more sophisticated client-server tool. You will find on the SQLite website selection criteria between SQLite and client-server DBMSs.


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=SQLite/fr&oldid=80836](https://docs.alliancecan.ca/mediawiki/index.php?title=SQLite/fr&oldid=80836)"
