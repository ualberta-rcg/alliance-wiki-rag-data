# SQLite

SQLite is a database management tool used to build commonly called "pocket databases" because they offer all the features of relational databases without the client-server architecture.  A key advantage is that the data resides on a single disk file, easily copied to another computer. Software written in various languages can read and write to the database file using standard SQL queries via the language's database interaction API.

Like any database, an SQLite database shouldn't be used on a shared filesystem (e.g., home, scratch, project).  It's best to copy your SQLite file to the local scratch `$SLURM_TMPDIR` space at the beginning of a job. This ensures optimal performance and avoids issues.  Note that SQLite isn't designed for concurrent writing from multiple threads or processes; for this, a client-server solution is recommended.

## Using SQLite Directly

Access an SQLite database directly using the native client:

```bash
sqlite3 foo.sqlite
```

If `foo.sqlite` doesn't exist, SQLite creates it, starting the client in an empty database. Otherwise, it connects to the existing database.  Execute queries like:

```sql
SELECT * FROM tablename;
```

This prints `tablename`'s contents to the screen.


## Accessing SQLite from Software

Typically, you interact with an SQLite (or other) database via function calls: open a connection, execute queries (reading, inserting, updating data), and close the connection to flush changes to the SQLite file.  The examples below assume a database with a table named `employee`, containing `name` (string) and `age` (integer) columns.


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

```cpp
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

SQLite is easy to use and suitable for relatively simple databases—not excessively large (hundreds of gigabytes or more) or complex in terms of their entity-relationship diagram.  Performance may degrade as the database grows; consider more sophisticated client-server database software in such cases. The SQLite website offers guidance on appropriate uses for SQLite, including a checklist for choosing between SQLite and client-server databases.


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=SQLite&oldid=80813](https://docs.alliancecan.ca/mediawiki/index.php?title=SQLite&oldid=80813)"
