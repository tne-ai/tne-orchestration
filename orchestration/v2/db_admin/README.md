# RAG v2 DB admin scripts.

These are temporary scripts for creating, configuring, and deleting embedding databases for RAG v2. This is a very manual CLI process because these scripts are adapted from v1 (which only ever had a single DB). This is for tempoarary use until this CLI process is replaced by a proper ETL GUI.

## Setting up.

The following commands expect the `RAG_DB_PASSWORD` env var to be set. The password is given here because it has already been committed several times across various repos (including this one).

```sh
export RAG_DB_PASSWORD=i6XFDgR6KGkTd
cd your/tne/root/troopship/rag
```

## Creating a new RAG embeddings DB:

### Check for rag_v2_* DB names that have already been created.

```sh
./v2/db_admin/psql_exec.sh \
    file=./v2/db_admin/sql/dbnone/list_databases.sql \
    | egrep rag_v2_
```

### Create a new RAG embeddings DB for a particular (e.g. "foobar") document set.

```sh
./v2/db_admin/psql_exec.sh \
    file=./v2/db_admin/sql/dbset/create_database.sql \
    dbset=rag_v2_foobar
# Expected output:
# CREATE DATABASE
```

### Create necessary `pgcrypto` and `vector` extensions.

```sh
./v2/db_admin/psql_exec.sh \
    file=./v2/db_admin/sql/dbname/create_extensions.sql \
    dbname=rag_v2_foobar
# Expected output:
# CREATE EXTENSION
# CREATE EXTENSION
```

### Create `embeddings` and `sources` tables.

```sh
./v2/db_admin/psql_exec.sh \
    file=./v2/db_admin/sql/dbname/create_tables.sql \
    dbname=rag_v2_foobar
# Expected output:
# CREATE TABLE
# CREATE TABLE
```

### Run python ETL tools.

At this point the new DB is ready for loading. [See ETL steps here.](../etl/README.md)

### Confirm expected number of rows in both tables.

```sh
./v2/db_admin/psql_exec.sh \
    file=./v2/db_admin/sql/dbname/count_rows.sql \
    dbname=rag_v2_foobar
# Expected output:
#  embeddings_count 
# ------------------
#               ???
# (1 row)
# 
#  sources_count 
# ---------------
#            ???
# (1 row)
```

### After ETL create vector index.

For optimal indexing, make sure all data is loaded first!

```sh
./v2/db_admin/psql_exec.sh \
    file=./v2/db_admin/sql/dbname/create_index.sql \
    dbname=rag_v2_foobar
# Expected output:
# CREATE INDEX
```

### Monitor index creation if you're curious.

```sh
./v2/db_admin/psql_exec.sh \
    file=./v2/db_admin/sql/dbname/monitor_create_index_progress.sql \
    dbname=rag_v2_foobar
# Expected output:
#  phase | tuples_done | tuples_total 
# -------+-------------+--------------
# ...
```

### Delete everything if needed.

```sh
./v2/db_admin/psql_exec.sh \
    file=./v2/db_admin/sql/dbname/drop_index.sql \
    dbname=rag_v2_foobar
# Expected output:
# DROP INDEX

./v2/db_admin/psql_exec.sh \
    file=./v2/db_admin/sql/dbname/drop_tables.sql \
    dbname=rag_v2_foobar
# Expected output:
# DROP TABLE
# DROP TABLE

./v2/db_admin/psql_exec.sh \
    file=./v2/db_admin/sql/dbset/drop_database.sql \
    dbset=rag_v2_foobar
# Expected output:
# DROP DATABASE
```
