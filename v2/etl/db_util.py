import os

import dotenv
import psycopg


def apply_to_db(db_config_file, apply_db_fun, *args, **kw_args):
    dotenv.load_dotenv(db_config_file)
    with psycopg.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USERNAME"),
        password=os.getenv("POSTGRES_PASSWORD"),
    ) as connection:
        with connection.cursor() as cursor:
            return apply_db_fun(connection, cursor, *args, **kw_args)


def execute_sql(db_config_file, sql):
    def apply_db_fun(connection, cursor):
        cursor.execute(sql)
        connection.commit()
        print(f"cursor.statusmessage: {cursor.statusmessage}")
        print(f"cursor.rowcount: {cursor.rowcount}")

    apply_to_db(db_config_file, apply_db_fun)
