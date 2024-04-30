import re
from faker import Faker
from typing import Optional
from pydantic import BaseModel

from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from typing import Generator, List, Dict, Union
import pandas as pd
import sqlalchemy
from sqlalchemy import MetaData, Table, text


MAX_DISTINCT = 10


class TableColumn(BaseModel):
    """Table column."""

    name: str
    dtype: Union[str, None]


class ForeignKey(BaseModel):
    """Foreign key."""

    # Referenced column
    column: TableColumn
    # References table name
    references_name: str
    # References column
    references_column: TableColumn


class FormatterTable(BaseModel):
    """Table."""

    name: str
    columns: Union[List[TableColumn], None]
    pks: Union[List[TableColumn], None] = None
    fks: Union[List[ForeignKey], None] = None


class RajkumarFormatter:
    """RajkumarFormatter class.
    From https://arxiv.org/pdf/2204.00498.pdf.
    """

    table_sep: str = "\n\n"

    def __init__(
        self,
        tables: Dict,
        data: Dict,
        unique_vals: Optional[Dict] = None,
        use_unique_vals: Optional[bool] = False,
    ) -> None:
        self.data = data
        self.tables = tables
        self.unique_vals = unique_vals
        self.mappings = {"alias2real": {}, "real2alias": {}}

        # Random data generator for obfuscation
        self.fake = Faker()
        Faker.seed(4321)  # Set random seed

        # Formatted table schema
        self.table_str = self.get_formatted_tables(self.tables)

        # Formatted sample data
        self.data_str = self.get_formatted_data(self.data)

        # Formatted unique values for each column
        self.unique_vals_str = (
            self.get_formatted_unique_values(self.unique_vals)
            if use_unique_vals
            else None
        )

        self.aliased_table_str = self._replace_keywords(
            self.table_str, self.mappings.get("real2alias")
        )
        self.aliased_data_str = self._replace_keywords(
            self.data_str, self.mappings.get("real2alias")
        )
        self.aliased_unique_vals_str = self._replace_keywords(
            self.data_str, self.mappings.get("real2alias")
        )
        self.mappings_str = str(self.mappings.get("real2alias"))

    def get_formatted_unique_values(self, unique_values_dict: Dict) -> str:
        """Format the distinct values for each table in a readable manner."""
        formatted_tables = []

        for table_name, unique_values in unique_values_dict.items():
            formatted_values = []
            for column, values in unique_values.items():
                aliased_values = []
                # Compute value aliases
                for v in values:
                    if v in self.mappings.get("real2alias").keys():
                        aliased_values.append(self.mappings.get("real2alias").get(v))
                    else:
                        aliased_v = self._get_alias(v)
                        aliased_values.append(aliased_v)
                        self.mappings.get("alias2real")[aliased_v] = v
                        self.mappings.get("real2alias")[v] = aliased_v

                values_str = ", ".join(map(str, values))
                formatted_values.append(f"{column}: {values_str}")

            formatted_table = "\n".join(formatted_values)
            formatted_tables.append(formatted_table)
        return "\n\n".join(formatted_tables)

    def get_formatted_data(self, data: Dict) -> str:
        data_str = ""
        data_list = [d for d in data.values()]
        for i, table_data in enumerate(data_list):
            # Extract table name and rows
            table_name = table_data.get("table_name")
            sample_rows = table_data.get("sample_rows")
            aliased_rows = []
            for row in sample_rows:
                aliased_row = []
                row_data = row.split()
                for v in row_data:
                    v_alias = self._get_alias(v)
                    aliased_row.append(v_alias)
                    self.mappings.get("alias2real")[v_alias] = v
                    self.mappings.get("real2alias")[v] = v_alias
                aliased_rows.append(" ".join(aliased_row))

            if not table_name or not sample_rows:
                raise ValueError(f"Malformed data {data}")

            table_str = f"SAMPLE DATA FROM: {table_name}\n\n"
            for n, row in enumerate(sample_rows):
                if n < len(table_data) - 1:
                    table_str += row + "\n"
                else:
                    table_str += row

            if i < len(data) - 1:
                data_str += table_str + "\n\n"
            else:
                data_str += table_str

        return data_str

    def format_table(self, table: Table) -> str:
        """Get table format."""
        table_fmt = []

        # Get a fake table name
        table_alias = self.fake.unique.word()
        self.mappings.get("real2alias")[table.name] = table_alias
        self.mappings.get("alias2real")[table_alias] = table.name

        # Collect columns
        for col in table.columns or []:
            col_alias = self.fake.unique.word()
            self.mappings["alias2real"][col_alias] = col.name
            self.mappings["real2alias"][col.name] = col_alias
            table_fmt.append(f"    {col.name} {col.dtype or 'any'}")

        # NOTE: no current tables have foreign or primary keys, this code is untested
        if table.pks:
            table_fmt.append(
                f"    primary key ({', '.join(pk.name for pk in table.pks)})"
            )
        for fk in table.fks or []:
            table_fmt.append(
                f"    foreign key ({fk.column.name}) references {fk.references_name}({fk.references_column.name})"  # noqa: E501
            )

        if table_fmt:
            all_cols = ",\n".join(table_fmt)
            table_str = f"CREATE TABLE {table.name} (\n{all_cols}\n)"
        else:
            table_str = f"CREATE TABLE {table.name}"

        return table_str

    def get_formatted_tables(self, tables: Dict) -> str:
        """Get tables format."""

        table_list = [table for table in tables.values()]
        table_strings = []
        for table in table_list:
            table_fmt = self.format_table(table)
            table_strings.append(table_fmt)

        table_str = self.table_sep.join(table_strings)

        return table_str

    def pull_schema(self, aliased: Optional[bool] = False) -> str:
        """Get formatted schema string."""

        if aliased:
            table_str = self.aliased_table_str
            data_str = self.aliased_data_str
            unique_vals_str = self.aliased_unique_vals_str
        else:
            table_str = self.table_str
            data_str = self.data_str
            unique_vals_str = self.unique_vals_str

        db_data = f"Schema\n\n{table_str}\n\nSample data\n\n{data_str}\n\nUnique values\n\n{unique_vals_str}"
        if aliased:
            db_data += f"Data mappings\n\n{self.mappings_str}"

        return db_data

    def _infer_dtype(self, v) -> str:
        try:
            # Trying to convert to int
            int(v)
            return "int"
        except ValueError:
            try:
                # Trying to convert to float
                float(v)
                return "float"
            except ValueError:
                # If both conversions fail, it's a string
                return "str"
        except Exception as e:
            return "str"

    def _get_alias(self, v):
        data_type = self._infer_dtype(v)
        if data_type == "int":
            v_alias = str(self.fake.unique.random_int())
        elif data_type == "float":
            v_alias = str(self.fake.unique.random_int()) + ".0"
        elif data_type == "str":
            v_alias = self.fake.unique.word()
        else:
            raise NotImplementedError

        return v_alias

    import re

    def _replace_keywords(self, data: str, mappings: Dict[str, str]):
        # Regular expression pattern for words
        word_pattern = r"\b\w+\b"

        def replace_match(match):
            word = match.group(0)
            # Handle special case for columns that have a colon attached
            key = word
            suffix = ""

            return mappings.get(key, key) + suffix

        # Use re.sub to replace each word found by the pattern with its mapped value
        return re.sub(word_pattern, replace_match, data)


@dataclass
class PostgresConnector:
    """Postgres connection."""

    user: str
    password: str
    dbname: str
    host: str
    port: int

    @cached_property
    def pg_uri(self) -> str:
        """Get Postgres URI."""
        uri = (
            f"postgresql://"
            f"{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}"
        )
        engine = sqlalchemy.create_engine(uri)
        conn = engine.connect()

        # assuming the above connection is successful, we can now close the connection
        conn.close()
        engine.dispose()

        return uri

    @contextmanager
    def connect(self) -> Generator[sqlalchemy.engine.base.Connection, None, None]:
        """Yield a connection to a Postgres db."""
        try:
            engine = sqlalchemy.create_engine(self.pg_uri)
            conn = engine.connect()
            yield conn
        finally:
            conn.close()
            engine.dispose()

    def run_sql_as_df(self, sql: str) -> pd.DataFrame:
        """Run SQL statement."""
        with self.connect() as conn:
            return pd.read_sql(sql, conn)

    def get_tables(self) -> List[str]:
        """Get all tables in the database."""
        engine = sqlalchemy.create_engine(self.pg_uri)
        metadata = MetaData()
        metadata.reflect(bind=engine)
        table_names = metadata.tables.keys()
        engine.dispose()
        return table_names

    def select_random(
        self, table: str, num_selections: int = 3
    ) -> Dict[str, Union[str, list[str]]]:
        """Randomly select three rows of data for LLM prompting"""
        with self.connect() as conn:
            rows = []
            sql = f"""
                SELECT * FROM {table}
                ORDER BY RANDOM()
                LIMIT {num_selections};
            """
            db_rows = conn.execute(text(sql)).fetchall()
            for row in db_rows:
                # Convert non-string values to str
                row_str_list = [str(i) for i in row.tuple()]
                rows.append(" ".join(row_str_list))

        return {"table_name": table, "sample_rows": rows}

    def get_distinct_values(self, table: str) -> dict:
        """Get up to MAX_DISTINCT distinct values for each column of a table."""
        distinct_values = {}

        with self.connect() as conn:
            for column in self.get_schema(table).columns:
                sql = f"""
                    SELECT DISTINCT {column.name} FROM {table}
                    LIMIT {MAX_DISTINCT};
                """
                values = conn.execute(text(sql)).fetchall()
                distinct_values[column.name] = [value[0] for value in values]

        return distinct_values

    def get_schema(self, table: str) -> FormatterTable:
        """Return Table."""
        with self.connect() as conn:
            columns = []
            sql = f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = '{table}';
            """
            schema = conn.execute(text(sql)).fetchall()
            for col, type_ in schema:
                columns.append(TableColumn(name=col, dtype=type_))
            return FormatterTable(name=table, columns=columns)

    def insert_sql_logs(self, table_name: str, data: dict):
        """Insert a key/value pair."""

        # Check for expected keys
        expected_keys = ["input_text", "output_sql", "db_name"]
        if not all(key in expected_keys for key in data.keys()):
            raise ValueError(
                f"Invalid keys in data. Expected {', '.join(expected_keys)}."
            )

        columns_str = ", ".join(data.keys())
        placeholders = ", ".join([f":{key}" for key in data.keys()])

        # Check if table exists
        with self.connect() as conn:
            result = conn.execute(
                text(
                    f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = '{table_name}'
                );
            """
                )
            ).fetchone()
            table_exists = result[0]
            if not table_exists:
                conn.execute(
                    text(
                        f"""
                    CREATE TABLE {table_name} (
                        input_text TEXT NOT NULL,
                        output_sql TEXT NOT NULL,
                        db_name TEXT NOT NULL
                    );
                """
                    )
                )

            query = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"
            conn.execute(text(query), data)
