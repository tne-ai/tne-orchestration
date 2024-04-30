import hashlib
import itertools

import numpy as np
from psycopg import sql

from v2.etl.db_util import apply_to_db
from v2.etl.json_util import read_json
from v2.etl.typer_util import typer_arg, typer_opt


def embedding_source_tuples_generator(input_json_files, validate_embedding):
    for input_json_file in input_json_files:
        doc_obj = read_json(input_json_file)
        doc_text = doc_obj["doc_text"]
        for chunk_index, chunk in enumerate(doc_obj["chunks"]):
            source = f"{input_json_file}:{chunk_index}"
            start_index, end_index = chunk["doc_text_range"]
            text = doc_text[start_index:end_index]
            id_ = hashlib.sha256(text.encode("utf-8")).digest()
            embedding = chunk["embedding"]
            if validate_embedding:
                assert len(embedding) == 1536
                assert np.isclose(np.linalg.norm(embedding), 1.0)
            yield id_, text, embedding, source


def _insert_embeddings_and_sources_db_fun(
    connection, cursor, embedding_source_tuples, batch_size
):
    total_embedding_source_tuples_count = 0
    total_inserted_embeddings_rowcount = 0
    total_inserted_sources_rowcount = 0

    while batch_embedding_source_tuples := list(
        itertools.islice(embedding_source_tuples, batch_size)
    ):
        total_embedding_source_tuples_count += len(batch_embedding_source_tuples)

        batch_embedding_tuples = [
            (id_, text, embedding)
            for id_, text, embedding, _ in batch_embedding_source_tuples
        ]
        # TODO: On conflict, update instead of nothing.
        insert_embeddings_sql = """
            insert into embeddings(id, text_, embedding)
            values (%s, %s, %s)
            on conflict (id) do nothing
        """
        cursor.executemany(insert_embeddings_sql, batch_embedding_tuples)
        inserted_embeddings_rowcount = cursor.rowcount
        total_inserted_embeddings_rowcount += inserted_embeddings_rowcount
        connection.commit()
        print(
            f"Inserted batch of {len(batch_embedding_tuples)} embeddings. Inserted rowcount: {inserted_embeddings_rowcount}"
        )

        batch_source_tuples = [
            (source, id_) for id_, _, _, source in batch_embedding_source_tuples
        ]
        # TODO: On conflict, update instead of nothing.
        insert_sources_sql = """
            insert into sources(source, embedding_id)
            values (%s, %s)
            on conflict (source) do nothing
        """
        cursor.executemany(insert_sources_sql, batch_source_tuples)
        inserted_sources_rowcount = cursor.rowcount
        total_inserted_sources_rowcount += inserted_sources_rowcount
        connection.commit()
        print(
            f"Inserted batch of {len(batch_source_tuples)} sources. Inserted rowcount: {inserted_sources_rowcount}"
        )

    print("-" * 80)
    print(f"total_embedding_source_tuples_count: {total_embedding_source_tuples_count}")
    print(f"total_inserted_embeddings_rowcount: {total_inserted_embeddings_rowcount}")
    print(f"total_inserted_sources_rowcount: {total_inserted_sources_rowcount}")


def insert_embeddings_and_sources(
    db_config_file: typer_arg(str, "TODO: Add help text."),
    input_json_files_file: typer_arg(str, "TODO: Add help text."),
    batch_size: typer_opt(int, "TODO: Add help text.") = 1_000,
    validate_embeddings: typer_opt(bool, "TODO: Add help text.") = True,
):
    """TODO: Add help text."""

    with open(input_json_files_file) as fp:
        input_json_files = [line.strip() for line in fp.readlines()]

    embedding_source_tuples = embedding_source_tuples_generator(
        input_json_files, validate_embeddings
    )

    apply_to_db(
        db_config_file,
        _insert_embeddings_and_sources_db_fun,
        embedding_source_tuples,
        batch_size,
    )


# This implementation was supposed to be faster but was a tad slower. Hmmm...
def _alternate_insert_embeddings_and_sources(
    db_config_file: typer_arg(str, "TODO: Add help text."),
    input_json_files_file: typer_arg(str, "TODO: Add help text."),
    batch_size: typer_opt(int, "TODO: Add help text.") = 1_000,
    validate_embeddings: typer_opt(bool, "TODO: Add help text.") = True,
):
    """TODO: Add help text."""

    with open(input_json_files_file) as fp:
        input_json_files = [line.strip() for line in fp.readlines()]

    embedding_source_tuples = embedding_source_tuples_generator(
        input_json_files, validate_embeddings
    )

    def apply_db_fun(connection, cursor):
        while batch_embedding_source_tuples := list(
            itertools.islice(embedding_source_tuples, batch_size)
        ):
            # TODO: On conflict, update instead of nothing.
            sql_template = """
                insert into embeddings (id, text_, embedding)
                values {values}
                on conflict (id) do nothing
            """
            values_composed = sql.SQL(", ").join(
                sql.SQL("({}, {}, {})").format(
                    sql.Literal(id_), sql.Literal(text), sql.Literal(embedding)
                )
                for id_, text, embedding, _ in batch_embedding_source_tuples
            )
            sql_composed = sql.SQL(sql_template).format(values=values_composed)
            cursor.execute(sql_composed)
            connection.commit()

            # TODO: On conflict, update instead of nothing.
            sql_template = """
                insert into sources (source, embedding_id)
                values {values}
                on conflict (source) do nothing
            """
            values_composed = sql.SQL(", ").join(
                sql.SQL("({}, {})").format(sql.Literal(source), sql.Literal(id_))
                for id_, _, _, source in batch_embedding_source_tuples
            )
            sql_composed = sql.SQL(sql_template).format(values=values_composed)
            cursor.execute(sql_composed)
            connection.commit()

    apply_to_db(db_config_file, apply_db_fun)
