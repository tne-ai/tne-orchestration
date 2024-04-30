from typing import List, Tuple

import psycopg
from psycopg import sql

from v2.api.api import EmbeddingsConfig
from v2.app.nn import nn
from v2.app.openapi_util import call_embedding


# The schema for the embeddings table is here:
# TODO(Guy): Update this URL when new DB scripts added:
# https://github.com/TNE-ai/cloud/blob/main/services/crag-agent/postgres_admin/sql/create_tables.sql
# Some relevant points about cosine similarity:
# * similarity(x, y) = dot(x, y) / (norm(x) * norm(y))
# * For any openai_ada_002 embedding, norm(x) = 1
# * So, similarity(x, y) = dot(x, y)
# * pgvector operator <#> computes -dot(x, y)
#
# Here is how the returned rows might be used:
# for id_, text, similarity in rows:
#     ...
async def retrieve_related_texts(
    embeddings_config: EmbeddingsConfig, text: str, count: int
) -> List[Tuple[bytes, str, float]]:
    embedding_model = nn(embeddings_config.model)
    embedding_api_key = nn(embeddings_config.api_key)
    embedding = await call_embedding(embedding_model, embedding_api_key, text)
    connection = await psycopg.AsyncConnection.connect(
        host=embeddings_config.db_host,
        port=embeddings_config.db_port,
        dbname=embeddings_config.db_name,
        user=embeddings_config.db_username,
        password=embeddings_config.db_password,
    )
    async with connection:
        async with connection.cursor() as cursor:
            sql_template = """
                select id, text_, -1 * (embedding <#> {embedding}::vector)
                from embeddings
                order by embedding <#> {embedding}::vector
                limit {count}
            """
            sql_composed = sql.SQL(sql_template).format(
                embedding=sql.Literal(embedding), count=sql.Literal(count)
            )
            await cursor.execute(sql_composed)
            rows = await cursor.fetchall()
            return rows
