import re
from typing import Dict, List, Set, Tuple

import psycopg
from psycopg import sql

from v2.api.api import EmbeddingsConfig
from v2.app.nn import nn
from v2.app.openai_util import call_embedding, call_embedding_2
from v2.app.state import State

# The schema for the embeddings table is here:
# https://github.com/TNE-ai/troopship/blob/main/rag/v2/db_admin/sql/dbname/create_tables.sql

# Some relevant points about cosine similarity:
# * similarity(x, y) = dot(x, y) / (norm(x) * norm(y))
# * For any openai_ada_002 embedding, norm(x) = 1
# * So, similarity(x, y) = dot(x, y)
# * pgvector operator <#> computes -dot(x, y)

# Here is how the returned rows might be used:
# for id_, text, similarity in rows:
#     ...


async def _execute_sql(
    embeddings_config: EmbeddingsConfig, sql_composed: sql.Composed
) -> List[Tuple]:
    connection = await psycopg.AsyncConnection.connect(
        host=embeddings_config.db_host,
        port=embeddings_config.db_port,
        dbname=embeddings_config.db_name,
        user=embeddings_config.db_username,
        password=embeddings_config.db_password,
    )
    async with connection:
        async with connection.cursor() as cursor:
            await cursor.execute(sql_composed)
            rows = await cursor.fetchall()
            return rows


async def retrieve_related_texts(
    state: State, text: str
) -> List[Tuple[str, str, float]]:
    embeddings_config = state.config.embeddings_config
    count = state.config.ann_retrieval_count
    embedding_model = nn(embeddings_config.model)
    embedding_api_key = nn(embeddings_config.api_key)
    embedding = await call_embedding(state, embedding_model, embedding_api_key, text)
    # TODO(Guy): Add "where cosine_similarity >= {min_cosine_similarity}"
    # Like retrieve_related_texts_2 does below.
    sql_template = """
        select
            encode(id, 'hex') as id,
            text_,
            -1 * (embedding <#> {embedding}::vector)
        from embeddings
        order by embedding <#> {embedding}::vector
        limit {count}
    """
    sql_composed = sql.SQL(sql_template).format(
        embedding=sql.Literal(embedding), count=sql.Literal(count)
    )
    return await _execute_sql(embeddings_config, sql_composed)


async def retrieve_related_texts_2(
    embeddings_config: EmbeddingsConfig,
    query_text: str,
    max_count: int,
    min_similarity: float,
) -> List[Tuple[str, str, float]]:
    embedding_model = nn(embeddings_config.model)
    embedding_api_key = nn(embeddings_config.api_key)
    embedding = await call_embedding_2(embedding_model, embedding_api_key, query_text)
    sql_template = """
        with cosine_similarities as (
            select
                encode(id, 'hex') as id,
                text_,
                -1 * (embedding <#> {embedding}::vector) as cosine_similarity
            from embeddings
        )
        select id, text_, cosine_similarity
        from cosine_similarities
        where cosine_similarity >= {min_cosine_similarity}
        order by cosine_similarity desc
        limit {count}
    """
    sql_composed = sql.SQL(sql_template).format(
        embedding=sql.Literal(embedding),
        min_cosine_similarity=sql.Literal(min_similarity),
        count=sql.Literal(max_count),
    )
    return await _execute_sql(embeddings_config, sql_composed)


async def get_paths_for_embedding_ids(
    embeddings_config: EmbeddingsConfig, embedding_ids: Set[str]
) -> Dict[str, List[str]]:
    paths_for_embedding_ids: Dict[str, List[str]] = {
        embedding_id: [] for embedding_id in embedding_ids
    }
    sql_template = """
        select encode(embedding_id, 'hex') as embedding_id, source
        from sources
        where embedding_id = any(array[{embedding_ids}])
    """
    sql_composed = sql.SQL(sql_template).format(
        embedding_ids=sql.SQL(", ").join(
            sql.Literal(bytes.fromhex(id)) for id in embedding_ids
        )
    )
    rows = await _execute_sql(embeddings_config, sql_composed)
    pattern = re.compile("[.]json:[0-9]+$")
    for embedding_id, source in rows:
        path = re.sub(pattern, ".pdf", source)
        paths_for_embedding_ids[embedding_id].append(path)
    return paths_for_embedding_ids
