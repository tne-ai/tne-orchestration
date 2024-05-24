from v2.api.api import AnnsRequest, AnnsResponse, RagAnn, RagAnnSource
from v2.app.postgres_util import get_paths_for_embedding_ids, retrieve_related_texts_2


async def query_anns(request: AnnsRequest) -> AnnsResponse:

    embedding_rows = await retrieve_related_texts_2(
        request.embeddings_config,
        request.query_text,
        request.max_count,
        request.min_similarity,
    )

    embedding_ids = set(embedding_id for embedding_id, _, _ in embedding_rows)
    paths_for_embedding_ids = await get_paths_for_embedding_ids(
        request.embeddings_config, embedding_ids
    )

    anns = [
        RagAnn(
            embedding_id=embedding_id,
            text=text,
            similarity=similarity,
            sources=[
                RagAnnSource(file_path=path)
                for path in paths_for_embedding_ids[embedding_id]
            ],
        )
        for embedding_id, text, similarity in embedding_rows
    ]

    return AnnsResponse(request_id=request.request_id, anns=anns)
