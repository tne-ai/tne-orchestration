from v2.api.api import RagAnn, RagMetrics
from v2.app.nn import nn
from v2.app.postgres_util import retrieve_related_texts
from v2.app.state import State


async def retrieve_anns(state: State) -> None:
    anns_input_text = nn(nn(state.updated_record.anns_input).text)
    rows = await retrieve_related_texts(
        state.config.embeddings_config,
        anns_input_text,
        state.config.ann_retrieval_count,
    )
    ann_count_after_retrieval = len(rows)
    rows = [row for row in rows if row[2] >= state.config.ann_similarity_min_value]
    ann_count_after_similarity_min_value = len(rows)
    rows = sorted(rows, key=lambda row: row[2], reverse=True)
    rows = rows[: state.config.ann_similarity_max_count]
    ann_count_after_similarity_max_count = len(rows)

    anns = {
        embedding_id.hex(): RagAnn(
            embedding_id=embedding_id.hex(), text=text, similarity=similarity
        )
        for embedding_id, text, similarity in rows
    }

    await state.put_record(
        anns=anns,
        metrics=RagMetrics(
            ann_count_after_retrieval=ann_count_after_retrieval,
            ann_count_after_similarity_min_value=ann_count_after_similarity_min_value,
            ann_count_after_similarity_max_count=ann_count_after_similarity_max_count,
        ),
    )
    await state.put_internal_info_event("Completed: retrieve_anns")
