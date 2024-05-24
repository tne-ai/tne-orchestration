from v2.api.api import RagAnn, RagAnnSource
from v2.app.nn import nn
from v2.app.postgres_util import get_paths_for_embedding_ids
from v2.app.state import State


async def retrieve_ann_sources(state: State) -> None:
    anns = nn(state.updated_record.anns)
    embedding_ids = set(anns.keys())
    paths_for_embedding_ids = await get_paths_for_embedding_ids(
        state.config.embeddings_config, embedding_ids
    )
    anns = {
        embedding_id: RagAnn(
            sources=[
                RagAnnSource(file_path=path)
                for path in paths_for_embedding_ids[embedding_id]
            ]
        )
        for embedding_id in embedding_ids
    }
    await state.put_record(anns=anns)
    await state.put_internal_info_event("Completed: retrieve_ann_sources")
