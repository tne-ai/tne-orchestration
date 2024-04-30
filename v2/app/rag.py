import asyncio
import logging

from v2.api.api import (
    EmbeddingId,
    RagRecord,
    RagRequest,
    RagResponse,
    RequestId,
    ThreadId,
)
from v2.api.util import merge_records
from v2.app.generate_ann_evaluation import generate_ann_evaluation
from v2.app.generate_ann_relevancy import generate_ann_relevancy
from v2.app.generate_anns_input import generate_anns_input
from v2.app.generate_anns_summary import generate_anns_summary
from v2.app.generate_full_history_summary import generate_full_history_summary
from v2.app.generate_prev_record_summary import generate_prev_record_summary
from v2.app.generate_rag_output import generate_rag_output
from v2.app.nn import nn
from v2.app.retrieve_anns import retrieve_anns
from v2.app.state import State

# TODO(Guy): Handle client cancellation (e.g. connection closed).
# TODO(Guy): Configurable timeouts. Both duration and critical-ness.


logger = logging.getLogger()


async def batch_merge_patch_records_into_responses(
    thread_id: ThreadId,
    request_id: RequestId,
    patch_records_queue: asyncio.Queue,
    responses_queue: asyncio.Queue,
) -> None:
    patch_record: RagRecord
    while (patch_record := await patch_records_queue.get()) is not None:

        batched_patch_records = [patch_record]
        for _ in range(patch_records_queue.qsize()):
            batched_patch_records.append(patch_records_queue.get_nowait())

        assert all(record for record in batched_patch_records[:-1])
        got_none = False
        if batched_patch_records[-1] is None:
            batched_patch_records = batched_patch_records[:-1]
            got_none = True

        merged_record = RagRecord(request_id=request_id)
        for patch_record in batched_patch_records:
            merged_record = merge_records(merged_record, patch_record)

        response = RagResponse(thread_id=thread_id, patch_record=merged_record)
        await responses_queue.put(response)
        for _ in batched_patch_records:
            patch_records_queue.task_done()

        if got_none:
            break

        sleep_seconds_between_batches = 0.2  # TODO(Guy): Make configurable.
        await asyncio.sleep(sleep_seconds_between_batches)  # Promotes more batching.

    await responses_queue.put(None)
    patch_records_queue.task_done()  # For that terminating None value.


async def generate_ann_evaluation_and_relevancy(
    state: State, embedding_id: EmbeddingId
) -> None:
    await generate_ann_evaluation(state, embedding_id)
    await generate_ann_relevancy(state, embedding_id)


async def generate_ann_evaluations_and_relevancies(state: State) -> None:
    anns = nn(state.updated_record.anns)
    ann_evaluation_tasks = [
        asyncio.create_task(generate_ann_evaluation_and_relevancy(state, embedding_id))
        for embedding_id in anns.keys()
    ]
    await asyncio.gather(*ann_evaluation_tasks)


async def rag(request: RagRequest, responses_queue: asyncio.Queue) -> None:
    records_queue_max_size = min(100_000, 10 * responses_queue.maxsize)
    patch_records_queue: asyncio.Queue = asyncio.Queue(maxsize=records_queue_max_size)
    batch_merge_patch_records_into_responses_task = asyncio.create_task(
        batch_merge_patch_records_into_responses(
            request.thread_id,
            request.new_record.request_id,
            patch_records_queue,
            responses_queue,
        )
    )

    state = State(request, patch_records_queue)

    try:
        generate_prev_record_summary_task = asyncio.create_task(
            generate_prev_record_summary(state)
        )
        generate_full_history_summary_task = asyncio.create_task(
            generate_full_history_summary(state)
        )
        await generate_anns_input(state)
        await retrieve_anns(state)
        await generate_ann_evaluations_and_relevancies(state)
        generate_anns_summary_task = asyncio.create_task(generate_anns_summary(state))
        await generate_rag_output(state)
        await generate_prev_record_summary_task
        await generate_full_history_summary_task
        await generate_anns_summary_task

    except Exception as exception:
        message = f'Caught exception: "{exception}", type: "{type(exception)}"'
        logger.exception(message)
        await state.put_internal_error_event(message)

    await state.finalize()
    await batch_merge_patch_records_into_responses_task
