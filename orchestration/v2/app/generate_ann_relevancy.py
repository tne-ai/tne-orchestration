import re
from typing import List

from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from v2.api.api import EmbeddingId, LlmOutput, RagAnn, RagRecord
from v2.app.messages_util import (
    create_history_messages_without_compression,
    system_message,
)
from v2.app.nn import nn
from v2.app.openai_util import call_chat_completion
from v2.app.state import State


# TODO(Guy): How to make all this message formatting more configurable via the API?
def _create_messages(
    state: State, embedding_id: EmbeddingId
) -> List[ChatCompletionMessageParam]:

    messages = create_history_messages_without_compression(state.request.history)

    user_input = nn(state.updated_record.user_input)
    ann_text = nn(state.updated_record.anns)[embedding_id].text
    ann_evaluation = nn(state.updated_record.anns)[embedding_id].evaluation

    messages.extend(
        [
            system_message(
                f"""
                Here is the "LATEST USER INPUT":
                {user_input}
                """
            ),
            system_message(
                f"""
                And here is some "ADDITIONAL CONTENT" that may be relevant to the "LATEST USER INPUT":
                {ann_text}
                """
            ),
            system_message(
                f"""
                Here is an evaluation of how the "ADDITIONAL CONTENT" may be relevant to the "LATEST USER INPUT":
                {ann_evaluation}
                """
            ),
            system_message(
                f"""
                Please rate the relevancy of the "ADDITIONAL CONTENT" given above.
                You should rate the relevancy as a real number between 0.0 and 1.0.
                If the "ADDITIONAL CONTENT" is not at all relevant to the "LATEST USER INPUT", then you should rate the relevancy as 0.0.
                If the "ADDITIONAL CONTENT" is highly relevant to the "LATEST USER INPUT", then you should rate the relevancy as 1.0.
                In no case should you provide a number less than 0.0 or greater than 1.0.
                There should be one and only one number in your reponse.
                Please provide the relevancy with as little text as possible.
                For example: "The relevancy is 0.83"
                """
            ),
            system_message(
                f"""
                Now, please rate the relevancy of the "ADDITIONAL CONTENT".
                """
            ),
        ]
    )

    return messages


async def _extract_relevancy(state: State, text: str) -> float:
    match = re.search("(1([.]0+)?|0([.][0-9]+)?)", text)
    if not match:
        await state.put_dependency_error_event(f"Invalid relevancy text: {text}")
        return 0.0
    if re.search("[0-9]", text[: match.start()]):
        await state.put_dependency_error_event(f"Invalid relevancy text: {text}")
        return 0.0
    if re.search("[0-9]", text[match.end() :]):
        await state.put_dependency_error_event(f"Invalid relevancy text: {text}")
        return 0.0
    relevancy = float(match.group(0))
    assert 0.0 <= relevancy <= 1.0
    return relevancy


async def generate_ann_relevancy(state: State, embedding_id: EmbeddingId) -> None:
    await call_chat_completion(
        state,
        "generate_ann_relevancy",
        state.config.ann_relevancies_llm_config,
        _create_messages(state, embedding_id),
        lambda llm_output: RagRecord(
            anns={
                embedding_id: RagAnn(
                    embedding_id=embedding_id,
                    raw_relevancy=llm_output,
                )
            }
        ),
    )

    text = nn(nn(nn(state.updated_record.anns)[embedding_id].raw_relevancy).text)
    relevancy = await _extract_relevancy(state, text)
    await state.put_record(
        anns={embedding_id: RagAnn(embedding_id=embedding_id, relevancy=relevancy)}
    )

    await state.put_internal_info_event(
        f"Completed: generate_ann_relevancy, embedding_id: {embedding_id}"
    )
