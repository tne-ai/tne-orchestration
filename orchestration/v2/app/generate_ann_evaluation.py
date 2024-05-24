from typing import List

from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from v2.api.api import EmbeddingId, RagAnn, RagRecord
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

    messages.extend(
        [
            system_message(
                f"""
                All of the messages above are the "PREVIOUS CONTENT".
                Here is the "LATEST USER INPUT":
                {user_input}
                """
            ),
            system_message(
                f"""
                And here is some "ADDITIONAL CONTENT" that may (or may not be) relevant to the "LATEST USER INPUT":
                {ann_text}
                """
            ),
            system_message(
                f"""
                Please evaluate the relevancy of the "ADDITIONAL CONTENT" given directly above.
                The "ADDITIONAL CONTENT" should be evaluated for relevancy to the "LATEST USER INPUT" within the context of all "PREVIOUS CONTENT".
                Please explain what parts of the "ADDITIONAL CONTENT" are relevant and how they address the "LATEST USER INPUT".
                You can mostly ignore those parts of the "ADDITIONAL CONTENT" that are irrelevant.
                """
            ),
            # system_message(
            #     f"""
            #     Again, as a reminder, here is the "ADDITIONAL CONTENT":
            #     {ann_text}
            #     """
            # ),
            system_message(
                f"""
                Now, please evaluate the relevancy the "ADDITIONAL CONTENT" as instructed above.
                """
            ),
        ]
    )

    return messages


async def generate_ann_evaluation(state: State, embedding_id: EmbeddingId) -> None:
    await call_chat_completion(
        state,
        "generate_ann_evaluation",
        state.config.ann_evaluations_llm_config,
        _create_messages(state, embedding_id),
        lambda llm_output: RagRecord(
            anns={
                embedding_id: RagAnn(
                    embedding_id=embedding_id,
                    evaluation=llm_output,
                )
            }
        ),
    )
    await state.put_internal_info_event(
        f"Completed: generate_ann_evaluation, embedding_id: {embedding_id}"
    )
