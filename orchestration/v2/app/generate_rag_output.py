from typing import List

from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from v2.api.api import LlmOutput, RagAnn, RagRecord
from v2.app.messages_util import (
    create_history_messages_without_compression,
    list_relevant_anns,
    system_message,
    user_message,
)
from v2.app.nn import nn
from v2.app.openai_util import call_chat_completion
from v2.app.state import State


# TODO(Guy): How to make all this message formatting more configurable via the API?
def _create_messages(
    state: State, relevant_anns: List[RagAnn]
) -> List[ChatCompletionMessageParam]:
    messages = create_history_messages_without_compression(state.request.history)
    user_input = nn(state.updated_record.user_input)
    for ann in relevant_anns:
        messages.append(
            system_message(
                f"""
                Here is some additional content that is possibly relevant to the user's next input:
                {nn(ann.evaluation).text}
                """
            )
        )
    messages.extend(
        [
            system_message(
                """
                Please consider the above additional content when answering the user's next input.
                """
            ),
            user_message(user_input),
        ]
    )
    return messages


async def generate_rag_output(state: State) -> None:
    relevant_anns = list_relevant_anns(state.updated_record)
    if not relevant_anns:
        text = "Insufficient relevant content in the RAG DB to provide an answer."
        await state.put_record_obj(
            RagRecord(rag_output=LlmOutput(text=text, is_complete=True))
        )
        await state.put_internal_info_event("Skipped: generate_rag_output")
    else:
        await call_chat_completion(
            state,
            "generate_rag_output",
            state.config.rag_output_llm_config,
            _create_messages(state, relevant_anns),
            lambda llm_output: RagRecord(rag_output=llm_output),
        )
        await state.put_internal_info_event("Completed: generate_rag_output")
