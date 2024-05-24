from typing import List

from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from v2.api.api import RagRecord
from v2.app.messages_util import (
    create_history_messages_without_compression,
    system_message,
)
from v2.app.nn import nn
from v2.app.openai_util import call_chat_completion, max_embedding_token_count
from v2.app.state import State


# TODO(Guy): How to make all this message formatting more configurable via the API?
def _create_messages(
    state: State,
) -> List[ChatCompletionMessageParam]:

    messages = create_history_messages_without_compression(state.request.history)

    user_input = nn(state.updated_record.user_input)

    # Rough conservative estimate.
    estimated_word_limit = int(max_embedding_token_count / 1.5)

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
                Please rewrite the "LATEST USER INPUT", which is given directly above, into newly "REWRITTEN USER INPUT".
                The "REWRITTEN USER INPUT" must be able to be understood when separated from the "PREVIOUS CONTENT".
                Therefore, elaborate upon the "LATEST USER INPUT" when creating the "REWRITTEN USER INPUT" to include any context from the "PREVIOUS CONTENT" which is needed to understand the "LATEST USER INPUT".
                In other words, your task is to clarify the "LATEST USER INPUT" so that the "REWRITTEN USER INPUT" has no ambiguities when separated from the "PREVIOUS CONTENT".
                The "REWRITTEN USER INPUT" must retain its full intended meaning without benefit of context from the "PREVIOUS CONTENT".
                
                The "REWRITTEN USER INPUT", which you will create, will be embedded into a vector space by a text embedding model.
                The embedding vector will be used to search for additional content related to the "REWRITTEN USER INPUT".
                This is why the "REWRITTEN USER INPUT" must accurately capture the full intended meaning of the "LATEST USER INPUT" as interpreted within the context of the "PREVIOUS CONTENT".
                Also, the "REWRITTEN USER INPUT" must be dense without unnecessary wording, so that it can be embedded with semantic richness.
                The "REWRITTEN USER INPUT" should not exceed {estimated_word_limit} words.
                """
            ),
            system_message(
                f"""
                Again, as a reminder, here is the "LATEST USER INPUT":
                {user_input}
                """
            ),
            system_message(
                f"""
                Now, please rewrite, elaborate, and clarify the "LATEST USER INPUT" as instructed above.
                """
            ),
        ]
    )

    return messages


async def generate_anns_input(state: State) -> None:
    await call_chat_completion(
        state,
        "generate_anns_input",
        state.config.anns_input_llm_config,
        _create_messages(state),
        lambda llm_output: RagRecord(anns_input=llm_output),
    )
    await state.put_internal_info_event("Completed: generate_anns_input")
