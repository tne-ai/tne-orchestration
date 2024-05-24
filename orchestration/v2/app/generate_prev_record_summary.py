from typing import List

from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from v2.api.api import RagRecord
from v2.app.messages_util import PLACEHOLDER_MESSAGES
from v2.app.openai_util import call_chat_completion
from v2.app.state import State


# TODO(Guy): How to make all this message formatting more configurable via the API?
def _create_messages(
    state: State,
) -> List[ChatCompletionMessageParam]:
    # TODO(Guy): Implement generate_prev_record_summary._create_messages.
    # This will be needed before implementing history compression.
    return PLACEHOLDER_MESSAGES


async def generate_prev_record_summary(state: State) -> None:
    await call_chat_completion(
        state,
        "generate_prev_record_summary",
        state.config.prev_record_summary_llm_config,
        _create_messages(state),
        lambda llm_output: RagRecord(prev_record_summary=llm_output),
    )
    await state.put_internal_info_event("Completed: generate_prev_record_summary")
