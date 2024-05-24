from textwrap import dedent
from typing import List, cast

from openai.types.chat.chat_completion_message_param import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from v2.api.api import RagAnn, RagRecord
from v2.app.nn import nn

PLACEHOLDER_MESSAGES: List[ChatCompletionMessageParam] = [
    {
        "role": "user",
        "content": "Please write a creative paragraph about 300 words long.",
    }
]


def _create_message(role: str, content: str) -> ChatCompletionMessageParam:
    return cast(
        ChatCompletionMessageParam, {"role": role, "content": dedent(content).strip()}
    )


def assistant_message(content: str) -> ChatCompletionAssistantMessageParam:
    return cast(
        ChatCompletionAssistantMessageParam, _create_message("assistant", content)
    )


def system_message(content: str) -> ChatCompletionSystemMessageParam:
    return cast(ChatCompletionSystemMessageParam, _create_message("system", content))


def user_message(content: str) -> ChatCompletionUserMessageParam:
    return cast(ChatCompletionUserMessageParam, _create_message("user", content))


def list_relevant_anns(record: RagRecord) -> List[RagAnn]:
    anns = record.anns
    if not anns:
        return []
    min_relevancy = nn(record.config).ann_relevancy_min_value
    relevant_anns = [
        ann
        for ann in anns.values()
        if ann.evaluation and nn(ann.relevancy) >= min_relevancy
    ]
    return relevant_anns


def create_history_messages_without_compression(
    history: List[RagRecord],
) -> List[ChatCompletionMessageParam]:
    messages: List[ChatCompletionMessageParam] = [
        # TODO(Guy): Configure the expertise description via the API.
        system_message("You are a helpful expert.")
    ]
    for prev_record in history:
        user_input = nn(prev_record.user_input)
        messages.append(user_message(user_input))
        relevant_anns = list_relevant_anns(prev_record)
        if relevant_anns:
            for ann in relevant_anns:
                messages.extend(
                    [
                        system_message(
                            f"""
                            Here is some additional content that may (or may not) be relevant to the user's previous input:
                            {ann.text}
                            """
                        ),
                        system_message(
                            f"""
                            Here is an evaluation of how the additional content may be relevant to the user's previous input:
                            {ann.evaluation}
                            """
                        ),
                    ]
                )
            if prev_record.rag_output and prev_record.rag_output.text:
                messages.append(assistant_message(prev_record.rag_output.text))
    return messages
