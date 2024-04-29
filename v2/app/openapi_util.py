from typing import AsyncIterator, Callable, List

import tiktoken
from openai import AsyncOpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from v2.api.api import LlmConfig, LlmOutput, RagRecord
from v2.app.nn import nn
from v2.app.state import State

# TODO(Guy): This should be defined according to the embedding model.
max_embedding_token_count = 8191


async def call_embedding(embedding_model: str, api_key: str, text: str) -> List[float]:
    # TODO(Guy): Generalize these expectations:
    expected_embedding_model = "text-embedding-ada-002"
    expected_token_encoding = "cl100k_base"
    expected_embedding_len = 1536

    assert embedding_model == expected_embedding_model
    encoding = tiktoken.model.encoding_for_model(embedding_model)
    assert encoding.name == expected_token_encoding
    tokens = encoding.encode(text)
    if len(tokens) > max_embedding_token_count:
        tokens = tokens[:max_embedding_token_count]
    client = AsyncOpenAI(api_key=api_key)
    embedding_response = await client.embeddings.create(
        input=tokens, model=embedding_model
    )
    embedding = embedding_response.data[0].embedding
    assert len(embedding) == expected_embedding_len
    return embedding


async def _get_chat_completion(
    model: str, api_key: str, messages: List[ChatCompletionMessageParam]
) -> str:
    client = AsyncOpenAI(api_key=api_key)
    chat_completion = await client.chat.completions.create(
        model=model, stream=False, messages=messages
    )
    return chat_completion.choices[0].message.content or ""


async def _stream_chat_completion(
    model: str, api_key: str, messages: List[ChatCompletionMessageParam]
) -> AsyncIterator[str]:
    client = AsyncOpenAI(api_key=api_key)
    stream = await client.chat.completions.create(
        model=model, stream=True, messages=messages
    )
    async for chunk in stream:
        yield chunk.choices[0].delta.content or ""


async def call_chat_completion(
    state: State,
    llm_config: LlmConfig,
    messages: List[ChatCompletionMessageParam],
    record_creator: Callable[[LlmOutput], RagRecord],
) -> None:
    model, api_key = nn(llm_config.model), nn(llm_config.api_key)
    if llm_config.stream:
        async for chunk_text in _stream_chat_completion(model, api_key, messages):
            if chunk_text == "":
                continue
            await state.put_record_obj(record_creator(LlmOutput(text=chunk_text)))
        await state.put_record_obj(record_creator(LlmOutput(is_complete=True)))
    else:
        text = await _get_chat_completion(model, api_key, messages)
        await state.put_record_obj(
            record_creator(LlmOutput(text=text, is_complete=True))
        )
