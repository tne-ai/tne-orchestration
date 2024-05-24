from typing import AsyncIterator, Callable, List, Tuple

import tiktoken
from openai import AsyncOpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.create_embedding_response import CreateEmbeddingResponse

from v2.api.api import LlmConfig, LlmOutput, RagRecord
from v2.api.util import model_to_dict
from v2.app.nn import nn
from v2.app.state import State

# TODO(Guy): This should be defined according to the embedding model.
max_embedding_token_count = 8191


async def _put_s3_debug_object(
    state: State, s3_debug_desc: str, **s3_debug_object
) -> None:
    await state.put_s3_debug_object(s3_debug_desc, s3_debug_object)


async def _call_embedding_helper(
    embedding_model: str, api_key: str, text: str
) -> Tuple[List[float], CreateEmbeddingResponse]:

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

    return embedding, embedding_response


async def call_embedding(
    state: State, embedding_model: str, api_key: str, text: str
) -> List[float]:

    call_id = state.next_call_id()
    await _put_s3_debug_object(
        state,
        f"call_embedding_args_{call_id}",
        embedding_model=embedding_model,
        text=text,
    )

    embedding, embedding_response = await _call_embedding_helper(
        embedding_model, api_key, text
    )

    await _put_s3_debug_object(
        state,
        f"call_embedding_results_{call_id}",
        embedding_model=embedding_model,
        embedding_response=model_to_dict(embedding_response),
    )

    return embedding


async def call_embedding_2(
    embedding_model: str, api_key: str, text: str
) -> List[float]:
    embedding, _ = await _call_embedding_helper(embedding_model, api_key, text)
    return embedding


async def _get_chat_completion(
    state: State,
    caller: str,
    model: str,
    api_key: str,
    messages: List[ChatCompletionMessageParam],
) -> str:

    call_id = state.next_call_id()
    await _put_s3_debug_object(
        state,
        f"call_chat_completion_args_for_{caller}_{call_id}",
        model=model,
        stream=False,
        messages=messages,
    )

    client = AsyncOpenAI(api_key=api_key)
    chat_completion = await client.chat.completions.create(
        model=model, stream=False, messages=messages
    )

    await _put_s3_debug_object(
        state,
        f"call_chat_completion_results_for_{caller}_{call_id}",
        model=model,
        stream=False,
        chat_completion=model_to_dict(chat_completion),
    )

    return chat_completion.choices[0].message.content or ""


async def _stream_chat_completion(
    state: State,
    caller: str,
    model: str,
    api_key: str,
    messages: List[ChatCompletionMessageParam],
) -> AsyncIterator[str]:

    call_id = state.next_call_id()
    await _put_s3_debug_object(
        state,
        f"call_chat_completion_args_for_{caller}_{call_id}",
        model=model,
        stream=True,
        messages=messages,
    )

    client = AsyncOpenAI(api_key=api_key)
    stream = await client.chat.completions.create(
        model=model, stream=True, messages=messages
    )
    chat_completion_chunks = []
    async for chunk in stream:
        chat_completion_chunks.append(model_to_dict(chunk))
        yield chunk.choices[0].delta.content or ""

    await _put_s3_debug_object(
        state,
        f"call_chat_completion_results_for_{caller}_{call_id}",
        model=model,
        stream=True,
        chat_completion_chunks=chat_completion_chunks,
    )


async def call_chat_completion(
    state: State,
    caller: str,
    llm_config: LlmConfig,
    messages: List[ChatCompletionMessageParam],
    record_creator: Callable[[LlmOutput], RagRecord],
) -> None:
    model = nn(llm_config.model)
    api_key = nn(llm_config.api_key)
    if llm_config.stream:
        async for chunk_text in _stream_chat_completion(
            state, caller, model, api_key, messages
        ):
            if chunk_text == "":
                continue
            await state.put_record_obj(record_creator(LlmOutput(text=chunk_text)))
        await state.put_record_obj(record_creator(LlmOutput(is_complete=True)))
    else:
        text = await _get_chat_completion(state, caller, model, api_key, messages)
        await state.put_record_obj(
            record_creator(LlmOutput(text=text, is_complete=True))
        )
