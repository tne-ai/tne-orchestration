from abc import ABC, abstractmethod

import tiktoken
from openai import OpenAI

from v2.etl.openai_util import set_openai_api_key
from v2.etl.typer_util import TyperEnum


class Embedder(ABC):
    @abstractmethod
    def tokenize(self, text):
        pass

    @abstractmethod
    def embed(self, tokens):
        pass


class OpenAIAda002Embedder(Embedder):
    _embedding_model = "text-embedding-ada-002"
    _embedding_len = 1536
    _token_encoding = "cl100k_base"

    def __init__(self):
        super().__init__()

        # TODO: openai_env_file arg should be command line parameter.
        set_openai_api_key("../openai.env")

        self._client = OpenAI()
        self._encoding = tiktoken.model.encoding_for_model(
            OpenAIAda002Embedder._embedding_model
        )
        assert self._encoding.name == OpenAIAda002Embedder._token_encoding

    def tokenize(self, text):
        tokens = self._encoding.encode(text)
        return tokens

    def embed(self, tokens):
        max_token_count = 8191
        if len(tokens) > max_token_count:
            tokens = tokens[:max_token_count]
        embedding_response = self._client.embeddings.create(
            input=tokens, model=OpenAIAda002Embedder._embedding_model
        )
        embedding = embedding_response.data[0].embedding
        assert len(embedding) == OpenAIAda002Embedder._embedding_len
        return embedding


class EmbedderId(TyperEnum):
    openai_ada_002 = "openai_ada_002", OpenAIAda002Embedder
