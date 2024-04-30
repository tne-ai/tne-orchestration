"Data model classes defining the RAG API."

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel

from v2.api.updatable import UpdatableField, UpdatableModel, UpdatePolicy

# Notes:
#
# ANN == Approximate Nearest Neighbor
# ANNs == Approximate Nearest Neighbors
#
# For now, any field marked Optional is fine being left unset.
# At some point, some Optional fields will become required.
# Mainly, that will be certain LLM and embeddings DB config fields.

# Some type aliases for readability:
ThreadId = str
RequestId = str
EmbeddingId = str

UF = UpdatableField
UP = UpdatePolicy


class LlmConfig(UpdatableModel):
    "Configuration of which LLM to use and how to build messages for it."

    model: Optional[str] = UF(None, UP.reject)
    api_key: Optional[str] = UF(None, UP.reject)
    stream: bool = UF(True, UP.reject)
    temperature: Optional[float] = UF(None, UP.reject)
    messages_builder_id: Optional[str] = UF(None, UP.reject)

    reduced_token_limit: Optional[int] = UF(None, UP.reject)
    no_summary_weight: Optional[float] = UF(None, UP.reject)
    anns_summary_weight: Optional[float] = UF(None, UP.reject)
    record_summary_weight: Optional[float] = UF(None, UP.reject)

    # TODO(Guy): More model config values applicable across models.
    # Perhaps the above will become pass-through config for SlashTNE?
    # But we'll still need dynamic prompt messages building.


class EmbeddingsConfig(UpdatableModel):
    "Configuration of which embeddings DB and model to use."

    db_host: Optional[str] = UF(None, UP.reject)
    db_port: Optional[int] = UF(None, UP.reject)
    db_name: Optional[str] = UF(None, UP.reject)
    db_username: Optional[str] = UF(None, UP.reject)
    db_password: Optional[str] = UF(None, UP.reject)

    model: Optional[str] = UF(None, UP.reject)
    api_key: Optional[str] = UF(None, UP.reject)


class RagConfig(UpdatableModel):
    "Configuration for how RAG should process the given request."

    embeddings_config: EmbeddingsConfig = UF(EmbeddingsConfig(), UP.reject)

    generate_anns_input: bool = UF(True, UP.reject)
    generate_ann_evaluations: bool = UF(True, UP.reject)
    generate_ann_relevancies: bool = UF(True, UP.reject)
    generate_anns_summary: bool = UF(True, UP.reject)
    generate_prev_record_summary: bool = UF(True, UP.reject)
    generate_full_history_summary: bool = UF(True, UP.reject)

    anns_input_llm_config: LlmConfig = UF(LlmConfig(), UP.reject)
    ann_evaluations_llm_config: LlmConfig = UF(LlmConfig(), UP.reject)
    ann_relevancies_llm_config: LlmConfig = UF(LlmConfig(), UP.reject)
    anns_summary_llm_config: LlmConfig = UF(LlmConfig(), UP.reject)
    prev_record_summary_llm_config: LlmConfig = UF(LlmConfig(), UP.reject)
    full_history_summary_llm_config: LlmConfig = UF(LlmConfig(), UP.reject)
    rag_output_llm_config: LlmConfig = UF(LlmConfig(), UP.reject)

    ann_evaluation_before_relevancy: bool = UF(True, UP.reject)
    ann_retrieval_count: int = UF(20, UP.reject)
    ann_similarity_min_value: float = UF(0.8, UP.reject)
    ann_similarity_max_count: int = UF(10, UP.reject)
    ann_relevancy_min_value: float = UF(0.8, UP.reject)
    ann_relevancy_max_count: int = UF(5, UP.reject)

    sleep_seconds_between_batches: float = UF(0.2, UP.reject)


class LlmOutput(UpdatableModel):
    "The optionally streamed output of an LLM invokation."

    text: Optional[str] = UF(None, UP.concat)
    is_complete: Optional[bool] = UF(None, UP.set_once)
    token_count: Optional[int] = UF(None, UP.set_once)


class RagAnn(UpdatableModel):
    "A single approximate nearest neighbor (ANN) result."

    # embedding_id also leads to embedding vector, text, and source
    embedding_id: EmbeddingId = UF(None, UP.expect_equal)
    text: Optional[str] = UF(None, UP.concat)
    similarity: Optional[float] = UF(None, UP.set_once)
    evaluation: Optional[LlmOutput] = UF(None, UP.merge)
    raw_relevancy: Optional[LlmOutput] = UF(None, UP.merge)
    relevancy: Optional[float] = UF(None, UP.set_once)


class RagEventSeverity(Enum):
    "Severity of the event."

    Info = "Info"
    "Normal info. Response stream progressing or complete."

    Warning = "Warning"
    "Abnormal info. Response stream progressing or complete."

    Error = "Error"
    "Response stream terminated and incomplete."


class RagEventSource(Enum):
    "Source of the event."

    User = "User"
    "Source is usage of the API."

    Dependency = "Dependency"
    "Source is some library/service dependency."

    Internal = "Internal"
    "Source is internal to RAG service. A bug."


class RagEvent(UpdatableModel):
    "Events occurring during RAG useful for indicating status."

    severity: RagEventSeverity = UF(None, UP.reject)
    source: RagEventSource = UF(None, UP.reject)
    retry: bool = UF(None, UP.reject)
    message: str = UF(None, UP.reject)


class RagMetrics(UpdatableModel):
    "Metrics about the performance of RAG useful for expert user tuning."

    ann_count_after_retrieval: Optional[int] = UF(None, UP.set_once)
    ann_count_after_similarity_min_value: Optional[int] = UF(None, UP.set_once)
    ann_count_after_similarity_max_count: Optional[int] = UF(None, UP.set_once)
    ann_count_after_relevancy_min_value: Optional[int] = UF(None, UP.set_once)
    ann_count_after_relevancy_max_count: Optional[int] = UF(None, UP.set_once)


class RagRecord(UpdatableModel):
    "An accumulating record of a single user_input to rag_output iteration."

    # Fields that appear in both requests and responses:
    request_id: RequestId = UF(None, UP.expect_equal)

    # Fields that appear only in requests:
    config: Optional[RagConfig] = UF(None, UP.reject)
    user_input: Optional[str] = UF(None, UP.reject)

    # Fields that appear only in responses:
    anns_input: Optional[LlmOutput] = UF(None, UP.merge)
    anns: Optional[Dict[EmbeddingId, RagAnn]] = UF(None, UP.merge)
    anns_summary: Optional[LlmOutput] = UF(None, UP.merge)
    prev_record_summary: Optional[LlmOutput] = UF(None, UP.merge)
    full_history_summary: Optional[LlmOutput] = UF(None, UP.merge)
    rag_output: Optional[LlmOutput] = UF(None, UP.merge)
    events: Optional[List[RagEvent]] = UF(None, UP.concat)
    metrics: Optional[RagMetrics] = UF(None, UP.merge)

    # TODO(Guy): Multiple ANN query inputs?


class RagRequest(BaseModel):
    "A single RagRequest will be posted to the API."

    thread_id: ThreadId
    history: List[RagRecord]
    new_record: RagRecord


class RagResponse(BaseModel):
    "Multiple RagResponses will be streamed back from the API."

    thread_id: ThreadId  # Is this helpful in response?
    patch_record: RagRecord
