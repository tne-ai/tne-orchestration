import asyncio
import os
import sys
import time
import uuid

from dotenv import load_dotenv

from v2.api.api import (
    EmbeddingsConfig,
    LlmConfig,
    RagConfig,
    RagRecord,
    RagRequest,
    RagResponse,
)
from v2.api.util import (
    async_iterate_streaming_request,
    merge_records,
    rag_record_to_yaml_str,
)


def _random_uuid():
    return str(uuid.uuid4())


def _create_config(env_file):
    load_dotenv(env_file)
    embeddings_db_password = os.getenv("EMBEDDINGS_DB_PASSWORD")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    shared_llm_config = LlmConfig(model="gpt-4-0125-preview", api_key=openai_api_key)
    return RagConfig(
        embeddings_config=EmbeddingsConfig(
            db_host="postgresql-ebp.cfwmuvh4blso.us-west-2.rds.amazonaws.com",
            db_port=5432,
            db_name="crag_agent_db",
            db_username="postgres",
            db_password=embeddings_db_password,
            model="text-embedding-ada-002",
            api_key=openai_api_key,
        ),
        anns_input_llm_config=shared_llm_config,
        ann_evaluations_llm_config=shared_llm_config,
        ann_relevancies_llm_config=shared_llm_config,
        anns_summary_llm_config=shared_llm_config,
        prev_record_summary_llm_config=shared_llm_config,
        full_history_summary_llm_config=shared_llm_config,
        rag_output_llm_config=shared_llm_config,
    )


async def main():
    _, env_file = sys.argv

    config = _create_config(env_file)
    thread_id = _random_uuid()
    history = []

    while True:
        print(f"\n{'=' * 80}\n")
        try:
            user_input = input("input> ")
        except (EOFError, KeyboardInterrupt):
            return
        finally:
            print()

        request_id = _random_uuid()
        new_record = RagRecord(
            request_id=request_id, config=config, user_input=user_input
        )
        request = RagRequest(
            thread_id=thread_id, history=history, new_record=new_record
        )

        response_count = 0
        start_seconds = time.time()

        def print_progress(done: bool = False):
            elapsed_seconds = time.time() - start_seconds
            print(
                f"response_count: {response_count:4d}, elapsed_seconds: {elapsed_seconds:7.3f}",
                end="\n" if done else "\r",
            )

        async def on_response(response: RagResponse, response_str: str):
            nonlocal response_count
            response_count += 1
            print_progress()

        async def on_error(error_code: int, error_str: str) -> bool:
            print(f"error_code: {error_code}, error_str: {error_str}")
            sys.exit(1)
            return False  # Just for type checker's sake.

        print_progress()
        responses, request_bytes_written, response_bytes_read, elapsed_seconds = (
            await async_iterate_streaming_request(
                "http://localhost:8080/v2/rag", request, on_response, on_error
            )
        )
        print_progress(done=True)

        assert response_count == len(responses)

        patch_records = [response.patch_record for response in responses]
        merged_record = request.new_record
        for patch_record in patch_records:
            merged_record = merge_records(merged_record, patch_record)
        print(f"\n{'-' * 80}\n")
        print(rag_record_to_yaml_str(merged_record, "merged_record"))

        print(f"\n{'-' * 80}\n")
        print(f"response_count:         {response_count:6d}")
        print(f"request_bytes_written:  {request_bytes_written:6d}")
        print(f"response_bytes_read:    {response_bytes_read:6d}")
        print(f"elapsed_seconds:        {elapsed_seconds:10.3f}")

        history.append(merged_record)


if __name__ == "__main__":
    asyncio.run(main())
