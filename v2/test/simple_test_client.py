import sys
from typing import List

from v2.api.api import RagRecord, RagResponse
from v2.api.util import (
    iterate_streaming_request,
    record_from_json_str,
    record_to_json_str,
    request_from_json_str,
)


def write_patch_records(patch_records_file: str, patch_records: List[RagRecord]):
    lines = [record_to_json_str(patch_record) + "\n" for patch_record in patch_records]
    with open(patch_records_file, "w") as patch_records_fp:
        patch_records_fp.writelines(lines)


def read_patch_records(patch_records_file: str) -> List[RagRecord]:
    with open(patch_records_file) as patch_records_fp:
        lines = patch_records_fp.readlines()
    patch_records = [record_from_json_str(line) for line in lines]
    return patch_records


def main():
    _, request_file, patch_records_file = sys.argv

    with open(request_file) as request_fp:
        request = request_from_json_str(request_fp.read())

    def on_response(response: RagResponse, response_str: str):
        print(response_str)

    def on_error(error_code: int, error_str: str) -> bool:
        print(f"error_code: {error_code}, error_str: {error_str}")
        sys.exit(1)
        return False  # Just for type checker's sake.

    responses, request_bytes_written, response_bytes_read, elapsed_seconds = (
        iterate_streaming_request(
            "http://localhost:8080/v2/rag", request, on_response, on_error
        )
    )

    patch_records = [request.new_record] + [
        response.patch_record for response in responses
    ]

    print("-" * 80)
    print(f"response_count:         {len(responses):6d}")
    print(f"request_bytes_written:  {request_bytes_written:6d}")
    print(f"response_bytes_read:    {response_bytes_read:6d}")
    print(f"elapsed_seconds:        {elapsed_seconds:10.3f}")

    write_patch_records(patch_records_file, patch_records)


if __name__ == "__main__":
    main()
