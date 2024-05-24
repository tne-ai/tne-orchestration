"Code that may be helpful to clients of the RAG API."

import json
import time
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

import httpx
import requests
import yaml
from pydantic import BaseModel

from v2.api.api import AnnsRequest, AnnsResponse, RagRecord, RagRequest, RagResponse
from v2.api.updatable import merge_updatable_models


def merge_records(basis_record: RagRecord, patch_record: RagRecord) -> RagRecord:
    return merge_updatable_models(basis_record, patch_record)


# NOTE: If you want a JSON str, then use model_to_json_str(model), and
# not json.dumps(model_to_dict(model)) because the dict route has
# enum objects which json.dumps doesn't serialize well.


_T = TypeVar("_T", bound=BaseModel)


# Generic serialization functions:


def model_to_dict(model: _T) -> Dict[str, Any]:
    return model.model_dump(exclude_none=True, exclude_defaults=True)


def model_to_json_str(model: _T, indent: Optional[int] = None) -> str:
    return model.model_dump_json(
        exclude_none=True, exclude_defaults=True, indent=indent
    )


def model_to_yaml_str(model: _T, root_field_name: Optional[str] = "model") -> str:
    # Round-trip through json to force enums into strs before yaml.dump.
    model_dict = json.loads(model_to_json_str(model))
    if root_field_name:
        model_dict = {root_field_name: model_dict}
    return yaml.dump(model_dict, sort_keys=False)


#
# Generic deserialization functions:
#


def model_from_dict(model_type: Type[_T], dict_: Dict[str, Any]) -> _T:
    return model_type(**dict_)


def model_from_json_str(model_type: Type[_T], json_str: str) -> _T:
    return model_from_dict(model_type, json.loads(json_str))


#
# RagRequest typed wrapper functions for serialization/deserialization:
#


def rag_request_to_dict(request: RagRequest) -> Dict[str, Any]:
    return model_to_dict(request)


def rag_request_to_json_str(request: RagRequest, indent: Optional[int] = None) -> str:
    return model_to_json_str(request, indent)


def rag_request_to_yaml_str(
    request: RagRequest, root_field_name: Optional[str] = "request"
) -> str:
    return model_to_yaml_str(request, root_field_name)


def rag_request_from_dict(dict_: Dict[str, Any]) -> RagRequest:
    return model_from_dict(RagRequest, dict_)


def rag_request_from_json_str(json_str: str) -> RagRequest:
    return model_from_json_str(RagRequest, json_str)


#
# RagResponse typed wrapper functions for serialization/deserialization:
#


def rag_response_to_dict(response: RagResponse) -> Dict[str, Any]:
    return model_to_dict(response)


def rag_response_to_json_str(
    response: RagResponse, indent: Optional[int] = None
) -> str:
    return model_to_json_str(response, indent)


def rag_response_to_yaml_str(
    response: RagResponse, root_field_name: Optional[str] = "response"
) -> str:
    return model_to_yaml_str(response, root_field_name)


def rag_response_from_dict(dict_: Dict[str, Any]) -> RagResponse:
    return model_from_dict(RagResponse, dict_)


def rag_response_from_json_str(json_str: str) -> RagResponse:
    return model_from_json_str(RagResponse, json_str)


#
# RagRecord typed wrapper functions for serialization/deserialization:
#


def rag_record_to_dict(record: RagRecord) -> Dict[str, Any]:
    return model_to_dict(record)


def rag_record_to_json_str(record: RagRecord, indent: Optional[int] = None) -> str:
    return model_to_json_str(record, indent)


def rag_record_to_yaml_str(
    record: RagRecord, root_field_name: Optional[str] = "record"
) -> str:
    return model_to_yaml_str(record, root_field_name)


def rag_record_from_dict(dict_: Dict[str, Any]) -> RagRecord:
    return model_from_dict(RagRecord, dict_)


def rag_record_from_json_str(json_str: str) -> RagRecord:
    return model_from_json_str(RagRecord, json_str)


#
# AnnsRequest typed wrapper functions for serialization/deserialization:
#


def anns_request_to_dict(request: AnnsRequest) -> Dict[str, Any]:
    return model_to_dict(request)


def anns_request_to_json_str(request: AnnsRequest, indent: Optional[int] = None) -> str:
    return model_to_json_str(request, indent)


def anns_request_to_yaml_str(
    request: AnnsRequest, root_field_name: Optional[str] = "request"
) -> str:
    return model_to_yaml_str(request, root_field_name)


def anns_request_from_dict(dict_: Dict[str, Any]) -> AnnsRequest:
    return model_from_dict(AnnsRequest, dict_)


def anns_request_from_json_str(json_str: str) -> AnnsRequest:
    return model_from_json_str(AnnsRequest, json_str)


#
# AnnsResponse typed wrapper functions for serialization/deserialization:
#


def anns_response_to_dict(response: AnnsResponse) -> Dict[str, Any]:
    return model_to_dict(response)


def anns_response_to_json_str(
    response: AnnsResponse, indent: Optional[int] = None
) -> str:
    return model_to_json_str(response, indent)


def anns_response_to_yaml_str(
    response: AnnsResponse, root_field_name: Optional[str] = "response"
) -> str:
    return model_to_yaml_str(response, root_field_name)


def anns_response_from_dict(dict_: Dict[str, Any]) -> AnnsResponse:
    return model_from_dict(AnnsResponse, dict_)


def anns_response_from_json_str(json_str: str) -> AnnsResponse:
    return model_from_json_str(AnnsResponse, json_str)


#
# Synchronous client helper function for iterating a streaming request.
# (Similar to async version below.)
#


def iterate_streaming_request(
    service_url: str,
    request: RagRequest,
    on_response: Callable[[RagResponse, str], None],
    on_error: Callable[[int, str], bool],
) -> Optional[Tuple[List[RagResponse], int, int, float]]:
    request_bytes = rag_request_to_json_str(request).encode()
    start_seconds = time.time()
    response_obj = requests.post(
        service_url,
        data=request_bytes,
        headers={"Content-Type": "application/json"},
        stream=True,
    )
    status_code = response_obj.status_code
    if status_code != 200:
        keep_going = on_error(status_code, f"ERROR STATUS CODE: {status_code}")
        if not keep_going:
            return None
    responses = []
    request_bytes_written = len(request_bytes)
    response_bytes_read = 0
    for line_bytes in response_obj.iter_lines():
        response_bytes_read += len(line_bytes) + 1  # for stripped newline
        line_str = line_bytes.decode()
        if not (line_str.startswith("{") and line_str.endswith("}")):
            keep_going = on_error(500, f"INVALID LINE: {line_str}")
            if not keep_going:
                return None
            continue
        response = rag_response_from_json_str(line_str)
        responses.append(response)
        on_response(response, line_str)
    elapsed_seconds = time.time() - start_seconds
    return responses, request_bytes_written, response_bytes_read, elapsed_seconds


#
# Asynchronous client helper function for iterating a streaming request.
# (Similar to sync version above.)
#


async def async_iterate_streaming_request(
    service_url: str,
    request: RagRequest,
    on_response: Callable[[RagResponse, str], Coroutine[None, None, None]],
    on_error: Callable[[int, str], Coroutine[None, None, bool]],
) -> Optional[Tuple[List[RagResponse], int, int, float]]:
    request_bytes = rag_request_to_json_str(request).encode()
    start_seconds = time.time()
    client = httpx.AsyncClient()
    async with client.stream(
        "POST",
        service_url,
        content=request_bytes,
        headers={"Content-Type": "application/json"},
    ) as httpx_response:
        status_code = httpx_response.status_code
        if status_code != 200:
            keep_going = await on_error(
                status_code, f"ERROR STATUS CODE: {status_code}"
            )
            if not keep_going:
                return None
        responses = []
        request_bytes_written = len(request_bytes)
        response_bytes_read = 0
        async for line in httpx_response.aiter_lines():
            response_bytes_read += len(line) + 1  # for stripped newline
            if not (line.startswith("{") and line.endswith("}")):
                keep_going = await on_error(500, f"INVALID LINE: {line}")
                if not keep_going:
                    return None
                continue
            response = rag_response_from_json_str(line)
            responses.append(response)
            await on_response(response, line)
    elapsed_seconds = time.time() - start_seconds
    return responses, request_bytes_written, response_bytes_read, elapsed_seconds


async def async_iterate_streaming_request_generator(
    service_url: str,
    request: RagRequest,
    on_response: Callable[[RagResponse, str], Coroutine[None, None, None]],
    on_error: Callable[[int, str], Coroutine[None, None, bool]],
) -> AsyncGenerator[RagResponse, None]:
    request_bytes = rag_request_to_json_str(request).encode()
    client = httpx.AsyncClient()
    async with client.stream(
        "POST",
        service_url,
        content=request_bytes,
        headers={"Content-Type": "application/json"},
    ) as response:
        status_code = response.status_code
        if status_code != 200:
            keep_going = await on_error(
                status_code, f"ERROR STATUS CODE: {status_code}"
            )
            if not keep_going:
                return
        async for line in response.aiter_lines():
            if not (line.startswith("{") and line.endswith("}")):
                keep_going = await on_error(500, f"INVALID LINE: {line}")
                if not keep_going:
                    return
                continue
            rag_response = rag_response_from_json_str(line)
            await on_response(rag_response, line)
            yield rag_response
