import asyncio
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from v2.api.util import request_from_dict, response_to_json_str
from v2.app.rag import RagResponse, rag


async def _health_live_handler() -> JSONResponse:
    # TODO(Guy): Implement more detailed liveness health check.
    return JSONResponse(content={"status": "ok"})


async def _health_ready_handler() -> JSONResponse:
    # TODO(Guy): Implement more detailed readiness health check.
    return JSONResponse(content={"status": "ok"})


async def _responses_generator(responses_queue: asyncio.Queue) -> AsyncGenerator:
    response: RagResponse
    while (response := await responses_queue.get()) is not None:
        response_json = response_to_json_str(response)
        yield response_json + "\n"
        responses_queue.task_done()
    responses_queue.task_done()  # For that terminating None value.


async def _rag_handler(request_dict: dict) -> StreamingResponse:
    request = request_from_dict(request_dict)
    responses_queue: asyncio.Queue = asyncio.Queue(1_000)
    asyncio.create_task(rag(request, responses_queue))
    return StreamingResponse(
        _responses_generator(responses_queue),
        media_type="application/x-ndjson",
        status_code=200,  # Stream may contain errors.
    )


def add_api_routes(app: FastAPI):
    app.add_api_route("/v2/health/live", _health_live_handler, methods=["GET"])
    app.add_api_route("/v2/health/ready", _health_ready_handler, methods=["GET"])
    app.add_api_route("/v2/rag", _rag_handler, methods=["POST"])
