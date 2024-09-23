import asyncio
import gzip
import json
import logging
import os
from datetime import datetime
from enum import Enum
from io import BytesIO
from textwrap import dedent, indent
from typing import Any, Optional

import aioboto3
from opentelemetry import trace
from opentelemetry.trace import SpanContext

from v2.api.api import RequestId, ThreadId
from v2.app.nn import nn

S3DebugDesc = str
S3DebugSite = tuple[datetime, SpanContext, S3DebugDesc]
S3DebugObject = dict[str, Any]
S3DebugSiteObject = tuple[S3DebugSite, S3DebugObject]
S3DebugQueue = asyncio.Queue[Optional[S3DebugSiteObject]]

_logger = logging.getLogger(__name__)


class _EnumJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.name
        return json.JSONEncoder.default(self, obj)


def get_current_span_context() -> SpanContext:
    return trace.get_current_span().get_span_context()


def _format_datetime_str(utc_datetime: datetime) -> str:
    return utc_datetime.strftime("%Y_%m_%d_%H_%M_%S_%f_Z")


def _format_date_str(utc_datetime: datetime) -> str:
    return utc_datetime.strftime("%Y_%m_%d_Z")


def _format_time_str(utc_datetime: datetime) -> str:
    return utc_datetime.strftime("%H_%M_%S_%f_Z")


def _fix_indentation(s: str, indent_levels: int, trailing_newline: bool = True) -> str:
    s = s.strip("\n")
    s = dedent(s)
    if indent_levels > 0:
        s = indent(s, "    " * indent_levels)
    if not trailing_newline:
        s = s.rstrip("\n")
    return s


def _format_and_gzip_header(
    gzip_file: gzip.GzipFile,
    k8s_cluster_name: str,
    utc_datetime: datetime,
    trace_id: int,
    thread_id: ThreadId,
    request_id: RequestId,
):
    # In a Kubernetes deployment, the HOSTNAME should be the name of the pod.
    k8s_pod_name = os.getenv("HOSTNAME", "MISSING_K8S_POD_NAME")
    datetime_str = _format_datetime_str(utc_datetime)
    trace_id_str = f"{trace_id:032x}"
    formatted_header = _fix_indentation(
        f"""
        {{
            "k8s_cluster_name": "{k8s_cluster_name}",
            "k8s_pod_name": "{k8s_pod_name}",
            "utc_datetime": "{datetime_str}",
            "otel_trace_id": "{trace_id_str}",
            "rag_thread_id": "{thread_id}",
            "rag_request_id": "{request_id}",
            "rag_debug_objects": {{
        """,
        indent_levels=0,
    )
    gzip_file.write(formatted_header.encode("utf-8"))


def _format_and_gzip_member(
    gzip_file: gzip.GzipFile,
    utc_datetime: datetime,
    span_id: int,
    s3_debug_desc: S3DebugDesc,
    s3_debug_object: S3DebugObject,
    has_previous_member: bool,
):
    if has_previous_member:
        gzip_file.write(",\n".encode("utf-8"))
    datetime_str = _format_datetime_str(utc_datetime)
    span_id_str = f"{span_id:016x}"
    debug_object_str = json.dumps(s3_debug_object, indent=4, cls=_EnumJSONEncoder)
    debug_object_str = _fix_indentation(debug_object_str, indent_levels=3)
    debug_object_str = debug_object_str.strip()
    formatted_member = _fix_indentation(
        f"""
        "{datetime_str}_span_{span_id_str}_object_{s3_debug_desc}": {{
            "utc_datetime": "{datetime_str}",
            "otel_span_id": "{span_id_str}",
            "rag_debug_description": "{s3_debug_desc}",
            "rag_debug_object": {debug_object_str}
        }}
        """,
        indent_levels=2,
        trailing_newline=False,
    )

    gzip_file.write(formatted_member.encode("utf-8"))


def _format_and_gzip_footer(gzip_file: gzip.GzipFile):
    formatted_footer = _fix_indentation(
        """
            }
        }
        """,
        indent_levels=0,
    )
    gzip_file.write(formatted_footer.encode("utf-8"))


async def _write_s3_debug_content(
    bytes_io: BytesIO, cluster_name: str, utc_datetime: datetime, trace_id: int
):
    region = "us-west-2"
    bucket = "debug.rag.tne.ai"
    date_str = _format_date_str(utc_datetime)
    datetime_str = _format_datetime_str(utc_datetime)
    trace_id_str = f"{trace_id:032x}"
    key = f"{cluster_name}/{date_str}/{datetime_str}_trace_{trace_id_str}.json.gz"
    try:
        async with aioboto3.Session().client("s3") as s3:
            await s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=bytes_io.getvalue(),
                ContentType="application/json",
                ContentEncoding="gzip",
            )
        console_url = (
            f"https://{region}.console.aws.amazon.com/s3/object/{bucket}/{key}"
        )
        _logger.info(f"Wrote S3 debug content, console_url: {console_url}")
    except Exception as exception:
        _logger.error(f"Failed writing S3 debug content, exception: {exception}")
        raise


async def process_s3_debug_queue(
    thread_id: ThreadId, request_id: RequestId, s3_debug_queue: S3DebugQueue
) -> None:

    # TODO(Guy): Configure k8s_cluster_name as env. var.
    k8s_cluster_name = "TODO_K8S_CLUSTER_NAME"

    with BytesIO() as bytes_io:

        first_utc_datetime = None
        trace_id = None

        with gzip.GzipFile(fileobj=bytes_io, mode="wb") as gzip_file:
            has_previous_member = False
            s3_debug_site_object: Optional[S3DebugSiteObject]
            while (s3_debug_site_object := await s3_debug_queue.get()) is not None:
                s3_debug_site, s3_debug_object = s3_debug_site_object
                utc_datetime, span_context, s3_debug_desc = s3_debug_site
                if first_utc_datetime is None:
                    first_utc_datetime = utc_datetime
                else:
                    assert first_utc_datetime <= utc_datetime
                if trace_id is None:
                    trace_id = span_context.trace_id
                    _format_and_gzip_header(
                        gzip_file,
                        k8s_cluster_name,
                        first_utc_datetime,
                        trace_id,
                        thread_id,
                        request_id,
                    )
                else:
                    assert trace_id == span_context.trace_id
                span_id = span_context.span_id
                _format_and_gzip_member(
                    gzip_file,
                    utc_datetime,
                    span_id,
                    s3_debug_desc,
                    s3_debug_object,
                    has_previous_member,
                )
                has_previous_member = True
                s3_debug_queue.task_done()
            gzip_file.write("\n".encode("utf-8"))
            _format_and_gzip_footer(gzip_file)

        await _write_s3_debug_content(
            bytes_io, k8s_cluster_name, nn(first_utc_datetime), nn(trace_id)
        )

    s3_debug_queue.task_done()  # For that terminating None value.
