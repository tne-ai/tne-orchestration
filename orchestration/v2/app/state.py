import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional

from v2.api.api import (
    EmbeddingId,
    LlmOutput,
    RagAnn,
    RagConfig,
    RagEvent,
    RagEventSeverity,
    RagEventSource,
    RagMetrics,
    RagRecord,
    RagRequest,
)
from v2.api.util import merge_records
from v2.app.nn import nn
from v2.app.s3_debug import (
    S3DebugDesc,
    S3DebugObject,
    S3DebugQueue,
    S3DebugSite,
    S3DebugSiteObject,
    get_current_span_context,
)


class State:
    def __init__(
        self,
        request: RagRequest,
        patch_records_queue: asyncio.Queue,
        s3_debug_queue: S3DebugQueue,
    ):
        self.request = request
        self.updated_record = request.new_record
        self.config: RagConfig = nn(request.new_record.config)
        self.patch_records_queue = patch_records_queue
        self.s3_debug_queue = s3_debug_queue
        self.next_call_id_int = 0
        self.finalized = False

    async def finalize(self) -> None:
        assert not self.finalized
        self.finalized = True
        await self.patch_records_queue.put(None)
        await self.s3_debug_queue.put(None)

    def next_call_id(self) -> str:
        assert not self.finalized
        formatted_next_call_id = f"{self.next_call_id_int:03d}"
        self.next_call_id_int = (self.next_call_id_int + 1) % 1_000
        return formatted_next_call_id

    async def put_s3_debug_object(
        self, s3_debug_desc: S3DebugDesc, s3_debug_object: S3DebugObject
    ):
        assert not self.finalized
        s3_debut_site: S3DebugSite = (
            datetime.now(timezone.utc),
            get_current_span_context(),
            s3_debug_desc,
        )
        s3_debug_site_object: S3DebugSiteObject = (s3_debut_site, s3_debug_object)
        await self.s3_debug_queue.put(s3_debug_site_object)

    async def put_record_obj(self, patch_record: RagRecord):
        assert not self.finalized
        self.updated_record = merge_records(self.updated_record, patch_record)
        await self.patch_records_queue.put(patch_record)

    async def put_record(
        self,
        *,  # All following params match RagRecord ctor params.
        anns_input: Optional[LlmOutput] = None,
        anns: Optional[Dict[EmbeddingId, RagAnn]] = None,
        anns_summary: Optional[LlmOutput] = None,
        prev_record_summary: Optional[LlmOutput] = None,
        full_history_summary: Optional[LlmOutput] = None,
        rag_output: Optional[LlmOutput] = None,
        events: Optional[List[RagEvent]] = None,
        metrics: Optional[RagMetrics] = None,
    ):
        await self.put_record_obj(
            RagRecord(
                anns_input=anns_input,
                anns=anns,
                anns_summary=anns_summary,
                prev_record_summary=prev_record_summary,
                full_history_summary=full_history_summary,
                rag_output=rag_output,
                events=events,
                metrics=metrics,
            )
        )

    async def put_event(
        self, source: RagEventSource, severity: RagEventSeverity, message: str
    ) -> None:
        await self.put_record(
            events=[RagEvent(source=source, severity=severity, message=message)]
        )

    async def put_user_info_event(self, msg: str) -> None:
        await self.put_event(RagEventSource.User, RagEventSeverity.Info, msg)

    async def put_user_warning_event(self, msg: str) -> None:
        await self.put_event(RagEventSource.User, RagEventSeverity.Warning, msg)

    async def put_user_error_event(self, msg: str) -> None:
        await self.put_event(RagEventSource.User, RagEventSeverity.Error, msg)

    async def put_dependency_info_event(self, msg: str) -> None:
        await self.put_event(RagEventSource.Dependency, RagEventSeverity.Info, msg)

    async def put_dependency_warning_event(self, msg: str) -> None:
        await self.put_event(RagEventSource.Dependency, RagEventSeverity.Warning, msg)

    async def put_dependency_error_event(self, msg: str) -> None:
        await self.put_event(RagEventSource.Dependency, RagEventSeverity.Error, msg)

    async def put_internal_info_event(self, msg: str) -> None:
        await self.put_event(RagEventSource.Internal, RagEventSeverity.Info, msg)

    async def put_internal_warning_event(self, msg: str) -> None:
        await self.put_event(RagEventSource.Internal, RagEventSeverity.Warning, msg)

    async def put_internal_error_event(self, msg: str) -> None:
        await self.put_event(RagEventSource.Internal, RagEventSeverity.Error, msg)
