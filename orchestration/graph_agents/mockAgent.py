import asyncio
from typing import Dict, AsyncGenerator

async def mockAgent(
        inputs: Dict,
        params: Dict,
        uid: str,
        session_id: str = "",
) -> AsyncGenerator:
    """Manages LLM agents (more documentation forthcoming)."""
    for i in range(10):
        await asyncio.sleep(1.0)
        yield f"{i}"

    yield "___END___";
    yield "0123456789";        
