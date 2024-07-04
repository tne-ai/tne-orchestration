import asyncio
from typing import Dict, AsyncGenerator

class GraphAIAgent:
    def __init__(self, callback=None):
        self._callback = callback or self._noop
        self.orig_question = ""

    def _noop(self, callback_type, data):
        pass

    def _process_event(self, callback_type, data):
        self._callback(callback_type, data)

                    
    async def agents(
            self,
            category: str,
            agent_id: str,
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
            
        
