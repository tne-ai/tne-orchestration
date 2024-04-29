import pandas as pd
from typing import Optional
from pydantic import BaseModel, Json


class _Base(BaseModel):
    pass


class LLMResponse(BaseModel):
    text: Optional[str] = None
    """Text outputted by the LLM"""
    data: Optional[pd.DataFrame] = None
    """Data outputted by the LLM"""

    class Config:
        arbitrary_types_allowed = True
