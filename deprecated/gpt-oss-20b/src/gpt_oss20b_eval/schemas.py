from typing import List, Optional
from pydantic import BaseModel, Field


class TestCase(BaseModel):
    id: str
    category: str = Field(default="core")
    prompt: str
    expected: Optional[str] = None
    checks: List[str] = Field(default_factory=lambda: ["non_empty"])


class EvalResult(BaseModel):
    id: str
    category: str
    prompt: str
    response: str
    checks: dict
    passed: bool
    latency_ms: int
