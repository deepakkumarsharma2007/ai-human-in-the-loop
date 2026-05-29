from pydantic import Field, BaseModel
from typing import List, Optional

class LLMGuardRequestException(Exception):
    pass

class ScannerResult(BaseModel):
    type: str
    value: float | bool
    message: str

class GuardRailsResponse(BaseModel):
    isvalid: bool
    sanitizedstring: Optional[str] = None
    scannerresults: List[ScannerResult] = Field(default_factory=list)
