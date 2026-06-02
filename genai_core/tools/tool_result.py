from typing import Any
from pydantic import BaseModel

class ToolResult(BaseModel):
    """
    Represents the result of a tool execution.

    Attributes:
        result (str): The main result or output of the tool.
        resulttype (str): The type or category of the result.
        sharedresult (dict[str, Any]): Additional shared data or metadata related to the result.
    """
    result: str
    resulttype: str
    sharedresult: dict[str, Any]