from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal
from core.utils import generate_uuid7_id

class Message(BaseModel):
    """Represents a single message within a conversation."""
    message_id: str = Field(default_factory=generate_uuid7_id)
    parent_message_id: str | None = None
    content: Dict[str, Any]
    references: List[Dict[str, str]] = []
    execution_path: str | None = None
    role: Literal["user", "agent"]

class Conversation(BaseModel):
    """Represents a full conversation with a user."""
    conversation_id: str = Field(default_factory=generate_uuid7_id)
    conversation_name: str = "New Conversation"
    messages: List[Message] = []
    last_updated_date: str