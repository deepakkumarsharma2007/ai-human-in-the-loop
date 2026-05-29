import time
from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone

class ConversationInfo(BaseModel):
    """A Pydantic model to represent and validate a conversation document."""
    
    # The 'alias' tells Pydantic to map this field to '_id' in the database.
    # The default_factory can be used to generate IDs, e.g., default_factory=str(uuid.uuid4)
    id: str = Field(alias='_id')
    useralias: str
    title: str
    app: str
    client_platform: Optional[str] = None
    additional_kwargs: Optional[dict] = None
    
    # Pydantic uses 'default_factory' just like dataclasses
    createdon: float = Field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    lastupdatedon: float = Field(default_factory=lambda: datetime.now(timezone.utc).timestamp())

    # This configuration allows Pydantic to work with attribute access
    class Config:
        populate_by_name = True