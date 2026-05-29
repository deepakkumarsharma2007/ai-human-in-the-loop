from datetime import datetime
from typing import List
from bson import ObjectId
from pydantic import BaseModel, Field


class ChunkEmbedding(BaseModel):
    chunk: str
    embedding: List[float]
    chunk_size: str


class EmbeddedChunk(BaseModel):
    id: str = Field(alias="_id")
    chunk_name: str
    chunk: str
    embedding: List[float]
    chunk_size: str
    created_date: datetime

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class FileContent(BaseModel):
    file_content: str
    content_type: str

class UserQueryResponse(BaseModel):
    user_query: str
    response: str