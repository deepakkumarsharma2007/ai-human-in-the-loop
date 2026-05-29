from pydantic import BaseModel, Field, model_validator
from typing import Literal, Dict, Any


class TextPart(BaseModel):
    """
    Represents a text message part for Ace agent communication.
    """
    kind: Literal["text"] = Field(
        default="text",
        description="The type of the message part. Always 'text' for TextPart."
    )
    text: str = Field(
        ...,
        description="The textual content of the message part.",
        min_length=1
    )


class FilePart(BaseModel):
    """
    Represents a file message part for Ace agent communication.
    """
    kind: Literal["file"] = Field(
        default="file",
        description="The type of the message part. Always 'file' for FilePart."
    )
    file: Dict[str, Any] = Field(
        ...,
        description="A dictionary containing file metadata and content."
    )

    @model_validator(mode="after")
    def check_either_one(cls, values):
        uri, bytes_ = values.file.get("uri"), values.file.get("bytes")
        if (uri is None and bytes_ is None) or (uri is not None and bytes_ is not None):
            raise ValueError("Exactly one of uri or bytes must be provided.")
        return values


class DataPart(BaseModel):
    """
    Represents a data message part for Ace agent communication.
    """
    kind: Literal["data"] = Field(
        default="data",
        description="The type of the message part. Always 'data' for DataPart."
    )
    data: Dict[str, Any] = Field(
        ...,
        description="A dictionary containing arbitrary structured data."
    )

