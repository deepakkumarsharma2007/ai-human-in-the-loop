from pydantic import BaseModel, Field
from pydantic.types import UUID
from enum import Enum


# Score values for positive and negative feedback
POSITIVE_FEEDBACK_SCORE = 10
NEGATIVE_FEEDBACK_SCORE = 5


class FeedbackType(str, Enum):
    """
    Model for feedback type
    """
    GOOD = 'good'
    BAD = 'bad'

class ChatFeedback(BaseModel):
    """
    Model to capture chat feedback.
    """
    run_id: UUID = Field(..., description="Run ID of the trace to capture feedback against.")
    comment: str | None = Field(..., max_length=1000, description="Feedback comment") 
    type: FeedbackType = Field(..., description="Feedback type good/bad")
