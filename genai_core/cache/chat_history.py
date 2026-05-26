import os
import redis
from typing import List
from langchain.schema import AIMessage, HumanMessage, BaseMessage
from slb_aegis_genai.core.cache.aegisredischatmessagehistory import AegisRedisChatMessageHistory

class AegisChatHistory:
    def __init__(self):
        """Initialize Redis client and set TTL for sessions."""
        # Read Redis configuration from environment variables
        REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
        REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
        REDIS_USER = os.getenv("REDIS_USER", "default")
        REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")

        # Connect using redis-py
        self.client = redis.StrictRedis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            username=REDIS_USER,
            password=REDIS_PASSWORD,
            decode_responses=True,
            ssl=True  # Required for Azure Enterprise Redis
        )

    def _get_redis_url(self) -> str | None:
        """Construct Redis URL from environment variable."""
        return os.getenv("REDIS_URL", None)

    def _serialize_message(self, message: BaseMessage) -> dict:
        """Serialize a BaseMessage to a dictionary."""
        return {
            "type": type(message).__name__,
            "data": message.model_dump()
        }

    def _deserialize_message(self, message_dict: dict) -> BaseMessage:
        """Deserialize a dictionary to a BaseMessage."""
        message_type = message_dict["type"]
        if message_type == "AIMessage":
            return AIMessage(**message_dict["data"])
        elif message_type == "HumanMessage":
            return HumanMessage(**message_dict["data"])
        else:
            raise ValueError(f"Unsupported message type: {message_type}")

    def get_history(self, session_id: str) -> List[BaseMessage]:
        """Retrieve chat history from Redis."""
        redishistory = AegisRedisChatMessageHistory(session_id, redis_client=self.client)
        return redishistory.messages

    def add_message(self, session_id: str, message: BaseMessage):
        """Add a new message to chat history."""
        redishistory = AegisRedisChatMessageHistory(session_id, redis_client=self.client)
        redishistory.add_message(message)

    def clear_history(self, session_id: str):
        """Clear session history."""
        redishistory = AegisRedisChatMessageHistory(session_id, redis_client=self.client)
        redishistory.clear()