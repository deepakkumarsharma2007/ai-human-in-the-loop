from typing import List
from langchain.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.chat_history import BaseChatMessageHistory
from genai_core.chat_history.chat_history_base import ChatMessageHistoryProvider

class ChatHistory:

    def __init__(self, chat_message_history_provider: ChatMessageHistoryProvider):
        """
        Initialize Chat History Provider.
        Args:
            chat_message_history_provider (ChatMessageHistoryProvider): wrapper instance for the chat message history provider.
        """
        self.chat_message_history_provider = chat_message_history_provider

        if not self._verify_parents():
            raise ValueError("Parent class mismatch. Expected ChatMessageHistoryProvider")

    def _verify_parents(self) -> bool:
        """
        Verify if self.chat_message_history_provider is inherited from ChatMessageHistoryProvider

        Returns:
        bool: True if both parent classes are inherited, False otherwise.
        """
        if isinstance(self.chat_message_history_provider, ChatMessageHistoryProvider):
            return True
        else:
            return False

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
        """
        Retrieve chat history from Chat History Provider.
        Args:
            session_id (str): The session ID for which to retrieve the chat history.
        Returns:
            List[BaseMessage]: A list of BaseMessage objects representing the chat history.
        """
        history_client = self.chat_message_history_provider.get_instance(session_id=session_id)
        if not isinstance(history_client, BaseChatMessageHistory):
            raise TypeError("Client must be an instance of BaseChatMessageHistory")
        return history_client.messages

    def add_message(self, session_id: str, message: BaseMessage):
        """
        Add a new message to chat history.
        Args:
            session_id (str): The session ID for which to add the message.
            message (BaseMessage): The message to be added to the chat history.
        """
        history_client = self.chat_message_history_provider.get_instance(session_id=session_id)
        if not isinstance(history_client, BaseChatMessageHistory):
            raise TypeError("Client must be an instance of BaseChatMessageHistory")
        history_client.add_message(message)

    def clear_history(self, session_id: str):
        """
        Clear session history.
        Args:
            session_id (str): The session ID for which to clear the chat history.
        """
        history_client = self.chat_message_history_provider.get_instance(session_id=session_id)
        if not isinstance(history_client, BaseChatMessageHistory):
            raise TypeError("Client must be an instance of BaseChatMessageHistory")
        history_client.clear()