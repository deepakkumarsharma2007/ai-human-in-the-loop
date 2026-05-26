from abc import ABC, abstractmethod
from typing import List
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.messages.tool import ToolMessage

class ChatMessageHistoryProvider(ABC):
    """
    Abstract base class for Aegis chat message history provider.
    Provides a template for creating chat history clients.

    Init method initializes the DB client connection which can be reused while initializing BaseChatMessageHistory instance.
    init DOES NOT create a BaseChatMessageHistory client.
    creation of instance of BaseChatMesssageHistory is delegated to method 'get_instance'

    """

    @abstractmethod
    def get_instance(self, session_id: str) -> BaseChatMessageHistory:
        """
        Create a chat history client for the given session ID.
        
        Args:
            session_id (str): The session ID for which to create the client

        Returns:
            An instance of BaseChatMessageHistory.
        """
        pass

class AegisChatMessageHistoryMixin:

    def transform_message(self, message: AIMessage) -> BaseMessage:
        """
        Transform an AIMessage with additional_args into a desired format.
        Modify this method as per your transformation logic.
        """
        # Example transformation logic (replace with actual logic as needed)
        if "tool_call_id" not in message.additional_kwargs:
            return message
        # Set the 'transformed' flag in the additional_kwargs
        return ToolMessage(content=message.content,
                            tool_call_id=message.additional_kwargs.get("tool_call_id", "unknown"),
                            id=message.additional_kwargs.get("id", "unknown"))

    @property
    def messages(self) -> List[BaseMessage]:
        """
        Get the list of messages from the chat history and convert AI messages with additional_kwargs.
        This method applies transformation logic on default messages property
        """
        tmpMessages = super().messages
        transformed_messages = []
        for message in tmpMessages:
            if isinstance(message, AIMessage) and hasattr(message, 'additional_kwargs'):
                # Perform transformation on AIMessage with additional_kwargs
                transformed_message = self.transform_message(message)
                transformed_messages.append(transformed_message)
            else:
                transformed_messages.append(message)
        return transformed_messages