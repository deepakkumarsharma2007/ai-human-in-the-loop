from genai_core.chat_history.chat_history import ChatHistory
from genai_core.chat_history.mongodb_chat_history import MongoChatMessageHistoryProvider

def create_agent_chat_history() -> ChatHistory:
    """
    Create instance of ChatHistory
    Provide instance of MongoChatMessageHistoryProvider for Mongo based chat history
    """
    mongo_chat_history = ChatHistory(chat_message_history_provider = MongoChatMessageHistoryProvider())
    return mongo_chat_history

