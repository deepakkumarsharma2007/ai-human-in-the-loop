import os
from pymongo import MongoClient
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from genai_core.chat_history.chat_history_base import ChatMessageHistoryMixin, ChatMessageHistoryProvider

class MongoDBChatMessageHistory(MongoDBChatMessageHistory, ChatMessageHistoryMixin):
    """
    Class for langchain's MongoDBChatMessageHistory.
    Inherits from MongoDBChatMessageHistory and ChatMessageHistoryMixin.
    Add methods or properties as needed for your specific use case.
    """
    pass
 
class MongoChatMessageHistoryProvider(ChatMessageHistoryProvider):

    """
    MongoChatMessageHistoryProvider is a MongoDB-based chat message history client.
    Inherited from ChatMessageHistoryProvider.
    It provides methods to create a chat history client and transform AI messages with additional arguments.
    """

    def __init__(self,
                    database_name:str = None,
                    collection_name:str = None,
                    connection_string:str = None) -> None:
        """
        Init method initializes the DB client connection.
        init DOES NOT create a BaseChatMessageHistory client.
        creation of instance of BaseChatMesssageHistory is delegated to method 'get_instance'
        MANDATORY: Create a DATABASE connection. Assign it as self.db_connection

        Attributes:
            database_name (str): The name of the MongoDB database.
            collection_name (str): The name of the MongoDB collection.
            connection_string (str): The MongoDB connection string.
        """
        self.connection_string = os.getenv("MONGO_CHATHISTORY_CONNECTION_STRING") if connection_string is None else connection_string
        self.database_name = os.getenv("MONGO_CHATHISTORY_DB_NAME") if database_name is None else database_name
        self.collection_name = os.getenv("MONGO_CHATHISTORY_COLLECTION_NAME") if collection_name is None else collection_name

        # MANDATORY: Create a DATABASE connection
        self.db_connection = MongoClient(self.connection_string)

        
    def get_instance(self, session_id: str) -> MongoDBChatMessageHistory:
        """
        Create a MongoDBChatMessageHistory instance for the given session ID and client
        Uses the initialized MongoDB client connection.
        Attributes:
            session_id (str): The session ID for which to create the client.
        Returns:
            MongoDBChatMessageHistory: An instance of MongoDBChatMessageHistory.
        """

        if self.db_connection is None:
            raise ValueError("Mongo client cannot be None")

        mongodb_chat_history_service = MongoDBChatMessageHistory(
            session_id=session_id,
            client=self.db_connection,
            database_name=self.database_name,
            collection_name=self.collection_name,
            connection_string=None,
            create_index=False
        )
        return mongodb_chat_history_service