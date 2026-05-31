import os
import json
from fastapi import Depends
from genai_core.agent.chat_service import ChatService
from langchain_openai import AzureOpenAIEmbeddings
from genai_core.mongodb.repo.conversation_document_repo import ConversationDocumentRepo
from pymongo import MongoClient
from functools import lru_cache

@lru_cache(maxsize=1)
def get_conversation_repo():
    """Singleton dependency injector for the conversation repository."""
    connection_string = os.environ.get('MONGODB_URI')
    if not connection_string:
        raise EnvironmentError("MONGODB_URI environment variable not set")
    mongo_client = MongoClient(connection_string)
    return ConversationDocumentRepo(
        client=mongo_client,
        db_name=os.environ.get('MONGO_CHATHISTORY_DB_NAME', 'ChatHistoryDB'),
        collection_name=os.environ.get('MONGO_CHATHISTORY_COLLECTION_NAME', 'chathistory')
    )

async def get_chat_use_case(repo: ConversationDocumentRepo = Depends(get_conversation_repo)) -> ChatService:
    """Dependency injector for the chat use case."""
    chat_service = ChatService(repo)
    return chat_service

async def get_conversationhistory_use_case(repo: ConversationDocumentRepo = Depends(get_conversation_repo)) -> ChatService:
    """Dependency injector for the chat use case."""
    chat_service = ChatService(repo)
    return chat_service