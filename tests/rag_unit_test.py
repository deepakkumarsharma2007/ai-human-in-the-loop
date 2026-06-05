from pymongo import MongoClient

from genai_core.mongodb.repo.conversation_document_repo import ConversationDocumentRepo


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