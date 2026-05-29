from pymongo import MongoClient
from pymongo.collection import Collection, ReturnDocument
from typing import Optional
import time
from pymongo import DESCENDING
from models.conversationinfo import ConversationInfo
import asyncio


class ConversationDocumentRepo:
    """A repository that uses a Pydantic model for data operations."""
    def __init__(self, client: MongoClient, db_name: str, collection_name: str):
        self.client = client
        self.collection: Collection = self.client[db_name][collection_name]

    async def count_by_useralias(self, useralias: str) -> int:
        """Returns the total count of conversations for a useralias."""
        return await asyncio.to_thread(lambda: self.collection.count_documents({'useralias': useralias}))

    async def find_by_useralias(
            self, useralias: str, skip: int = 0, limit: int = 20
        ) -> list[ConversationInfo]:
            """Finds all conversations for a given useralias with pagination."""
            cursor = await asyncio.to_thread(
                lambda: self.collection.find({'useralias': useralias}).sort("lastupdatedon", DESCENDING).skip(skip).limit(limit)
            )
            documents = list(cursor)
            return [ConversationInfo.model_validate(doc) for doc in documents]

    async def save(self, conversation: ConversationInfo) -> ConversationInfo:
        """Saves a conversation to the database (creates or updates)."""
        conversation.lastupdatedon = time.time()
        doc = conversation.model_dump(by_alias=True)
        # Run the sync method in a thread to make it awaitable
        await asyncio.to_thread(self.collection.replace_one, {'_id': doc['_id']}, doc, upsert=True)
        return conversation

    async def find_by_id(self, conversation_id: str) -> Optional[ConversationInfo]:
        """Finds a single conversation by its ID."""
        document = await asyncio.to_thread(self.collection.find_one, {'_id': conversation_id})
        if document:
            return ConversationInfo.model_validate(document)
        return None

    async def update_title(self, conversation_id: str, new_title: str) -> Optional[ConversationInfo]:
        """Updates only the title of a specific conversation."""
        result_doc = await asyncio.to_thread(
            self.collection.find_one_and_update,
            {'_id': conversation_id},
            {'$set': {'title': new_title, 'lastupdatedon': time.time()}},
            return_document=ReturnDocument.AFTER
        )
        if result_doc:
            return ConversationInfo.model_validate(result_doc)
        return None

    async def delete_by_id(self, conversation_id: str) -> int:
        """Deletes a conversation by its ID."""
        result = await asyncio.to_thread(self.collection.delete_one, {'_id': conversation_id})
        return result.deleted_count