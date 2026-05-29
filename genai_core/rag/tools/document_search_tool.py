import json
from logging import Logger
import logging
import os

from pydantic import BaseModel, Field
from typing import Any, Optional
from langchain_core.tools import BaseTool

# from core.agent import create_model, create_react_agent
# from core.auth_middleware import audit_info_decorator
# from genai_core.log_exceptions_middleware import log_exceptions_decorator
from urllib.parse import quote

from genai_core.logs.agent_logging import DKSAgentLogger
from genai_core.rag.embedding.rag_vector_search import find_relevant_chunks_from_mongodb_vector_store

class MongoDBRAGSearchAdapterSchema(BaseModel):
    query: str = Field(
        ...,
        description="Users query in natural language. The query should be related to document search in MongoDB databases.",
    ),
    # auditcontext: Optional[dict[str, Any]] = Field(
    #     default_factory=dict, description="auditcontext of the current request"
    # )


class MongoDBRAGSearchToolAdapter(BaseTool):
    """
    MongoDBRAGSearchToolAdapter is an agent adapter tool designed to handle natural language queries related to document search in MongoDB Databases.
        - It receives a natural language query and an audit context.
    """

    name: str = "MongoDB_RAG_Document_Search_Query_Tool"
    description: str = """This tool is designed to handle user query to search documents in MongoDB databases.
    Does semantic search in MongoDB RAG based index. It takes a natural language query as input and returns the search results from MongoDB RAG based index.
    It can query the MongoDB vector store that contains the document chunks and their corresponding embeddings to find the most relevant documents based on the user's natural language query.
    It takes a natural language query as input and returns the search results from MongoDB RAG based index.
    
    It can query the MongoDB vector store that contains the document chunks and their corresponding embeddings 
    to find the most relevant documents based on the user's natural language query.
    """
    
    args_schema: type[MongoDBRAGSearchAdapterSchema] = MongoDBRAGSearchAdapterSchema
    logger: Logger = DKSAgentLogger.get_logger(__name__)

    # @audit_info_decorator()
    # @log_exceptions_decorator()
    async def _arun(
        self,
        query: str,
        # ctx: Context,
        auditcontext: Optional[dict[str, Any]] = None,
    ) -> str:
        query = quote(query, safe="")
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string.")
    
        result = self.find_documents_rag_search(query)

        return result if isinstance(result, str) else json.dumps(result)

    def _run(self) -> Any:
            raise NotImplementedError("This tool is async only. Use _arun method.")
    
    def find_documents_rag_search(self, user_query: str):
        """
        Does semantic search in MongoDB RAG based index.
        """
        try:
            # Log the user query
            self.logger.debug(f"Received natural language query: {user_query}")

            results = find_relevant_chunks_from_mongodb_vector_store(user_query)
        
            return [result.page_content for result in results]
        
        except Exception as e:
            self.logger.error(f"Error processing natural language query {user_query} \n\n exception: {e}")
            raise

def mongodb_document_search_agent_adapter() -> MongoDBRAGSearchToolAdapter:
    """
    Returns MongoDB Natural Language Query agent adapter tool instance.
    """
    mongodburl = os.getenv("MONGODB_URI")
    if mongodburl is None:
        raise ValueError(
            "Environment variable 'MONGODB_URI' is not set."
        )
    mongodb_database_name = os.getenv("MONGODB_NATURAL_LANGUAGE_DATABASE_NAME", "")
    if not mongodb_database_name:
        raise ValueError(
            "Environment variable 'MONGODB_NATURAL_LANGUAGE_DATABASE_NAME' is not set or empty."
        )
    mongodb_document_search_agent_adapter = MongoDBRAGSearchToolAdapter()

    return mongodb_document_search_agent_adapter