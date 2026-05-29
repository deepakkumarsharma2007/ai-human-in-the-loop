import os
from typing import Optional

from dotenv import load_dotenv
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.embeddings import Embeddings
from langchain_openai import AzureOpenAIEmbeddings
from langchain_protocol import Any
from pymongo import MongoClient
from pymongo.synchronous.collection import Collection
from functools import lru_cache

@lru_cache(maxsize=1)
def get_mongodb_vector_store(collection: Optional[Collection[dict[str, Any]]] = None, 
                             embeddings: Optional[Embeddings] = None) -> tuple[MongoDBAtlasVectorSearch, Embeddings]:

    load_dotenv()
    embeddings = AzureOpenAIEmbeddings(
        model= os.getenv("DKS_AZURE_EMBED_OPENAI_MODEL_NAME"),
        # azure_deployment= os.getenv("DKS_AZURE_EMBED_OPENAI_DEPLOYMENT"),
        azure_endpoint= os.getenv("DKS_AZURE_EMBED_OPENAI_ENDPOINT"),
        api_key= os.getenv("DKS_AZURE_EMBED_OPENAI_API_KEY"),
        api_version= os.getenv("DKS_AZURE_EMBED_OPENAI_API_VERSION")
        )

    client = MongoClient(os.environ.get("MONGODB_URI"))
    collection = client[os.getenv("MONGO_AZURE_VECTOR_DATABASE_NAME")][os.getenv("MONGO_AZURE_VECTOR_COLLECTION_NAME")]


    vector_store = MongoDBAtlasVectorSearch(collection=collection,
                                            embedding=embeddings,
                                            embedding_key="embedding",
                                            text_key="text",
                                            index_name="vector_index",
                                            relevance_score_fn="cosine")
    return vector_store, embeddings