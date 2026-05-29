from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.documents import Document
from genai_core.rag.common.vector_store import get_mongodb_vector_store


def find_relevant_chunks_from_mongodb_vector_store(query: str):

    if not query or not isinstance(query, str):
        raise ValueError("Query must be a non-empty string.")
    
    vector_store_instance, embeddings = get_mongodb_vector_store()

    results = query_vector_db(vector_store_instance, query)

    return results

def query_vector_db(vector_store: MongoDBAtlasVectorSearch, query: str) -> list[Document]:
    results: list[Document] = vector_store.similarity_search(query, k=2)
    if results and len(results) > 0:
        print(results[0].metadata["title"])
        print(results[0].page_content)
    return results or []