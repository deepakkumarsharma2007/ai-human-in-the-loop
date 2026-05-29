
from datetime import datetime
import math
from typing import Dict, List
import logging
import uuid

from langchain_community.document_loaders import PyPDFLoader
from genai_core.rag.common import vector_store
from genai_core.rag.models.data_models import ChunkEmbedding, EmbeddedChunk, FileContent
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter



def chunk_texts_split_documents(documents: list[Document]) -> list[Document]:
    try:
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
        )
        return text_splitter.split_documents(documents)

    except Exception as e:
        logging.error(f"Error in chunk_texts: {e}")
        raise e
    
    
def bulk_insert_chunk_embeddings(
    chunk_embeddings: List[EmbeddedChunk], source: str
) -> List[str]:
    # Convert EmbeddedChunk objects to dictionaries using dict(by_alias=True)
    chunk_embeddings_dicts = [
        chunk.model_dump(by_alias=True) for chunk in chunk_embeddings
    ]

    vector_store_instance, embeddings = vector_store.get_mongodb_vector_store()
    
    docs = [Document(metadata= chunk, page_content=chunk["chunk"]) for chunk in chunk_embeddings_dicts]

    result = vector_store_instance.add_documents(docs) #, [chunk["_id"] for chunk in chunk_embeddings_dicts])
    return result



def load_documents() -> list[Document]:
    file_name = "genai_core\\rag\\books\\Azure Cloud Native Architecture.pdf"
    # file_name = "genai_core\\rag\\books\\Agentic-AI-MAS-Threat-Modelling-Guide-v1-FINAL.pdf"
    
    document_loader = PyPDFLoader(file_name)
    return document_loader.load()


def create_embeddings_of_chunks(chunks: list):
    # Prepare the data as a list of chunks
    # print(f"Chunks of create_embeddings_of_chunks: {chunks}")

    vector_store_instance, embeddings = vector_store.get_mongodb_vector_store()
    response_data: list[list[float]] = embeddings.embed_documents([chunk.page_content for chunk in chunks])  # Assuming this method returns a list of embeddings corresponding to the chunks


    # Create the result array with chunk, embeddings, and chunk size
    chunks_embeddings: list[ChunkEmbedding] = []
    for chunk, embedding in zip(chunks, response_data):
        chunks_embeddings.append(
            {
                "chunk": chunk.page_content,
                "embedding": embedding,
                "chunk_size": len(chunk.page_content),
            }
        )

    return chunks_embeddings


def insert_chunks_embeddings(
    article_name: str, chunks_embeddings: list[ChunkEmbedding], source: str
):
    chunk_embeddings: List[EmbeddedChunk] = []
    current_time = datetime.now()

    chunk_embeddings = [
        EmbeddedChunk(
            id=f"{article_name}_{uuid.uuid4()}",  # Combine file_name and a unique UUID
            created_date=current_time,
            chunk_name=article_name,
            chunk=chunk["chunk"],
            embedding=chunk["embedding"],
            chunk_size=convert_size(chunk["chunk_size"]),
        )
        for i, chunk in enumerate(chunks_embeddings)
    ]
    inserted_ids = bulk_insert_chunk_embeddings(chunk_embeddings, source)
    return {
        "article_name": article_name,
        "inserted_ids": inserted_ids,
    }


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"
