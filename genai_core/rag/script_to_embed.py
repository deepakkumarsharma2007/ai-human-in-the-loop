import time

from langchain_core.documents import Document
from genai_core.rag.embedding.embedding_generator import chunk_texts_split_documents, create_embeddings_of_chunks, insert_chunks_embeddings, load_documents
from genai_core.rag.common.vector_store import get_mongodb_vector_store

def add_documents_to_vector_store(split_docs: list[Document]):
    
    vector_store_instance,embeddings  = get_mongodb_vector_store()
    vector_store_instance.add_documents(split_docs)


def embed():

    start_time = time.time()
    file_name = "Azure_Cloud_Native_Architecture.pdf"
    documents = load_documents()
    # Further processing of documents, such as generating embeddings, can be done here.
    print(f"Actual Data:\n{documents[0].page_content}\n")
    print(f"Meta Data:\n{documents[0].metadata}\n")

    chunk_texts = chunk_texts_split_documents(documents)


    list_of_chunk_embeddings = create_embeddings_of_chunks(chunk_texts)

    print(f"Split Data:\n{chunk_texts[0].page_content}\n")
    print(f"Meta Data:\n{chunk_texts[0].metadata}\n")

    # add_documents_to_vector_store(check_texts)

    result = insert_chunks_embeddings(
            file_name, list_of_chunk_embeddings, "pdf"
        )
    
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken: {time_taken} seconds")


# if __name__ == "__main__":
#     main()