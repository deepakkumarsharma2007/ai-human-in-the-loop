# import math
# import re
# from typing import List
# import PyPDF2
# from pymongo import MongoClient
# from config.connection import vector_db_collection
# from models.data_models import EmbeddedChunk, FileContent
# from models.constants import pdf_file, md_file


# def convert_size(size_bytes):
#     if size_bytes == 0:
#         return "0B"
#     size_name = ("B", "KB", "MB", "GB", "TB")
#     i = int(math.floor(math.log(size_bytes, 1024)))
#     p = math.pow(1024, i)
#     s = round(size_bytes / p, 2)
#     return f"{s} {size_name[i]}"


# def bulk_insert_chunk_embeddings(
#     chunk_embeddings: List[EmbeddedChunk], source: str
# ) -> List[str]:
#     # Convert EmbeddedChunk objects to dictionaries using dict(by_alias=True)
#     chunk_embeddings_dicts = [
#         chunk.model_dump(by_alias=True) for chunk in chunk_embeddings
#     ]

#     # Perform bulk insert
#     result = vector_db_collection.insert_many(chunk_embeddings_dicts)
#     return result.inserted_ids


# def get_file_content(file_path: str, file_name: str) -> FileContent:
#     content_type: str
#     if file_name.lower().endswith(".pdf"):
#         with open(file_path, "rb") as f:
#             pdf_reader = PyPDF2.PdfReader(f)
#             text = ""
#             for page_num in range(len(pdf_reader.pages)):
#                 page = pdf_reader.pages[page_num]
#                 text += page.extract_text()
#             content_type = pdf_file
#     else:
#         with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
#             text = file.read()
#             content_type = md_file

#     return FileContent(file_content=text, content_type=content_type)
