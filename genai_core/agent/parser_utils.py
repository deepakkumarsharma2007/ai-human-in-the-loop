"""
Module for parser utilities using singleton pattern.
Provides cached instances of output parsers to improve performance and reduce memory usage.
"""
from functools import lru_cache
from langchain_core.output_parsers import PydanticOutputParser
from models.outputresponsemodel import OutputResponseModel

@lru_cache(maxsize=1)
def get_output_parser():
    """
    Singleton factory for PydanticOutputParser.
    Returns the same parser instance for all requests.
    Thread-safe due to @lru_cache.
    """
    return PydanticOutputParser(pydantic_object=OutputResponseModel)