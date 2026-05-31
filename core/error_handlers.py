import os


from typing import Dict, Any, Optional
from dotenv import load_dotenv

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.responses import JSONResponse
from core import error_types
from models.requestresponse import ChatRequestErrorResponse
from core.error_types import OrchestratorGraphError
from openai import RateLimitError, BadRequestError
from genai_core.logs.agent_logging import DKSAgentLogger
import copy


# Get app logger object
logger = DKSAgentLogger.get_logger()


def add_request_context_to_error_details(request: Request, details: Dict[str, Any] = {}) -> Dict:
    """
    Add request context data i.e. data from request like conversation id, trace ids from the request 
    into error details.

    Args:
        request(Request): Fastapi request object
        details(Dict[str, Any]): Error details dictionary
    
    Returns:
        (Dict[str, Any]): Augmented error details.
    """    

    # Add conversation id if exsits on request
    if not details.get('conversation_id', None) and hasattr(request.state, "conversation_id"):
        details['conversation_id'] = request.state.conversation_id
    
    # Add message id if exsits on request
    if not details.get('message_id', None) and hasattr(request.state, "message_id"):
        details['message_id'] = request.state.message_id
    
    # Add trace id if exsits on request
    if not details.get('trace_id', None) and hasattr(request.state, "trace_id"):
        details['trace_id'] = request.state.trace_id
    
    return details
    



# Handler for validation errors
def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Handle input validation errors.
    
    Args:
        request(Request): Fastapi request object.
        exc(RequestValidationError): Fastapi request validation error object.
    
    Returns:
        (JSONResponse): Response with status code 400 and content as per ChatRequestErrorResponse.
    """    
    chat_request_validation_error = ChatRequestErrorResponse(error = 'Invalid input', 
                                                             code = 'ValidationError',
                                                             details = {'validation_errors': exc.errors()})
    logger.info(f'Validation error - Invalid Input {chat_request_validation_error.model_dump_json()}')    
    return JSONResponse(
        status_code=400,
        content=chat_request_validation_error.model_dump()
    )

def get_error_response_for_orchestrator_base_error(exception: error_types.OrchestratorBaseError, error_details: Dict[str, Any]) -> ChatRequestErrorResponse:
    """
    Get error response for orchestrator base error for given exception and error details
    """
    if not isinstance(exception, error_types.OrchestratorBaseError):
        error = "Unexpected orchestrator error."
        code = "OrchestratorError"
    else:
        error = exception.error
        code = exception.code

    return ChatRequestErrorResponse(
        error = error,
        code = code,
        details = error_details)


# Handler for custom exceptions
def orchestrator_known_errors_handler(request: Request, exc: error_types.OrchestratorBaseError) -> JSONResponse:
    """
    Handle known errors i.e. errors raised with OrchestratorBaseError. 
    
    Args:
        request(Request): Fastapi request object.
        exc(OrchestratorBaseError): OrchestratorBaseError extended class error object with custom error.
    
    Returns:
        (JSONResponse): Response with status code preset and content as per ChatRequestErrorResponse.
    """
    error_details = getattr(exc, 'error_details', {})
    error_details.update(exc.details)
    error_details = add_request_context_to_error_details(request, details=error_details)    
    chat_request_known_error = get_error_response_for_orchestrator_base_error(exception=exc, error_details=error_details)
    logger.error(f'Known exception - {str(exc)}')
    logger.exception(exc)
    return JSONResponse(
        status_code=exc.status_code,
        content=chat_request_known_error.model_dump()
    )

def get_orchestrator_content_filter_error_response(exception: BadRequestError, error_details: Dict[str, Any]) -> ChatRequestErrorResponse:
    """
    Get the error response instance for llm type exception BadRequestError
    """
    error_details['error_exception'] = str(exception)

    # Extract error message and code from exception if available
    error = (
        "Request couldn't be processed because it may include content that violates our usage guidelines. "
        "Please try rephrasing or removing any sensitive or restricted content."
    )
    code = getattr(exception, "code", getattr(exception, "status_code", None))
    return ChatRequestErrorResponse(
        error=error,
        code=code,
        details=error_details
    )

async def orchestrator_content_filter_errors_handler(request: Request, exc: BadRequestError) -> JSONResponse:
    """
    Handle LLM content filter errors.
    
    Args:
        request(Request): Fastapi request object.
        exc(BadRequestError): Azure OpenAI error object for content filter violations.

    Returns:
        (JSONResponse): Response with status code 400 and content as per ChatRequestErrorResponse.
    """
    # @TODO: Raise similar error from genai core and handle. As this depends on specific library.
    error_details = getattr(exc, 'error_details', {})
    error_details = copy.deepcopy(error_details)  # Make a deep copy to avoid mutating nested structures
    error_details = add_request_context_to_error_details(request, details=error_details)

    chat_error_response_for_quota = get_orchestrator_content_filter_error_response(exception=exc, error_details=error_details)

    return JSONResponse(
        status_code=400,
        content=chat_error_response_for_quota.model_dump()
    )

def get_orchestrator_quota_error_response(exception: RateLimitError, error_details: Dict[str, Any]) -> ChatRequestErrorResponse:
    """
    Get the error response instance for llm type exception RateLimitError
    """
    error_details['error_exception'] = str(exception)

    if not isinstance(exception, RateLimitError):
        error="Unexpected LLM Error"
        code="Unexpected LLM Error"
    elif "insufficient_quota" in str(exception):                               
        error="LLM Quota Exceeded"
        code="AzureOpenAIQuotaExceeded"
    else:
        error="LLM Rate Limit Exceeded"
        code="AzureOpenAIRateLimitExceeded"
        
    return ChatRequestErrorResponse(
        error=error,
        code=code,
        details=error_details
    )  

# Handler for insufficient quta errors
async def orchestrator_quota_errors_handler(request: Request, exc: RateLimitError) -> JSONResponse:
    """
    Handle LLM quota and rate limit error.    
    
    Args:
        request(Request): Fastapi request object.
        exc(RateLimitError): Azure open ai error object on rate limit olr quota error.
    
    Returns:
        (JSONResponse): Response with status code 429 and content as per ChatRequestErrorResponse.
    """
    # @TODO: Raise similar error from genai core and handle. As this depends on specific library. 
    error_details = add_request_context_to_error_details(request)
    error_details['error_exception'] = str(exc)

    chat_error_response_for_quota = get_orchestrator_quota_error_response(exception=exc, error_details=error_details)

    return JSONResponse(
        status_code=429,
        content=chat_error_response_for_quota.model_dump()
    )


def get_unhandled_exception_response(exception: Exception, error_details: Optional[Dict[str, Any]] = {}) -> ChatRequestErrorResponse:
    """
    Get unhandled exception response object for given exception and error details.
    """
    if not error_details:
        error_details = {}
    
    error_details['error_exception'] = str(exception)

    return ChatRequestErrorResponse(
        error="An unexpected error occurred", 
        code = "InternalServerError", 
        details=error_details)
    
# Handlers for unexpected errors 
def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle unexpected errors i.e. errors not raised with AceOrchestratorBaseError. 
    
    Args:
        request(Request): Fastapi request object.
        exc(Exception): Exception object
    
    Returns:
        (JSONResponse): Response with status code 500 and content as per ChatRequestErrorResponse.
    """
    error_details = getattr(exc, 'error_details', {})
    error_details = add_request_context_to_error_details(request, details=error_details)
    error_details['error_exception'] =  str(exc)
    chat_unhandled_error = get_unhandled_exception_response(exception=exc, error_details=error_details)
    logger.error(f'Unexpected error occurred - {str(exc)}')
    logger.exception(exc)
    return JSONResponse(
        status_code=500,
        content=chat_unhandled_error.model_dump()
    )

# Handlers for OrchestratorGraphError
async def react_agent_graph_error_exception_handler(request: Request, exc: OrchestratorGraphError) -> JSONResponse:
    """
    Handle node errors i.e. errors occurred at CoreReactAgent nodes.

    Args:
        request(Request): Fastapi request object.
        exc(Exception): Exception object
    
    Returns:
        (JSONResponse): Response with status code 500 and content as per ChatRequestErrorResponse.
    """
    error_details = exc.details
    error_details['error_exception'] =  exc.error
    chat_node_error = ChatRequestErrorResponse(
        error="Unexpected error occurred during graph execution.",
        code = exc.code,
        details = error_details)
    logger.error(f'Node error occurred - {str(exc)}')
    logger.exception(exc)
    return JSONResponse(
        status_code=500,
        content=chat_node_error.model_dump()
    )

# Handler for HTTP errors
def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """
    Handle HTTPException for returning consistent error schema.


    Args:
        request(Request): Fastapi request object.
        exc(Exception): StarletteHTTPException object
    
    Returns:
        (JSONResponse): Response with error status code and content as per ChatRequestErrorResponse.
    """
    messages = {
        404: "Not found",
        400: "Bad request",
        401: "Unauthorized",
        403: "Forbidden",
        500: "Internal server error"
    }

    error_response = ChatRequestErrorResponse(
        error=messages.get(exc.status_code, "HTTP error"),
        code = "OrchestratorHTTPError",
        details = {}
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump()
    )
