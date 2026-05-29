
import json
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, status, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from genai_core.logs.agent_logging import DKSAgentLogger
from genai_core.agent.audit_context import AuditContext
from models.requestresponse import (
    OrchestratorChatRequest,
    ChatRequestErrorResponse,
    ConversationResponse,
    AllConversationResponseItem,
)
from genai_core.agent.chat_service import ChatService
from genai_core.mongodb.repo.conversation_document_repo import ConversationDocumentRepo
from genai_core.chat_history import ChatHistory
from core.dependencies import (
    get_chat_use_case,
    get_conversation_repo,
    get_conversationhistory_use_case,
)
from core.dependency_agent_chat_history import create_agent_chat_history
from core.feedback_service import FeedbackRepository, get_feedback_repository_dep
from core.azure_ad_auth import AzureADBearer
from core.utils import generate_conversation_id, generate_uuid7_id
from datetime import datetime, timezone
from models.requestresponse import ConversationListResponse, ChatMessage
from models.feedback_models import (
    ChatFeedback,
    FeedbackType,
    POSITIVE_FEEDBACK_SCORE,
    NEGATIVE_FEEDBACK_SCORE,
)
from genai_core.agent.react_orchestrator import (
    get_mcp_tools,
    get_semantic_document_search_tools,
    get_reactorchestrator_agent
)
from genai_core.agent.react_agent import (
    get_document_references_from_data,
    get_document_references_from_datapart,
)
from models.requestresponse import DataPart
from core.error_types import AgentOrchestratorBaseError
from core.error_handlers import (
    get_orchestrator_quota_error_response,
    get_error_response_for_orchestrator_base_error,
    get_unhandled_exception_response,
    get_orchestrator_contentFilter_error_response,
)
from openai import BadRequestError, RateLimitError

# Get app logger object
logger = DKSAgentLogger.get_logger()

security = AzureADBearer()

conversations_api_router = APIRouter(prefix="/conversations", tags=["Conversations"])


def update_auditcontext_with_headers(
    request: Request,
    auditcontext: AuditContext = Depends(security),
) -> AuditContext:
    client_platform = request.headers.get("x-client-platform")
    if client_platform:
        auditcontext.client_platform = client_platform
    # Add more header updates here if needed
    return auditcontext


@conversations_api_router.post(
    "", response_model=OrchestratorChatRequest, status_code=status.HTTP_200_OK
)
async def create_or_send_or_hitl(
    request: OrchestratorChatRequest,
    stream_mode: str = Query("no", description="Enable streaming mode (yes/no)"),
    chat_service: ChatService = Depends(get_chat_use_case),
    auditcontext: AuditContext = Depends(update_auditcontext_with_headers),
) -> OrchestratorChatRequest | ChatRequestErrorResponse | StreamingResponse:
    """Create a conversation or send a chat message."""

    if not request.message or not request.message.parts:
        raise ValueError("User chat request is empty or invalid")

    # Prepare audit context
    auditcontext.references = []
    docreference = None
    if request.message.extensions:
        for ext in request.message.extensions:
            if isinstance(ext, dict) and ext.get("type") == "document":
                data = ext.get("data")
                if data is not None:
                    docreference = get_document_references_from_data(data)
                break
    if len(request.message.parts) > 1:
        for part in request.message.parts:
            if getattr(part, "kind", None) == "data":
                # Only pass DataPart type to get_document_references_from_datapart
                if isinstance(part, DataPart):
                    docreference = get_document_references_from_datapart(part)
                    break
    if docreference:
        auditcontext.references.append(docreference)

    # Get tools for the agent
    tools_from_mcp = await get_mcp_tools(auditcontext=auditcontext)
    tools_from_semantic_search = get_semantic_document_search_tools()
    tools = tools_from_semantic_search + tools_from_mcp

    # Initialize the agent
    agent = await get_reactorchestrator_agent(orchestrator_tools=tools)
    await chat_service.initialize(agent)

    content_parts = request.message.parts
    text_parts = [part for part in content_parts if part.kind == "text"]
    first_content_part = text_parts[0]

    if request.conversationid is None:
        # Create a new conversation
        conversation_id = generate_conversation_id(
            auditcontext.user_alias, first_content_part
        )
        request.conversationid = conversation_id

    auditcontext.session_id = str(request.conversationid)

    if stream_mode != "yes":
        # For non stream reasponse
        agent_message = await chat_service.execute(
            conversationid=request.conversationid,
            user_chat_request=request.message,
            auditcontext=auditcontext,
        )

        return OrchestratorChatRequest(
            conversationid=request.conversationid, message=agent_message
        )
    


    # For stream mode
    # return StreamingResponse(
    #     get_streamed_response(
    #         chat_service, request.conversationid, request.message, auditcontext
    #     ),
    #     media_type="text/event-stream",
    # )


@conversations_api_router.get("", response_model=ConversationListResponse)
async def list_conversations(
    repo: ConversationDocumentRepo = Depends(get_conversation_repo),
    auditcontext: AuditContext = Depends(security),
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(20, ge=1, le=100, description="Max number of items to return"),
):
    """List all conversations with pagination and total count."""
    convos = await repo.find_by_useralias(
        auditcontext.user_alias, skip=skip, limit=limit
    )
    total = await repo.count_by_useralias(auditcontext.user_alias)

    def to_utc_datetime(ts):
        if isinstance(ts, float) or isinstance(ts, int):
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        elif isinstance(ts, str):
            try:
                dt = datetime.fromisoformat(ts)
                return dt.astimezone(timezone.utc).isoformat()
            except Exception:
                return ts
        elif isinstance(ts, datetime):
            return ts.astimezone(timezone.utc).isoformat()
        return str(ts)

    items = [
        AllConversationResponseItem(
            conversationid=c.id,
            conversationname=c.title,
            lastupdateddate=to_utc_datetime(c.lastupdatedon),
        )
        for c in convos
    ]
    return ConversationListResponse(items=items, total=total)


@conversations_api_router.get("/{id}", response_model=ConversationResponse)
async def get_conversation(
    id: str,
    use_case: ChatService = Depends(get_conversationhistory_use_case),
    repo: ConversationDocumentRepo = Depends(get_conversation_repo),
    chathistoryclient: ChatHistory = Depends(create_agent_chat_history),
    auditcontext: AuditContext = Depends(security),
    request: Request = Request,
):
    """Get conversation by ID."""

    if not id or not str(id).strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Conversation ID must not be empty",
        )

    conversation = await repo.find_by_id(id)  # To check if conversation exists

    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found"
        )

    if conversation.useralias != auditcontext.user_alias:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this conversation",
        )

    request.state.conversation_id = id
    convo = await use_case.get_chathistorybyconversationid(
        conversation_id=id, chathistoryclient=chathistoryclient, conversationrepo=repo
    )
    if not convo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found"
        )
    return convo