from genai_core.chat_history.chat_history import ChatHistory
from models.conversationinfo import ConversationInfo
from models.requestresponse import ConversationResponse, ConversationMessage, TextPart, ChatMessage, DataPart, FilePart
from datetime import datetime, timezone
from genai_core.agent.react_orchestrator import execute_agent
from genai_core.mongodb.repo.conversation_document_repo import ConversationDocumentRepo
from core.audit_context import AuditContext
from langchain_core.messages import BaseMessage
import uuid
from typing import List, Dict, Any, AsyncGenerator
from fastapi.exceptions import RequestValidationError
from core.utils import generate_uuid7_id, uuidv7_to_datetime
from genai_core.agent.react_agent import CoreReActAgent
from core.error_types import OrchestratorGraphError, AgentOrchestratorBaseError

# Define or import DKS_ORCHESTRATOR_APP_NAME
DKS_ORCHESTRATOR_APP_NAME = "DKSOrchestrator"  # Replace with the actual app name if needed

class ChatService:

    """Handles the logic for creating/continuing conversations and generating responses."""
    def __init__(self, conversation_repository: ConversationDocumentRepo):    
        self.repo = conversation_repository
    
    async def initialize(self, orchestratoragent: CoreReActAgent):
        self.orchestrator_agent = orchestratoragent

    async def execute_for_agent_response(self, conversationid: str, user_chat_request: ChatMessage, auditcontext: AuditContext) -> ChatMessage:
        pass

    async def create_update_conversation(self, 
                                         conversationid:str, 
                                         user_prompt:str, 
                                         auditcontext:AuditContext) -> None:
        """
        Create or Update conversation for given conversation id.

        Args:
            conversationid (str): Identifer to uniquely identify chat conversation.
            user_prompt (str): User chat query text.
            auditcontext (AuditContext): User auditcontext and auth data.
        Returns:
            (None)
        """
        if not conversationid:
            return None
        
        conversation = await self.repo.find_by_id(conversationid)
        if not conversation:
            conversation = ConversationInfo(
                _id=conversationid,
                app=DKS_ORCHESTRATOR_APP_NAME,
                title=user_prompt if user_prompt else "Untitled Conversation",
                useralias=auditcontext.user_alias or "",
                client_platform=auditcontext.client_platform,
                lastupdatedon=datetime.now(timezone.utc).timestamp()
            )
        else:
            conversation.lastupdatedon = datetime.now(timezone.utc).timestamp()
        await self.repo.save(conversation)

                            
    async def execute(self, conversationid: str, user_chat_request: ChatMessage, auditcontext: AuditContext) -> ChatMessage:
        """Executes the chat use case, either creating a new conversation or continuing an existing one."""

        if not user_chat_request or not user_chat_request.parts:
            raise RequestValidationError("User chat request is empty or invalid")

        user_prompt = ' '.join(part.text for part in user_chat_request.parts if part.kind == "text")

        # Create or Update the conversation
        await self.create_update_conversation(conversationid, user_prompt, auditcontext)
        
        # Prepare the agent response content
        response = await execute_agent(self.orchestrator_agent, user_prompt, auditcontext)

       
        # Succesful response
        if response and isinstance(response, dict) and 'message' in response:
            responsemessageid = response.get('chat_message_id', generate_uuid7_id())
            response_cits = response.get('citations', [])
            response_artifact_parts = response.get('response_artifact_parts', [])
            response_runid = [{"type":"run_id", "value": response.get('run_id', None)}]

            return ChatMessage(
                role="agent",
                id=responsemessageid,
                parts=[
                    TextPart(
                        kind="text",
                        text=response.get('message', 'No response.') if response else f"This is a mock agent response to: {user_prompt}"
                    )
                ] + response_artifact_parts,
                extensions = response_runid + response_cits
            )

        # Otherwise
        # Return a default error UserChatResponse if no valid response is generated
        raise AgentOrchestratorBaseError(code="OrchestratorError", message="No valid response from agent.", details={})
    
    async def get_chathistorybyconversationid(self, conversation_id: str,
                                        chathistoryclient: ChatHistory,
                                        conversationrepo: ConversationDocumentRepo) -> ConversationResponse:
        """
        Retrieve chat history for a given conversation ID and user alias.

        Args:
            conversation_id (str): The ID of the conversation.
            user_alias (str): The alias of the user.

        Returns:
            Optional[list]: A list of chat messages if found, otherwise None.
        """    

        conversation = await conversationrepo.find_by_id(conversation_id)
        if not conversation:
            raise ValueError("Conversation does not exist")
        
        # Fetch chat history from Redis
        # We set filter_for_errors=False to include all messages, including any error messages, in the chat history.
        chat_history: List[BaseMessage] = chathistoryclient.get_history(session_id=conversation_id, filter_for_errors=False)
        if not chat_history:
            return ConversationResponse(
                messages=[]
            )

        # Create a list of ConversationMessage objects from the base message
        def base_message_to_agent_response(message: BaseMessage) -> ConversationMessage:
            # Extract content and references if present
            content = (
                [TextPart(text=str(item)) for item in message.content if isinstance(item, str)]
                if isinstance(message.content, list)
                else [TextPart(text=str(message.content))]
            )
            
            response_artifacts = message.additional_kwargs.get("response_artifact_parts", [])
            response_artifact_parts = []
            for artifact in response_artifacts:
                if artifact.get("kind") == "data":
                    response_artifact_parts.append(DataPart(kind="data", data=artifact.get("data")))
                elif artifact.get("kind") == "file":
                    response_artifact_parts.append(FilePart(kind="file", file=artifact.get("file")))
           
            content.extend(response_artifact_parts)
            extensions = message.additional_kwargs.get("references", [])
            
            # Add run id to extensions
            extensions.append({
                "type": "run_id",
                "value": message.additional_kwargs.get("run_id", None)
            })

            # Map message.type to 'user' or 'agent'
            role = message.type
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "agent"
            createddatetime=(
                    uuidv7_to_datetime(str(message.additional_kwargs.get("messageid")))
                    if message.additional_kwargs.get("messageid") not in [None, ""]
                        and (
                            # Check if messageid is a valid UUIDv7
                            isinstance(message.additional_kwargs.get("messageid"), str)
                            and (
                                lambda mid: (
                                    uuid.UUID(mid).version == 7
                                ) if (
                                    # Try to parse as UUID, catch exceptions
                                    mid is not None and mid != ""
                                ) else False
                            )(str(message.additional_kwargs.get("messageid")))
                        )
                    else datetime.now(timezone.utc)
                )            
            return ConversationMessage(
                id=str(message.additional_kwargs.get("messageid")) 
                    if message.additional_kwargs.get("messageid") not in [None, ""] 
                    else generate_uuid7_id(),
                role=role,
                parts=[part for part in content],  # Ensures correct type for List[TextPart | FilePart | DataPart]
                extensions=extensions,
                createddatetime=createddatetime
            )

        chat_messages = [
            base_message_to_agent_response(message)
            for message in chat_history
        ]

        # Create a Conversation object
        return ConversationResponse(
            messages=chat_messages)