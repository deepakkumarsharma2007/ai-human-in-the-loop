from asyncio import tools
import json
import os
from typing import Any, Dict, List, Optional
from uuid import uuid7
from dotenv import load_dotenv
from langchain.messages import HumanMessage
# from langchain_openai import AzureChatOpenAI
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph, RetryPolicy
from langsmith.run_helpers import get_current_run_tree
from openai import BadRequestError
from pydantic import BaseModel, Field
from genai_core.agent.agent_base import AgentBase
from core.audit_context import AuditContext
from genai_core.agent.check_pointer import CheckPointer
from genai_core.agent.exceptions.agent_exception import AgentException
from core.models.model_base import ModelBase
from langchain_core.messages.base import BaseMessage
from langchain_core.messages import ToolMessage
from genai_core.agent.shared_agent_state import SharedAgentState
from genai_core.agent.tool_executor_custom import CustomToolExecutorNode
from genai_core.agent.utils.utils import generate_uuid7_id
from genai_core.cache.mongodb_checkpointer import MongoDBCheckPointer
from genai_core.cache.redis_checkpointer import RedisCheckPointer
from genai_core.chat_history.chat_history import ChatHistory
from genai_core.logs.agent_logging import DKSAgentLogger
from genai_core.logs.conversational_logger import ConversationLoggerAdapter
from models.parts import DataPart



TOOL_MENTION_PROCESSOR_NODE = "tool_mention_processor"
TOOL_SELECTOR_NODE = "tool_selector"
COMBINE_RESULTS_NODE = "combine_results"
EXECUTE_TOOLS_NODE = "execute_tools"
UPDATE_SEMANTIC_CACHE_NODE = "update_semantic_cache_node"
SEMANTIC_CACHE_CONDITION_NODE = "semantic_cache_condition_node"
GET_FROM_SEMANTIC_CACHE_NODE = "get_from_semantic_cache_node"
UPDATE_CHAT_HISTORY_NODE = "update_chat_history_node"
SHOULD_CONTINUE_ITERATION_NODE = "should_continue_iteration_node"


class CoreReActAgent(AgentBase):
    
    def __init__(
            self, 
            name: str,
            description: str,
            capability: str,
            typical_task: str,
            llm: ModelBase,
            tools: List[BaseTool],
            
            
            chat_history_client: Optional[ChatHistory] = None,
            checkpointer_type = CheckPointer.NONE,




            max_completion_tokens: Optional[int] = None,
            summarizer_llm: Optional[ModelBase] = None
            ):
        super().__init__(os.name, description, capability, typical_task, llm)
        

        if(not tools or not isinstance(tools, list)
           or not all(isinstance(tool, BaseTool) for tool in tools)
           ):
            raise AgentException("Tools must be a non-empty list of BaseTool instances.")

        if not isinstance(llm, ModelBase):
            raise AgentException("LLM must be an instance of ModelBase.")

        self.tools = tools
        self.llm = llm
        self.logger = DKSAgentLogger.get_logger()

        self.chat_history_client = chat_history_client
        self.checkpointer_type = checkpointer_type

        if summarizer_llm is None:
            raise AgentException("LLM 'summarizer_llm' must be provided")
        self.summarizer_llm = summarizer_llm
        self.construct_workflow()
        # self.create_model(name, 1000 if (max_completion_tokens == None or max_completion_tokens <= 0)  else max_completion_tokens)
        

    def construct_workflow(self) -> None:
        # Implement the logic to construct the workflow for the ReAct agent

        workflow = StateGraph(SharedAgentState)

        workflow.add_node(GET_FROM_SEMANTIC_CACHE_NODE, self.get_from_semantic_cache_node, retry_policy= RetryPolicy(max_attempts=2))
        workflow.add_node(TOOL_MENTION_PROCESSOR_NODE, self.tool_mention_processor)
        workflow.add_node(UPDATE_SEMANTIC_CACHE_NODE, self.update_semantic_cache_node)
        workflow.add_node(TOOL_SELECTOR_NODE, self.decide_tools_with_llm)
        tool_node = CustomToolExecutorNode(self.tools)
        workflow.add_node(EXECUTE_TOOLS_NODE, tool_node)
        workflow.add_node(SHOULD_CONTINUE_ITERATION_NODE, self.should_continue_iterate)
        workflow.add_node(COMBINE_RESULTS_NODE, self.combine_results)
        workflow.add_node(UPDATE_SEMANTIC_CACHE_NODE, self.update_semantic_cache_node)
        workflow.add_node(UPDATE_CHAT_HISTORY_NODE, self.update_chat_history_node)


        if self.checkpointer_type == CheckPointer.REDIS:
                self.graph: CompiledStateGraph = workflow.compile(
                    checkpointer= RedisCheckPointer().set_redis_checkpointer()
                )
                return
        elif self.checkpointer_type == CheckPointer.MONGODB:
                self.graph: CompiledStateGraph = workflow.compile(
                    checkpointer= MongoDBCheckPointer().set_mongodb_checkpointer()
                )
                return
        self.graph: CompiledStateGraph = workflow.compile()


        

    async def async_execute(self, user_query: str, session_id: str,  audit_context: dict
                            ) -> dict[str, Any]:
        # Implement the logic to execute the workflow for the ReAct agent
        try:
            # Validate user query
            if not user_query or not isinstance(user_query, str):
                raise AgentException("User query must be a non-empty string.")

            # Log the user query
            self.logger.info(f"Starting execution of ReAct agent - Received user query: {user_query}")

            user_message, run_id = self.set_user_session_details(user_query, session_id, audit_context)

            self.logger = ConversationLoggerAdapter(
                self.logger,
                {
                    "convid": session_id,
                    "messageid": user_message.additional_kwargs["messageid"],
                    "query": user_query,
                },
            )

            initial_state: MessagesState = SharedAgentState(
                {
                    'messages': [user_message],
                    'shareddata': []
                }
            )

            config: RunnableConfig = RunnableConfig(
                {
                    "thread_id": session_id,
                    "run_id": str(run_id),
                    "audit_context": audit_context,
                    
                },
                recursion_limit=100,
                max_concurrency=10
            )

            result = await self.graph.ainvoke(initial_state, config=config)
            last_message = result.get("messages", [])[-1] if result.get("messages") else None

            final_result = {}

            # Execute the workflow and get the response
            response = {}  # Placeholder for actual response from workflow execution

            # Log the response
            self.logger.info(f"Generated response: {response}")

            return response

        except BadRequestError as e:
            self.logger.error(f"BadRequestError occurred: {e}")
            self.logger.error(f"An error occurred during execution: {e.args}")
            if session_id and self.chat_history_client is not None:
                errordict = {
                    "errormessage": (
                        getattr(e.response, "text", None)
                        if getattr(e, "response", None)
                        else str(e)
                    ),
                    "errorcode": getattr(e, "code", getattr(e, "status_code", None)),
                }
                # Ensure correct argument order and all required arguments are provided
                self.chat_history_client.update_message(
                    session_id, user_message.additional_kwargs["messageid"], errordict
                )
                self.logger.debug("Chat history node - Cleared chat message.")
            e.error_details = {"query": user_query, "conversation_id": session_id}
            raise
        except AgentException as e:
            self.logger.error(f"AgentException occurred: {e}")
            e.error_details = {"query": user_query, "conversation_id": session_id}
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error occurred: {e}")
            e.error_details = {"query": user_query, "conversation_id": session_id}
            raise

    def set_user_session_details(self, user_query: str, session_id: str, audit_context: dict) -> tuple[HumanMessage, str]:
        current_run = get_current_run_tree()
        run_id = current_run.run_id if current_run else None
        user_message = HumanMessage(content=user_query, 
                                        additional_kwargs={"messageid": run_id})
        user_message.additional_kwargs["sessionid"] = run_id
        user_message.additional_kwargs["messageid"] = generate_uuid7_id()
        references = audit_context.get("references", [])
        if isinstance(references, list):
            user_message.additional_kwargs["references"] = references.copy()
        else:
            user_message.additional_kwargs["references"] = []
        user_message.additional_kwargs["run_id"] = str(run_id)

        if audit_context is not None:
            if isinstance(audit_context, AuditContext):
                audit_context = audit_context.to_dict()

        if session_id:
            self.chat_history_client.add_message(session_id, user_message)
            self.logger.info(f"User message added to chat history for session_id: {session_id}")
        return user_message, run_id

def get_artifacts_from_messages(messages: List[BaseMessage]):
    """
    Extracts artifacts from messages and combines them together.
    Returns:
        tuple: A tuple containing:
            - artifacts (list): Regular artifacts from messages.
            - response_artifacts (list): Artifacts marked as agent responses.
    """
    artifacts = []
    response_artifacts = []
    for message in messages:

        if not isinstance(message, ToolMessage):
            continue

        if not getattr(message, "artifact", None):
            continue

        if not isinstance(message.artifact, list):
            continue

        for artifact_item in message.artifact:
            if hasattr(artifact_item, "metadata"):
                if "agent_response" in artifact_item.metadata.get("source", ""):
                    response_artifacts.append(artifact_item)
                    continue

            artifacts.append(artifact_item)

    return artifacts, response_artifacts


def get_references_response_artifact_parts(chat_response_references: List[Dict]):

    # Get artifacts and response artifacts from chat response references
    chat_response_references_ = [
        ref
        for ref in chat_response_references
        if ref.get("type") != "response_artifacts"
    ]
    response_artifacts = [
        ref
        for ref in chat_response_references
        if ref.get("type") == "response_artifacts"
    ]

    # extract part from response artifacts
    response_artifact_parts = []
    for response_artifact in response_artifacts:
        resp_artifacts = response_artifact["artifacts"]
        for resp_artifact in resp_artifacts:
            response_artifact_parts = response_artifact_parts + getattr(
                resp_artifact, "parts", []
            )

    return chat_response_references_, response_artifact_parts


def get_execution_path_items_from_messages(messages: List[BaseMessage]):
    """ "
    Get execution path from state
    """
    execution_path_items = []

    # FOr every message in messages
    # Process for extracting execution path items
    for message in messages:

        # Only process message with tool calls
        if not getattr(message, "tool_calls", None):
            continue

        # For each message with tool call extract tool call
        # As tool calls are execution path
        for tool_call in message.tool_calls:
            tool_args = tool_call["args"].copy()

            # Remove the auditcontext and shared tool inputs form tool args as
            # Not required
            if tool_args.get("auditcontext"):
                del tool_args["auditcontext"]
            if tool_args.get("sharedtoolinputs"):
                del tool_args["sharedtoolinputs"]

            # Add if agent is registerd
            if tool_call["name"]:
                execution_path_items.append(
                    {"name": tool_call["name"], "info": {"args": tool_args}}
                )

    # Return the prepared execution path items
    return execution_path_items


def get_tags_from_context_and_response(auditContext, response_cits):
    """
    Private method to get tags from auditContext and response_cits.
    Returns a list of tags.
    """
    tags = []
    response_tags = get_tags_from_execution_path(response_cits)
    if response_tags:
        tags.extend(response_tags)
    if (
        auditContext
        and isinstance(auditContext, dict)
        and auditContext.get("client_platform")
    ):
        tags.append(f"Platform:{auditContext['client_platform']}")
    return tags


def get_reference_weblinks_from_message(message: BaseMessage) -> list[dict]:
    """
    Parses message content as json.

    Args:
        message_content (str): Message content to parse and get references from.

    Returns:
        list[dict]: List of dictionaries where each dictionary representing reference.
    """
    # Parse JSON, fall back to empty references on error.

    # Try decoding as json
    try:
        content_json = json.loads(message.content)
    except json.JSONDecodeError as e:
        return []

    # Handle error in processing references
    try:
        if type(content_json) != list:
            return []

        content_json_msgs = content_json
        references = []

        for content_json_msg in content_json_msgs:
            if (
                content_json_msg.get("result_type") != "json"
                or not content_json_msg.get("references")
                or not type(content_json_msg.get("references")) == list
            ):
                continue
            else:
                references = references + content_json_msg.get("references", [])

        # Remove duplicates
        references = list({r["title"]: r for r in references}.values())

    except Exception as e:
        return []

    return references


def get_reference_weblinks_from_messages(messages: List[BaseMessage]):
    """
    Get reference weblinks from messages.

    """
    weblinks = []
    for message in messages:
        if type(message) != ToolMessage:
            continue
        weblinks.append(get_reference_weblinks_from_message(message))

    # Need to merge weblinks as it is list of lists
    if len(weblinks) == 0:
        return weblinks

    weblinks_merged = [item for sublist in weblinks for item in sublist]

    # Remove duplicates from merged list
    weblinks_unique = list(
        {weblink["title"]: weblink for weblink in weblinks_merged}.values()
    )

    return weblinks_unique


def get_references_from_messages(
    messages: list[dict], message_limit: int = 100
) -> list[dict]:
    """
    Returns references from messages. Processes messages in reverse order, last first.

    Args:
        messages (list[dict]): List of dictionaries where each dictionary represents message - ToolMessage/AIMessage/HumanMessage.
        message_limit (int): No. of entries to process for retriveing references.

    Return:
        list[dict]: List of dictionaries where each dictionary represents reference.

    """

    # Set message to process
    # As messages are revered and we need to get in limit window
    messages_to_process = []
    for message_index, message in enumerate(reversed(messages)):
        if message_index > message_limit:
            break
        if isinstance(message, HumanMessage):
            break
        messages_to_process.append(message)

    # Reverse to reset original order
    messages_to_process = list(reversed(messages))

    # Get web links and execution path items from message to process
    web_links = get_reference_weblinks_from_messages(messages_to_process)
    execution_path_items = get_execution_path_items_from_messages(messages_to_process)
    artifacts_from_messages, response_artifacts = get_artifacts_from_messages(
        messages_to_process
    )

    return [
        {"type": "web_links", "links": web_links},
        {"type": "execution_path", "agents": execution_path_items},
        {"type": "artifacts", "artifacts": artifacts_from_messages},
        {"type": "response_artifacts", "artifacts": response_artifacts},
    ]


def get_document_references_from_datapart(datapart: DataPart) -> dict | None:
    """
    Extracts document references from a DataPart object.

    Args:
        datapart (DataPart): The DataPart object to extract references from.

    Returns:
        DocumentReferenceModel: A model containing document references.
    """
    document_references = {"documentIds": [], "documentFiles": []}
    if datapart and datapart.data:
        document_ids = datapart.data.get("documentIds", [])
        document_files = datapart.data.get("documentFiles", [])
        if document_ids or document_files:
            document_references = {
                "documentIds": document_ids,
                "documentFiles": document_files,
            }
            return {"type": "document", "data": document_references}
    return None


def get_document_references_from_data(dataval: Dict[str, Any]) -> dict | None:
    """
    Extracts document references from a data dictionary.

    Args:
        dataval (Dict[str, Any]): The data dictionary to extract references from.

    Returns:
        dict | None: A dictionary containing document references, or None if no references are found.
    """
    document_references = {"documentIds": [], "documentFiles": []}
    if dataval:
        document_ids = dataval.get("documentIds", [])
        document_files = dataval.get("documentFiles", [])
        if document_ids or document_files:
            document_references = {
                "documentIds": document_ids,
                "documentFiles": document_files,
            }
            return {"type": "document", "data": document_references}
    return None


def get_tags_from_execution_path(references: Any) -> list[str] | None:
    """
    Returns a list of tags from the execution path items.
    Each tag is formatted as "Agent:{name}" where name is from the execution path item.

    Args:
        references (Any): List of reference dictionaries, expected to include an 'execution_path' type.

    Returns:
        list[str] | None: List of tags, or None if not found.
    """
    if references:
        execution_path_cits = [
            cit for cit in references if cit.get("type") == "execution_path"
        ]
        if execution_path_cits and isinstance(execution_path_cits[0], dict):
            executed_agents = execution_path_cits[0].get("agents", [])
            tags = []
            for agent in executed_agents:
                name = agent.get("name")
                if name:
                    tags.append(f"Agent:{name}")
            return tags
        return None
    return None
