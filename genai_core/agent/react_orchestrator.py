import os
from typing import Dict, Any, AsyncGenerator
from core.models.azurechatopenai import AzureChatOpenAiModel
from core.dependency_agent_chat_history import create_agent_chat_history
from core.utils import generate_uuid7_id
from genai_core.agent.agents_prompts import DKS_AGENT_CAPABILITIES, DKS_AGENT_SYSTEM_PROMPT, DKS_AGENT_TYPICAL_TASKS
from genai_core.agent.check_pointer import CheckPointer
from genai_core.agent.react_agent import CoreReActAgent

from core import error_types
from genai_core.rag.tools.document_search_tool import mongodb_document_search_agent_adapter
from genai_core.agent.mcp_client_utils import (
    get_mcp_tools_with_server,
    StructuredToolToBaseToolAdapter,
    langchain_structured_tool_to_base_tool,
)
from langchain_mcp_adapters.sessions import StreamableHttpConnection

from core.audit_context import AuditContext
from datetime import timedelta
from langsmith import traceable
from langsmith import Client


# GenAI lens for tracing
# gen_ai_lens = genai_lens.GenAILens()


async def get_mcp_tools(
    auditcontext: AuditContext, transport: str = "streamable_http"
) -> list[StructuredToolToBaseToolAdapter]:
    """
    Returns a list of Document MCP agent tools from the MCP server.
    Requires MCP_RAG_MONGO_NATURAL_LANGUAGE_SERVER_URL environment variable and AuditContext.
    """
    mcp_enabled = str(os.environ.get("MCP_AGENT_ENABLED")).lower() == "true"
    if not mcp_enabled:
        return []
    MCP_RAG_MONGO_NATURAL_LANGUAGE_SERVER_URL = os.environ.get("MCP_RAG_MONGO_NATURAL_LANGUAGE_SERVER_URL")
    if MCP_RAG_MONGO_NATURAL_LANGUAGE_SERVER_URL is None:
        return []
    mcp_connection = StreamableHttpConnection(
        transport=transport,
        url=MCP_RAG_MONGO_NATURAL_LANGUAGE_SERVER_URL,
        timeout=timedelta(minutes=5),
        headers={"Authorization": "Bearer " + auditcontext.additional_args.get("authinfo", "dummy bearer token for testing")},
    )
    tools = await get_mcp_tools_with_server(connection=mcp_connection)
    return tools


async def get_semantic_document_search_tools(
    auditcontext: AuditContext
) -> list[StructuredToolToBaseToolAdapter]:
    
    tool_adapter = mongodb_document_search_agent_adapter()
    tools = langchain_structured_tool_to_base_tool([tool_adapter])
    return tools


async def get_reactorchestrator_agent(
    orchestrator_tools: list[StructuredToolToBaseToolAdapter],
) -> CoreReActAgent:
    # Create Tools / Business workflows
    chat_history = create_agent_chat_history()

    # Create Azure Chat OpenAI
    llm = AzureChatOpenAiModel()


    model_name = os.getenv("DKS_AZURE_OPENAI_DEPLOYMENT_NAME")
    deployment_name = os.getenv("DKS_AZURE_OPENAI_DEPLOYMENT_NAME")
    api_key = os.getenv("DKS_AZURE_OPENAI_API_KEY")
    api_base = os.getenv("DKS_AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("DKS_AZURE_OPENAI_API_VERSION")





    # make a smaller LLM for summarization and other smaller tasks to save cost, while using the bigger model for agent reasoning



    summarize_llm = AzureChatOpenAiModel(
        model_name=model_name,
        deployment_name=deployment_name,
        api_key=api_key,
        api_base=api_base,
        api_version=api_version,
    )

    # Set agent cache if semantic cache is enabled
    agent_cache = None

    # Create CoreReActAgent
    return CoreReActAgent(
        name="DKS Orchestrator Agent",
        capability=DKS_AGENT_CAPABILITIES,
        description=DKS_AGENT_SYSTEM_PROMPT,
        typical_task=DKS_AGENT_TYPICAL_TASKS,
        tools=orchestrator_tools,
        llm=llm,
        chat_history_client=chat_history,


        checkpointer_type= CheckPointer.REDIS,
        # agent_cache=agent_cache,


        summarizer_llm=summarize_llm,
    )


async def execute_agent(agent, userprompt, auditcontext: AuditContext):
    updatedauditcontext = auditcontext.to_dict()
    updatedauditcontext["additional_args"].pop("authinfo", None)
    response = await agent.execute(
        userprompt, sessionid=auditcontext.session_id, auditContext=updatedauditcontext
    )
    if response is None:
        return {
            "conversation_id": auditcontext.session_id,
            "error": "No response from agent",
        }

    if "agentresult" in response:
        responsemessageid = generate_uuid7_id()
        if "agentmessage" in response:
            # Extracting the parent message ID and response message ID from the response
            # parent_message_id = response['agentmessage'].additional_kwargs.get("parentmessageid", "")
            responsemessageid = response["agentmessage"].additional_kwargs.get(
                "messageid", responsemessageid
            )
        return {
            "conversation_id": auditcontext.session_id,
            "run_id": response.get("run_id", None),
            "chat_message_id": responsemessageid,
            "role": "agent",
            "message": response["agentresult"],
            # "chat_parent_message_id": parent_message_id,
            "citations": response.get("references", []),
            "response_artifact_parts": response.get("response_artifact_parts", []),
            "execution_path": response.get("execution_path", []),
        }
    # Otherwise, returns None
    # invalid result from agent


# async def execute_agent_with_stream(
#     agent, userprompt, auditcontext
# ) -> AsyncGenerator[Dict[str, Any], None]:
#     """
#     Trigger Agent execution with stream.
#     """
#     updatedauditcontext = auditcontext.to_dict()
#     updatedauditcontext["additional_args"].pop("authinfo", None)
#     async for event in agent.execute_with_stream(
#         userprompt, sessionid=auditcontext.session_id, auditContext=updatedauditcontext
#     ):
#         yield event
