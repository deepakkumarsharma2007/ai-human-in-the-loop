

import os
import asyncio
from aiohttp import request
from langchain_openai import AzureChatOpenAI
from pymongo import MongoClient
import pytest
from dotenv import load_dotenv

from core.utils import generate_conversation_id
from models.parts import TextPart
from models.requestresponse import ChatMessage, OrchestratorChatRequest
from tests.rag_unit_test import get_conversation_repo

load_dotenv()

# from core.audit_context import AuditContext
from core.audit_context import AuditContext
from genai_core.agent.chat_service import ChatService
from genai_core.agent.react_orchestrator import get_mcp_tools, get_reactorchestrator_agent, get_reactorchestrator_agent, get_semantic_document_search_tools
from genai_core.mongodb.repo.conversation_document_repo import ConversationDocumentRepo


EVAL_PROMPT: str = """You are a helpful and precise assistant for checking the quality of the answer. 
Please check if the answer is correct based on the question and reference answer. 
The question, reference answer, and answer to check are provided below.

Expected Response: {expected_response}
Actual Response: {actual_response}
---

(Answer with 'true' or 'false' only, without any explanation or additional text. The answer should be in lowercase.)
Does the actual response match the expected response? Answer 'true' if they match, and 'false' if they do not match.
"""


@pytest.mark.asyncio
async def test_llm_as_judge_top_level_network_section_test():
    question = "What are the five top-level groups for network section?"
    expected_response = "DNS, MOST COMMON ARCHITECTURE, DC CONNECTIVITY, ZONING, ROUTING."
    assert await query_and_validate(question, expected_response)
    

async def query_and_validate(question: str, expected_response: str) -> bool:


    response_text = await get_agent_response(question)
    prompt = EVAL_PROMPT.format(expected_response=expected_response, actual_response=response_text)
    # print(prompt)
    llm = AzureChatOpenAI() # gpt4, can use open source model here as well
    evaluation_response = llm.invoke(prompt).strip().lower()
    print(f"Evaluation Response: {evaluation_response}")

    if "true" in evaluation_response:
        print("The actual response matches the expected response.")
        return True
    elif "false" in evaluation_response:
        print("The actual response does NOT match the expected response.")
        return False
    else:
        raise ValueError(f"Unexpected evaluation response: {evaluation_response}. Expected 'true' or 'false'.")


async def get_agent_response(request_message: str):
    auditcontext =  AuditContext(user_oid="anonymous-user-987987-8768768", 
                        user_alias="deepakkumarsharma2007", 
                        session_id="", 
                        authinfo="dummy_bearer_token_for_testing", 
                        user_name="Deepak Kumar", 
                        email="deepakkumarsharma2007@gmail.com")

    repo = get_conversation_repo()
    chat_service = ChatService(repo)

    # Get tools for the agent
    tools_from_mcp = await get_mcp_tools(auditcontext=auditcontext)
    tools_from_semantic_search = await get_semantic_document_search_tools(auditcontext=auditcontext)
    tools = tools_from_semantic_search + tools_from_mcp

    # Initialize the agent
    agent = await get_reactorchestrator_agent(orchestrator_tools=tools)
    await chat_service.initialize(agent)

    
    auditcontext.session_id = "dummy id for testing"
    request = OrchestratorChatRequest(auditcontext.session_id, 
                ChatMessage(role="user",parts=[TextPart(text=request_message)]))

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

    agent_message = await chat_service.execute(
            conversationid=auditcontext.session_id,
            user_chat_request= request.message,
            auditcontext=auditcontext,
        )
    if(agent_message):
        return agent_message.parts[0].text
    return None
        
