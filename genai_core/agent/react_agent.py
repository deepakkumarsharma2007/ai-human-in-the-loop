from asyncio import tools
import os
from typing import Any, List, Optional
from uuid import uuid7
from dotenv import load_dotenv
from langchain.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import MessagesState
from langsmith.run_helpers import get_current_run_trace
from pydantic import BaseModel, Field
from genai_core.agent.agent_base import AgentBase
from genai_core.agent.audit_context import AuditContext
from genai_core.agent.exceptions.agent_exception import AgentException
from genai_core.agent.model_base import ModelBase
from genai_core.agent.shared_agent_state import SharedAgentState
from genai_core.agent.utils.utils import generate_uuid7_id
from genai_core.chat_history.chat_history import ChatHistory
from genai_core.logs.agent_logging import AgentLogger
from genai_core.logs.conversational_logger import ConversationLoggerAdapter

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




            checkpointer: InMemorySaver = InMemorySaver(),
            is_checkpointer_enabled : bool = False,
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
        self.is_checkpointer_enabled = is_checkpointer_enabled
        self.logger = AgentLogger.get_logger()

        self.chat_history_client = chat_history_client

        if summarizer_llm is None:
            raise AgentException("LLM 'summarizer_llm' must be provided")
        self.summarizer_llm = summarizer_llm
        self.construct_workflow()
        # self.create_model(name, 1000 if (max_completion_tokens == None or max_completion_tokens <= 0)  else max_completion_tokens)
        

    def construct_workflow(self) -> None:
        # Implement the logic to construct the workflow for the ReAct agent



        # add retry


        pass

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

        except AgentException as e:
            self.logger.error(f"AgentException occurred: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error occurred: {e}")
            raise

    def set_user_session_details(self, user_query: str, session_id: str, audit_context: dict) -> tuple[HumanMessage, str]:
        current_run = get_current_run_trace()
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


    # def chat_model_azureopenai(self):
       
    #     messages = [
    #         (
    #             "system",
    #         "You are a helpful assistant that acts as stock market research analyst"
    #         )
    #         ,(
    #             "human",
    #             "Tell me why Palantair was down on 01-01-2019"
    #         )
    #     ]
        
    #     response: BaseModel = self.llm.invoke(messages)
        
    #     print(response.content)
    #     print(response)
   
    # def conversation_history(self):
    #     from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        
    #     conversation = [
    #         SystemMessage("You are a helpful assistant that translates English to French."),
    #         HumanMessage("Translate: I love programming."),
    #         AIMessage("J'adore la programmation."),
    #         HumanMessage("Translate: I love building applications.")
    #     ]
        
    #     response = self.llm.invoke(conversation)
    #     print(response)
        
    # def create_model(self, name: str, max_completion_tokens: int):
    #     if not os.environ.get("DKS_AZURE_OPENAI_API_KEY"):
    #         raise ValueError('Missing DKS_AZURE_OPENAI_API_KEY')

    #     self.llm = AzureChatOpenAI(
    #         name=name,
    #         azure_endpoint=os.environ["DKS_AZURE_OPENAI_ENDPOINT"]
    #         ,azure_deployment=os.environ["DKS_AZURE_OPENAI_DEPLOYMENT_NAME"]
    #         ,openai_api_version=os.environ["DKS_AZURE_OPENAI_API_VERSION"]
    #         ,api_key=os.environ['DKS_AZURE_OPENAI_API_KEY']
    #         ,verbose=True
    #         , temperature=0.5
    #         , max_retries=1
    #         , max_completion_tokens= max_completion_tokens
    #     )