import datetime
import os
import re
from typing import Any, Dict, List, Literal, Optional, Union
from uuid import uuid7
from dotenv import load_dotenv
from langchain.messages import AIMessage, HumanMessage, ToolCall
from langgraph import types as langgraph_types
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph, RetryPolicy
from langsmith.run_helpers import get_current_run_tree
from openai import BadRequestError
from pydantic import BaseModel, Field
from genai_core.agent.agent_base import AgentBase
from core.audit_context import AuditContext
from genai_core.agent.agents_prompts import SUMMARY_AGENT_SYSTEM_PROMPT
from genai_core.agent.check_pointer import CheckPointer
from genai_core.agent.exceptions.agent_exception import AgentException
from core.models.model_base import ModelBase
from langchain_core.messages.base import BaseMessage
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from genai_core.agent.parser_utils import get_output_parser
from genai_core.agent.shared_agent_state import SharedAgentState
from genai_core.agent.tool_executor_custom import CustomToolExecutorNode
from genai_core.agent.utils.utils import generate_uuid7_id
from genai_core.cache.agent_cache_prompts import PROMPT_TO_CHECK_CAN_QUERY_BE_CACHED
from genai_core.cache.mongodb_checkpointer import MongoDBCheckPointer
from genai_core.cache.redis_checkpointer import RedisCheckPointer
from genai_core.chat_history.chat_history import ChatHistory
from genai_core.logs.agent_logging import DKSAgentLogger
from genai_core.logs.conversational_logger import ConversationLoggerAdapter
from models.outputresponsemodel import OutputResponseModel
from models.parts import DataPart
from genai_core.agent.utils.agent_utils import *



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
            strict_tools: Optional[List[str]] = None,
            sensitive_tools: Optional[List[str]] = None, # which needs redaction
            filtered_agents_to_exclude: Optional[List[str]] = None,
            
            chat_history_client: Optional[ChatHistory] = None,
            checkpointer_type = CheckPointer.NONE,
            agent_cache: Optional[AgentCacheManager] = None,




            max_completion_tokens: Optional[int] = None,
            summarizer_llm: Optional[ModelBase] = None
            ):
        super()._init__(name, description, capability, typical_task, llm)
        

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
        self.chathistorycount = int(os.getenv("CHATHISTORYCOUNT", 15))
        self.maxiterations = int(os.getenv("MAXITERATIONS", 3))

        if summarizer_llm is None:
            raise AgentException("LLM 'summarizer_llm' must be provided")
        self.summarizer_llm = summarizer_llm
        self.construct_workflow()
        # self.create_model(name, 1000 if (max_completion_tokens == None or max_completion_tokens <= 0)  else max_completion_tokens)
        

    def construct_workflow(self) -> None:
        # Implement the logic to construct the workflow for the ReAct agent

        workflow = StateGraph(SharedAgentState)
        
        # Nodes
        workflow.add_node(GET_FROM_SEMANTIC_CACHE_NODE, self.get_from_semantic_cache_node, retry_policy= RetryPolicy(max_attempts=2))
        workflow.add_node(TOOL_MENTION_PROCESSOR_NODE, self.tool_mention_processor)
        workflow.add_node(UPDATE_SEMANTIC_CACHE_NODE, self.update_semantic_cache_node)
        workflow.add_node(TOOL_SELECTOR_NODE, self.decide_tools_with_llm)
        tool_node = CustomToolExecutorNode(self.tools)
        workflow.add_node(EXECUTE_TOOLS_NODE, tool_node)
        workflow.add_node(SHOULD_CONTINUE_ITERATION_NODE, self.should_continue_iteration)
        workflow.add_node(COMBINE_RESULTS_NODE, self.combine_results)
        workflow.add_node(UPDATE_SEMANTIC_CACHE_NODE, self.update_semantic_cache_node)
        workflow.add_node(UPDATE_CHAT_HISTORY_NODE, self.update_chat_history_node)
        
        # Edge
        workflow.add_edge(START, GET_FROM_SEMANTIC_CACHE_NODE)
        workflow.add_conditional_edges(GET_FROM_SEMANTIC_CACHE_NODE, self.semantic_cache_condition_node, {
            "response_from_cache": TOOL_MENTION_PROCESSOR_NODE, # default node if cache hit
            "not_in_cache_goto_tool_selector": UPDATE_CHAT_HISTORY_NODE
        })
        workflow.add_conditional_edges(TOOL_MENTION_PROCESSOR_NODE, self.tool_mention_router,{
            "goto_tool_selector": TOOL_SELECTOR_NODE,
            "execute_tools": EXECUTE_TOOLS_NODE                                     
        })
        workflow.add_edge(TOOL_SELECTOR_NODE, EXECUTE_TOOLS_NODE)
        workflow.add_edge(EXECUTE_TOOLS_NODE, SHOULD_CONTINUE_ITERATION_NODE) # Conditional edge
        workflow.add_edge(COMBINE_RESULTS_NODE, UPDATE_SEMANTIC_CACHE_NODE)
        workflow.add_edge(UPDATE_SEMANTIC_CACHE_NODE, UPDATE_CHAT_HISTORY_NODE)
        workflow.add_edge(UPDATE_CHAT_HISTORY_NODE, END)


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

        self.print_graph_in_png("react_agent_workflow.png")



    def semantic_cache_condition_node(self, state: SharedAgentState) -> Literal["response_from_cache", "not_in_cache_goto_tool_selector"]:
        """
        Routes to respective nodes depending on cache hit/miss.

        Args:
            state(SharedAgentState) : state in the graph

        Returns:
            str : cache hit or miss
        """
        # Get last message
        if len(state["messages"]) < 1:
            error_message = (
                "No message found in state, routing to default node - tool selector."
            )
            self.logger.error(error_message)
            raise ValueError(error_message)

        # Check if last message is from cache response
        # If yes, route to the end, if no return to tool selector
        last_message = state["messages"][-1]
        if type(last_message) != AIMessage or not last_message.additional_kwargs.get(
            "response_from_cache", None
        ):
            return "not_in_cache_goto_tool_selector"

        # otherwise, we got the response from cache. We can stop
        return "response_from_cache"

    def tool_mention_router(self, state: SharedAgentState) -> Literal["execute_tools", "goto_tool_selector"]:
        """
        Routes to respective nodes depending on tool mention success/failure.

        Args:
            state(SharedAgentState) : state in the graph
        """
        # Get last message
        if len(state["messages"]) < 1:
            error_message = (
                "No message found in state, routing to default node - tool selector."
            )
            self.logger.error(error_message)
            raise ValueError(error_message)

        # check if mention specified in query
        # If yes, route to the tool executor, if no return to tool selector
        last_message = state["messages"][-1]
        if (
            type(last_message) != AIMessage
            or last_message.additional_kwargs.get("mentioned_tool_call_success", None)
            == None
        ):
            return "goto_tool_selector"

        return "execute_tools"
        

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
    
    def print_graph_in_png(self, filename: str) -> None:
        
        try:
            from IPython.display import Image, display
            print(self.graph)
            print(self.graph.get_graph(xray=True).draw_mermaid())

            png_data = self.graph.get_graph(xray=True).draw_mermaid_png()

            with open(filename, "wb") as f:
                f.write(png_data)

            print(f"Diagram saved as {filename}")

            display(Image(self.graph.get_graph(xray=True).draw_mermaid_png()))
        except ImportError:
            self.logger.warning("IPython is not installed. Unable to save workflow graph as PNG.")
    
    async def should_continue_iteration(
        self, state: SharedAgentState, config: RunnableConfig
    ) -> langgraph_types.Command:
        """
        Determines whether to continue iterating based on the number of tool calls in the last AIMessage.

        Args:
            state (SharedAgentState): The current state containing messages.

        Returns:
            bool: True if the number of tool calls is less than or equal to maxiterations, False otherwise.
        """
        try:
            self.logger.info("Executing SHOULD_CONTINUE_ITERATION node")
            if not state["messages"]:
                return langgraph_types.Command(graph=langgraph_types.Command.PARENT)

            # If last message is AIMessage, and tool_calls are empty or null, return command to combine results
            last_message = state["messages"][-1]
            if isinstance(last_message, AIMessage) and (
                not getattr(last_message, "tool_calls", None)
            ):
                self.logger.info(
                    "SHOULD_CONTINUE_ITERATION Execution Done. No tool call found. Routing to combine results."
                )
                return langgraph_types.Command(
                    goto=langgraph_types.Send(COMBINE_RESULTS_NODE, state)
                )

            # Iterate messages in reverse to count AIMessage with tool_calls until HumanMessage
            tool_call_count = 0
            invoked_tools = []
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    break
                if (
                    isinstance(msg, AIMessage)
                    and hasattr(msg, "tool_calls")
                    and msg.tool_calls
                ):
                    for tool_call in msg.tool_calls:
                        if tool_call["name"] not in invoked_tools:
                            invoked_tools.append(tool_call["name"])
                    tool_call_count += 1
            self.logger.debug(f"Count of AIMessage with tool_calls: {tool_call_count}")
            enabletoolcalls = tool_call_count <= self.maxiterations

            skip_history_for_strict_tool = True
            if len(invoked_tools) == 1 and invoked_tools[0] in self.ignore_tools:
                skip_history_for_strict_tool = False

            checkpointhistorymessages = []
            for message in reversed(state["messages"]):
                if isinstance(message, HumanMessage):
                    checkpointhistorymessages.append(message)
                    break
                checkpointhistorymessages.append(message)

            if (
                self.checkpointer_type == CheckPointer.NONE
                and self.chat_history_client
                and skip_history_for_strict_tool
            ):
                if len(state["messages"]) > 0:
                    historymessages = self.chat_history_client.get_history(
                        state["messages"][0].additional_kwargs["sessionid"]
                    )
                    if historymessages is not None and len(historymessages) > 0:
                        historymessages = list(reversed(historymessages))
                        # Skip the first history message to avoid duplication with current conversation messages
                        checkpointhistorymessages += historymessages[
                            1 : self.chathistorycount
                        ]
            checkpointhistorymessages.reverse()
            username, email = self.extract_username_alias(config)
            agentprompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
                [
                    SystemMessagePromptTemplate.from_template(
                        SUMMARY_AGENT_SYSTEM_PROMPT
                    ),
                    MessagesPlaceholder(variable_name="history", optional=True),
                ]
            )
            if (
                checkpointhistorymessages is not None
                and len(checkpointhistorymessages) > 0
            ):
                agentprompt = agentprompt.partial(
                    history=checkpointhistorymessages,
                    currentdatetime=datetime.datetime.now().isoformat(),
                    responseformat=get_output_parser().get_format_instructions(),
                    username=username,
                    email=email,
                    STRICT_AGENTS=self.strict_tools,
                )

            # Get references from history messages i.e. history considered for summarizing tool outputs
            chat_response_references = get_references_from_messages(
                checkpointhistorymessages
            )
            prompt_string = await agentprompt.ainvoke({})
            # Filter out excluded agents from tools
            filtered_tools = [
                tool
                for tool in self.tools
                if tool.name not in self.filtered_agents_to_exclude
            ]
            self.summarizerllm = self.summarizerllm.addtools(filtered_tools)
            combined_response = await self.summarizerllm.agenerateresponse(
                prompt_string, enabletoolcalls, failonguardrailscanning=False
            )

            if (
                hasattr(combined_response, "tool_calls")
                and combined_response.tool_calls
            ):
                self.logger.debug(
                    "Combined Response contains tool calls. Routing to execute tools."
                )
                state["messages"].append(
                    self.convert_to_ai_message_with_tool_calls(combined_response)
                )
                self.logger.info(
                    "SHOULD_CONTINUE_ITERATION Execution Done. Routing to tool executor."
                )
                return langgraph_types.Command(
                    goto=langgraph_types.Send(EXECUTE_TOOLS_NODE, state)
                )

            state["messages"].append(combined_response)
            # Set references for the combined_response node
            # setattr(combined_response, 'references', chat_response_references)
            self.logger.info(
                "SHOULD_CONTINUE_ITERATION Execution Done, routing to combine results."
            )
            return langgraph_types.Command(
                goto=langgraph_types.Send(COMBINE_RESULTS_NODE, state)
            )

        except Exception as e:
            self.logger.error(f"An error occurred in should_continue_iteration: {e}")
            raise

    def extract_username_alias(
        self, config: RunnableConfig
    ) -> tuple[str | None, str | None, str | None]:
        """
        Helper method to extract username alias from RunnableConfig.
        """
        auditcontext = {}
        if config is not None:
            auditcontext = config.get("configurable", {}).get("auditcontext", {})
        user_name: str | None = auditcontext.get("additional_args", {}).get(
            "user_name", None
        )
        email: str | None = auditcontext.get("additional_args", {}).get("email", None)
        return user_name, email
    
    def convert_to_ai_message_with_tool_calls(
        self, message: BaseMessage, outputparser: PydanticOutputParser = None
    ) -> AIMessage:
        """
        Converts a BaseMessage to an AIMessage while preserving tool calls.
        """
        try:
            # Extract tool_calls if they exist
            tool_calls = message.tool_calls if hasattr(message, "tool_calls") else []

            # Check if tool_calls are null or empty
            if not tool_calls:
                if outputparser is not None:
                    self.logger.debug("Output parser exists, parsing message content.")
                    output = outputparser.parse(message.content)
                    notoolcall = AIMessage(content=output.answer)
                    notoolcall.additional_kwargs["redact"] = str(output.private_data)
                    return notoolcall
                notoolcall = AIMessage(content=message.content)
                return notoolcall

            # Ensure tool_calls are properly formatted as a list of ToolCall objects
            formatted_tool_calls = [
                ToolCall(name=tc["name"], args=tc["args"], id=tc["id"])
                for tc in tool_calls
            ]
            toolcallmessage = AIMessage(
                content=message.content, tool_calls=formatted_tool_calls
            )
            for formatted_tc in formatted_tool_calls:
                if formatted_tc["name"] in self.sensitive_tools:
                    toolcallmessage.additional_kwargs["redact"] = "true"
            return toolcallmessage
        except Exception as e:
            self.logger.error(
                f"An error occurred in convert_to_ai_message_with_tool_calls: {e}"
            )
            raise AgentException(message.content, e.args)
        
    async def get_from_semantic_cache_node(
        self, state: SharedAgentState, config: RunnableConfig
    ) -> SharedAgentState:
        """
        Checks if user query is in cache, if not found goes to tool selector.
        If found user query found in cache routes to combine_results

        Args:
            state (SharedAgentState): state in graph.

        Returns:
            SharedAgentState: updated state in graph.

        Raises:
            ValueError: When last human message not found.
        """
        self.logger.info(
            "Semantic Cache: Get From Cache - Checking if user query is in cache."
        )

        # Check if agent cache is set, if not route too tool selector
        if not self.agent_cache:
            self.logger.info(
                "Semantic Cache: Get From Cache - Agent cache not enabled."
            )
            return state
        


        # To implement semantic caching

        return state
    

    def extract_session_and_run_id(self, config: RunnableConfig):
        """
        Helper method to extract session_id and run_id from RunnableConfig.
        """
        session_id: str | None = None
        run_id: str | None = None
        if config is not None:
            run_id = config.get("configurable", {}).get("run_id", None)
            session_id = config.get("configurable", {}).get("thread_id", None)
        return session_id, run_id

    def has_private_data(self, response: OutputResponseModel) -> str:
        """
        Helper method to determine if a message contains private data.
        """
        try:
            if str(response.hasemailreference).lower() == "true":
                return str(True).lower()
            elif response.agentsused and any(
                agent in self.sensitive_tools for agent in response.agentsused
            ):
                return str(True).lower()
            return str(False).lower()
        except Exception as e:
            self.logger.error(f"An error occurred in has_private_data: {e}")
            return str(False).lower()

    def extract_username_alias(
        self, config: RunnableConfig
    ) -> tuple[str | None, str | None, str | None]:
        """
        Helper method to extract username alias from RunnableConfig.
        """
        auditcontext = {}
        if config is not None:
            auditcontext = config.get("configurable", {}).get("auditcontext", {})
        user_name: str | None = auditcontext.get("additional_args", {}).get(
            "user_name", None
        )
        email: str | None = auditcontext.get("additional_args", {}).get("email", None)
        return user_name, email

    def get_additional_agent_context(self, messages: List[BaseMessage]) -> str | None:
        """
        Helper method to extract additional agent context from RunnableConfig.
        """
        # Iterate over messages in reverse to find the last self.chathistorycount HumanMessages
        extra_preferred_agent_in_context = None
        for msg in reversed(messages[-self.chathistorycount :]):
            if isinstance(msg, HumanMessage):
                # Ensure msg.content is a string before calling .strip()
                if isinstance(msg.content, str):
                    content = msg.content.strip()
                elif isinstance(msg.content, list):
                    # Join list elements into a string and strip
                    content = " ".join(str(item) for item in msg.content).strip()
                else:
                    content = str(msg.content).strip()
                if content.startswith("@"):
                    # Find the word after '@' until the next space
                    # Find all agent mentions at the start of the message (e.g., "@Agent1 @Agent2 ...")
                    # Only match consecutive @mentions at the beginning of the string
                    start_mentions = re.findall(r"^(@\w+)(?:\s+@(\w+))*", content)
                    if start_mentions:
                        # Flatten the tuple and remove empty strings
                        agents = [m for tup in start_mentions for m in tup if m]
                        # Remove '@' from agent names
                        agents = [agent.lstrip("@") for agent in agents]
                        extra_preferred_agent_in_context = (
                            agents[-1] if agents else None
                        )
                        break
                    else:
                        # Fallback: match a single @mention at the start
                        match = re.match(r"^@(\w+)", content)
                        if match:
                            extra_preferred_agent_in_context = match.group(
                                1
                            )  # Only the agent name, without '@'
                        else:
                            extra_preferred_agent_in_context = None
                    if match:
                        extra_preferred_agent_in_context = match.group(1)
                        break  # Stop as soon as @mention agent is found
        return extra_preferred_agent_in_context

    async def decide_tools_with_llm(
        self, state: SharedAgentState, config: RunnableConfig
    ) -> SharedAgentState:

        self.logger.info("Executing TOOL_SELECTOR_NODE")
        try:
            if not state.get("messages"):
                self.logger.error(
                    "InvalidArgument: 'state' must be a SharedAgentState instance with at least one HumanMessage"
                )
                raise ValueError(
                    "InvalidArgument: 'state' must be a SharedAgentState instance with at least one HumanMessage"
                )

            # Check for sessionid from RunnableConfig if available
            sessionid, runid = self.extract_session_and_run_id(config)
            username, email = self.extract_username_alias(config)
            last_human_message_1 = self.get_last_human_message(state, config)
            if sessionid is None and isinstance(last_human_message_1, HumanMessage):
                sessionid = last_human_message_1.additional_kwargs.get(
                    "sessionid", None
                )

            if self.chat_history_client:
                historymessages = self.chat_history_client.get_history(
                    last_human_message_1.additional_kwargs["sessionid"]
                )
                if historymessages is not None and len(historymessages) > 0:
                    if isinstance(historymessages[-1], HumanMessage):
                        historymessages = historymessages[:-1]
                        
            user_query = last_human_message_1
            tool_descriptions_str = ""
            current_datetime = datetime.datetime.now().isoformat()
            if self.checkpointer_type == CheckPointer.NONE:
                if len(state["messages"]) == 1 and isinstance(
                    state["messages"][0], HumanMessage
                ):
                    agentprompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
                        [
                            SystemMessagePromptTemplate.from_template(
                                "You are an intelligent agent.\n"
                                "Name of the agent : {name}.\n Description: {description}\n"
                                "You are capable of: {capabilities}\n"
                                "Typical tasks: {typicaltasks}\n"
                                "Available Tools: {tooldescription}\n"
                                "Based on the user query, decide which one of the tools to execute.\n"
                                "Based on the chat history, if user wants to execute a tool forecfully then call the tool\n"
                                "If the tool result contains URLs (Web links), Always include the URLs in the final response, do not truncate URLs\n"
                                "User Name: {username}\n"
                                "User Email: {email}\n"
                                "Current DateTime: The current system time is {currentdatetime}\n"
                            ),
                            HumanMessagePromptTemplate.from_template("{query}"),
                        ]
                    )
                    prompt_string = agentprompt.invoke(
                        {
                            "tooldescription": tool_descriptions_str,
                            "name": self.name,
                            "description": self.description,
                            "capabilities": self.capabilities,
                            "typicaltasks": self.typical_tasks,
                            "currentdatetime": current_datetime,
                            "username": username,
                            "email": email,
                            "query": user_query.content,
                        }
                    )
                else:
                    checkpointhistorymessages = []
                    for message in reversed(state["messages"][:-1]):
                        if len(
                            checkpointhistorymessages
                        ) >= self.chathistorycount and isinstance(
                            message, HumanMessage
                        ):
                            checkpointhistorymessages.append(message)
                            break
                        checkpointhistorymessages.append(message)
                    checkpointhistorymessages.reverse()
                    agentprompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
                        [
                            MessagesPlaceholder(
                                variable_name="history",
                                optional=True,
                                n_messages=self.chathistorycount,
                            ),
                            HumanMessagePromptTemplate.from_template("{query}"),
                        ]
                    )
                    if (
                        checkpointhistorymessages is not None
                        and len(checkpointhistorymessages) > 0
                    ):
                        agentprompt = agentprompt.partial(
                            history=checkpointhistorymessages
                        )
                    prompt_string = agentprompt.invoke({"query": user_query})
                    
            else:
                agentprompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
                    [
                        SystemMessagePromptTemplate.from_template(
                            "You are an intelligent agent named: {name}. \n Description: {description}\n"
                            "You are capable of: {capabilities}\n"
                            "Typical tasks: {typicaltasks}\n"
                            "Available Tools: {tooldescription}\n"
                            "Based on the user query, decide which one of the tools to execute.\n"
                            "If the tool result contains URLs (Web links), Always include the URLs in the final response, do not truncate URLs\n"
                            "Based on the chat history, if user wants to execute a tool forecfully then call the tool.\n"
                            "User Name: {username}\n"
                            "User Email: {email}\n"
                            "Current DateTime: The current system time is {currentdatetime}\n"
                            "**Output Response Format and Rules**\n"
                            "- Your response format must strictly comply with the provided JSON schema, Unless a Tool Call Is Requested. \n"
                            "- JSON Output Schema (Strict Adherence): \n"
                            "- You must respond only in valid JSON that conforms exactly to the OutputResponseModel schema below, if and only if you are not invoking a tool or responding with the internet search phrase.\n "
                            "   {responseformat} \n"
                            "- If you are invoking a tool, respond with the appropriate Tool Call format instead of the JSON schema.\n"
                        ),
                        MessagesPlaceholder(
                            variable_name="history",
                            optional=True,
                            n_messages=self.chathistorycount,
                        ),
                        HumanMessagePromptTemplate.from_template("{query}"),
                    ]
                )

                if self.chat_history_client:
                    historymessages = self.chat_history_client.get_history(
                        last_human_message_1.additional_kwargs["sessionid"]
                    )
                    if historymessages is not None and len(historymessages) > 0:
                        if isinstance(historymessages[-1], HumanMessage):
                            historymessages = historymessages[:-1]
                        agentprompt = agentprompt.partial(history=historymessages)

                prompt_string = agentprompt.invoke(
                    {
                        "tooldescription": tool_descriptions_str,
                        "name": self.name,
                        "description": self.description,
                        "capabilities": self.capabilities,
                        "typicaltasks": self.typical_tasks,
                        "currentdatetime": current_datetime,
                        "username": username,
                        "email": email,
                        "responseformat": get_output_parser().get_format_instructions(),
                        "query": user_query.content,
                    }
                )
            self.logger.debug(f"Prompt String: {prompt_string}")
            decision: BaseMessage = await self.llm.agenerateresponse(
                prompt_string, True
            )
            if not decision.tool_calls:
                self.logger.debug("No tool calls found in the decision.")
                # Create an AIMessage with the decision content
                last_human_message = self.get_last_human_message(state, config)
                if last_human_message is None:
                    self.logger.error(
                        "No HumanMessage found in the state to create AIMessage."
                    )
                    raise ValueError(
                        "No HumanMessage found in the state to create AIMessage."
                    )
                # Check if decision.content is valid JSON
                try:
                    llmresponse, is_json = extract_json_or_answer(decision.content)
                except Exception:
                    is_json = False
                if is_json:
                    # Check if required fields are present
                    output = get_output_parser().parse(llmresponse)
                    self.logger.debug(
                        "Creating AIMessage with decision content (parsed as JSON)."
                    )
                    tooldecisionaimessage = self.create_ai_message(
                        str(output.answer),
                        run_id=runid if runid is not None else "",
                        session_id=sessionid if sessionid is not None else "",
                        has_privatedata=self.has_private_data(output),
                    )
                else:
                    self.logger.debug("Creating AIMessage with decision content.")
                    tooldecisionaimessage = self.create_ai_message(
                        llmresponse,
                        run_id=runid if runid is not None else "",
                        session_id=sessionid if sessionid is not None else "",
                    )
                state["messages"].append(tooldecisionaimessage)
                self.logger.info(
                    "No Tool Call found from TOOL_SELECTOR_NODE, updated state with AIMessage."
                )
                return state

            self.logger.debug(f"Tool Calls Content: {decision.tool_calls}")
            state["messages"].append(
                self.convert_to_ai_message_with_tool_calls(decision)
            )
            self.logger.info(
                "Done Executing TOOL_SELECTOR_NODE, routing to EXECUTE_TOOLS_NODE"
            )
        except Exception as e:
            self.logger.error(f"An error occurred in decide_tools_with_llm: {e}")
            raise
        return state

    def get_mentions(self, query: str) -> List[str]:
        """
        Extract mentions from the user query using regex
        """
        pattern = r"@\w*"
        matches = re.findall(pattern, query)
        tool_name_dict = {"@" + tool.name: [] for tool in self.tools}
        for tool in self.tools:
            actual_name = "@" + tool.name
            tool_name_dict[actual_name].extend([actual_name, actual_name.lower()])
            if actual_name.lower().endswith("agent"):
                nick_names = [actual_name[:-5], actual_name.lower()[:-5]]
                tool_name_dict[actual_name].extend(nick_names)

        for key, value in tool_name_dict.items():
            for match in matches:
                if match in value:
                    query = query.replace(match, key)

        filtered_matches = [
            key
            for key, value in tool_name_dict.items()
            for match in matches
            if match in value
        ]
        return filtered_matches, query

    async def tool_mention_processor(
        self, state: SharedAgentState, config: RunnableConfig
    ) -> SharedAgentState:
        """
        This node passes the tool mentioned in the query to llm
        Node analyses if the the query is related to the tool/agent mentioned. If not, notifies user and asks to rephrase the query or confirm using incorrect mentions.
        """
        try:

            self.logger.info("Executing TOOL_MENTION_PROCESSOR_NODE node")

            if not state.get("messages") or not isinstance(
                state["messages"][-1], HumanMessage
            ):
                self.logger.error(
                    "State in TOOL_MENTION_PROCESSOR_NODE node - No last human message found in state."
                )
                raise ValueError(
                    "InvalidArgument: 'state' must be a SharedAgentState instance with at least one HumanMessage"
                )

            # Check for sessionid from RunnableConfig if available
            sessionid, runid = self.extract_session_and_run_id(config)
            if sessionid is None:
                sessionid = state["messages"][-1].additional_kwargs.get(
                    "sessionid", None
                )

            user_query = state["messages"][-1]
            toolnames, user_query.content = self.get_mentions(user_query.content)

            filtered_tools = [
                tool for tool in self.tools if "@" + tool.name in toolnames
            ]

            if not filtered_tools:
                """
                No tool mentioned in current or in the earlier messages.
                Will be directed to tool decider
                """
                return state

            tool_descriptions_str = "\n".join(
                [
                    f"Name: {tool.name}, Description: {tool.detailed_description}"
                    for tool in filtered_tools
                ]
            )
            if any(f.name in self.filtered_agents_to_exclude for f in filtered_tools):
                self.llm = self.llm.addtools(filtered_tools, tool_choice="any")
            else:
                self.llm = self.llm.addtools(filtered_tools)

            current_datetime = datetime.datetime.now().isoformat()
            if self.checkpointer_type != CheckPointer.NONE:
                if len(state["messages"]) == 1 and isinstance(
                    state["messages"][0], HumanMessage
                ):
                    agentprompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
                        [
                            SystemMessagePromptTemplate.from_template(
                                "You are an intelligent agent. Name of the agent : {name}.\n Description: {description}\n"
                                "You are capable of: {capabilities}\n"
                                "Typical tasks: {typicaltasks}\n"
                                "Available Tools: {tooldescription}\n"
                                "Current DateTime: The current system time is {currentdatetime}\n"
                                "Mentions: {toolnames}\n"
                                "Analyse and understand the scope of the tools/agents provided strictly. Based on the user query, analyse whether the above mentioned tool/agent calls are valid for the execution.\n"
                                "Do not create tool call if query is outside the tool/agent scope\n"
                                "In case of invalid mentions, let user know the exact invalid mention and ask whether he/she wants to enforce the same tool or wants to try with the refined query.\n"
                                "The output should be short, concise and in natural language. Do not give any excess information/suggestions in the response.\n"
                            ),
                            HumanMessagePromptTemplate.from_template("{query}"),
                        ]
                    )
                    prompt_string = agentprompt.invoke(
                        {
                            "tooldescription": tool_descriptions_str,
                            "name": self.name,
                            "description": self.description,
                            "capabilities": self.capabilities,
                            "typicaltasks": self.typical_tasks,
                            "query": user_query.content,
                            "toolnames": toolnames,
                            "currentdatetime": current_datetime,
                        }
                    )

                else:
                    checkpointhistorymessages = []
                    for message in reversed(state["messages"][:-1]):
                        if len(
                            checkpointhistorymessages
                        ) >= self.chathistorycount and isinstance(
                            message, HumanMessage
                        ):
                            checkpointhistorymessages.append(message)
                            break
                        checkpointhistorymessages.append(message)
                    checkpointhistorymessages.reverse()
                    agentprompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
                        [
                            MessagesPlaceholder(
                                variable_name="history",
                                optional=True,
                                n_messages=self.chathistorycount,
                            ),
                            HumanMessagePromptTemplate.from_template("{query}"),
                        ]
                    )
                    if (
                        checkpointhistorymessages is not None
                        and len(checkpointhistorymessages) > 0
                    ):
                        agentprompt = agentprompt.partial(
                            history=checkpointhistorymessages
                        )
                    prompt_string = agentprompt.invoke({"query": user_query.content})
            else:
                agentprompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
                    [
                        SystemMessagePromptTemplate.from_template(
                            "You are an intelligent agent. Name of the agent : {name}.\n Description: {description}\n"
                            "You are capable of: {capabilities}\n"
                            "Typical tasks: {typicaltasks}\n"
                            "Available Tools: {tooldescription}\n"
                            "Current DateTime: The current system time is {currentdatetime}\n"
                            "Mentions: {toolnames}\n"
                            "Analyse and understand the scope of the tools/agents provided strictly. Based on the user query, analyse whether the above mentioned tool/agent calls are valid for the execution.\n"
                            "Do not create tool call if query is outside the tool/agent scope\n"
                            "In case of invalid mentions, let user know the exact invalid mention and ask whether he/she wants to enforce the same tool or wants to try with the refined query.\n"
                            "The output should be short, concise and in natural language. Do not give any excess information/suggestions in the response.\n"
                        ),
                        MessagesPlaceholder(
                            variable_name="history",
                            optional=True,
                            n_messages=self.chathistorycount,
                        ),
                        HumanMessagePromptTemplate.from_template("{query}"),
                    ]
                )
                if self.chat_history_client:
                    historymessages = self.chat_history_client.get_history(
                        user_query.additional_kwargs["sessionid"]
                    )
                    if historymessages is not None and len(historymessages) > 0:
                        if isinstance(historymessages[-1], HumanMessage):
                            historymessages = historymessages[:-1]
                        agentprompt = agentprompt.partial(history=historymessages)

                prompt_string = agentprompt.invoke(
                    {
                        "tooldescription": tool_descriptions_str,
                        "name": self.name,
                        "description": self.description,
                        "capabilities": self.capabilities,
                        "typicaltasks": self.typical_tasks,
                        "query": user_query.content,
                        "toolnames": toolnames,
                        "currentdatetime": current_datetime,
                    }
                )

            decision: BaseMessage = await self.llm.agenerateresponse(
                prompt_string, True
            )
            if not decision.tool_calls:
                self.logger.debug(
                    "No tool calls found in the decision in specific tool call node."
                )
                # Create an AIMessage with the decision content
                last_human_message = self.get_last_human_message(state, config)
                if last_human_message is None:
                    self.logger.error(
                        "No HumanMessage found in the state to create AIMessage."
                    )
                    raise ValueError(
                        "No HumanMessage found in the state to create AIMessage."
                    )
                self.logger.debug("Creating AIMessage with decision content.")
                tooldecisionaimessage = self.create_ai_message(
                    str(decision.content),
                    run_id=runid if runid is not None else "",
                    session_id=sessionid if sessionid is not None else "",
                )
                tooldecisionaimessage.additional_kwargs[
                    "mentioned_tool_call_success"
                ] = False
                state["messages"].append(tooldecisionaimessage)
                self.logger.info("No tool calls found, routing to tool executor")
                return state
            self.logger.debug(
                f"Tool Calls Content in specific tool call: {decision.tool_calls}"
            )
            decision_withtoolcall = self.convert_to_ai_message_with_tool_calls(
                decision
            )
            decision_withtoolcall.additional_kwargs["mentioned_tool_call_success"] = (
                True
            )
            state["messages"].append(decision_withtoolcall)
            self.llm = self.llm.addtools(self.tools)
            self.logger.info("Done executing specific tool decider...")
        except Exception as e:
            self.logger.error(f"An error occurred in call_specific_tool_with_llm: {e}")
            raise
        return state

    def get_last_human_message(
        self, state: SharedAgentState, config: RunnableConfig
    ) -> Optional[HumanMessage]:
        """
        Retrieves the last HumanMessage from the given SharedAgentState.

        Args:
            state (SharedAgentState): The state containing the messages.

        Returns:
            Optional[HumanMessage]: The last HumanMessage if found, otherwise None.
        """
        try:
            for message in reversed(state["messages"]):
                if isinstance(message, HumanMessage):
                    return message
            return None
        except Exception as e:
            self.logger.error(f"An error occurred in get_last_human_message: {e}")
            raise

    def create_ai_message(
        self,
        content: str,
        run_id: str,
        session_id: str,
        references: list[dict] = [],
        has_privatedata: str = "false",
        response_artifact_parts: list[dict] = [],
    ) -> AIMessage:
        """
        Creates and returns an AIMessage with the given content and additional_kwargs
        derived from the last HumanMessage.

        Args:
            content (str): The content of the AIMessage.
            run_id (str): The run identifier.
            session_id (str): The session identifier.
            references (list[dict], optional): List of reference dictionaries. Defaults to [].
            has_privatedata (str, optional): Flag indicating if the message has private data. Defaults to "false".
            response_artifact_parts (list[dict], optional): List of response artifact dictionaries. Defaults to [].

        Returns:
            AIMessage: The created AIMessage.
        """
        try:
            aimessage = AIMessage(content=content)
            aimessage.additional_kwargs["messageid"] = generate_uuid7_id()
            aimessage.additional_kwargs["sessionid"] = session_id
            aimessage.additional_kwargs["run_id"] = run_id
            aimessage.additional_kwargs["references"] = references
            aimessage.additional_kwargs["response_artifact_parts"] = (
                response_artifact_parts
            )
            if str(has_privatedata).lower() == "true":
                aimessage.additional_kwargs["redact"] = "true"

            return aimessage
        except Exception as e:
            self.logger.error(f"An error occurred in create_ai_message: {e}")
            raise
    async def update_semantic_cache_node(self, state) -> SharedAgentState:
        """
        Updates cache if last ai message response is from cache.
        Args:
            state (SharedAgentState): Graph message state
        Returns:
            SharedAgentState: Graph message state.
        """
        self.logger.info("Executing UPDATE_SEMANTIC_CACHE_NODE")
        last_message = state["messages"][-1]

        if type(last_message) != AIMessage:
            return state

        # If the response was from cache or empty, we dont need to cache it.
        if (
            last_message.additional_kwargs.get("response_from_cache")
            or str(last_message.content).strip() == ""
        ):
            return state

        await self.update_semantic_cache(
            state["messages"][0].content, last_message.content
        )
        self.logger.info(
            "UPDATE_SEMANTIC_CACHE_NODE Execution Done, updated cache if applicable."
        )

        return state
    
    
    async def update_semantic_cache(
        self, user_query: str, query_response: str
    ) -> bool:
        """
        Updates cache for given user query and response
        """
        # Check if agent cache is disabled
        if not self.agent_cache:
            self.logger.debug("Semantic Cache: Update Cache - Agent cache not set")
            return None

        # Check with llm can we cache this question
        user_prompt = (
            f"""{{"question": "{user_query}","response":"{query_response}"}}"""
        )
        combined_output = (
            PROMPT_TO_CHECK_CAN_QUERY_BE_CACHED + "\nPROMPT:" + user_prompt
        )
        result = await self.llm.agenerateresponse(combined_output)
        try:
            cache_ques_result = json.loads(result.content)
        except Exception as e:
            self.logger.error(
                f"Semantic Cache: Update Cache - Error in parsing llm response to decide if question can be cached. Error: {e}"
            )
            return False

        # If not YES, do not proceed ahead
        if cache_ques_result.get("decider") != "YES":
            self.logger.debug(
                "Semantic Cache: Update Cache - We can not cache this question as per LLM response."
            )
            return False

        # We can cache this question, update cache
        self.logger.debug(
            f"Semantic Cache: Update Cache - Updating cache entry for the query and response."
        )
        self.agent_cache.set(user_query, query_response)
        self.logger.debug(f"Semantic Cache: Update Cache - Cache entry updated.")
        return True
    
    async def combine_results(
        self, state: SharedAgentState, config: RunnableConfig
    ) -> Union[SharedAgentState, langgraph_types.Send, langgraph_types.Command]:

        self.logger.info("Executing COMBINE_RESULTS_NODE")
        # Check if we have last human message in state
        last_human_message = self.get_last_human_message(state, config)

        if last_human_message is None:
            self.logger.error("No HumanMessage found in the state to combine results.")
            raise ValueError("No HumanMessage found in the state to combine results.")

        # Get the session and run IDs from the config
        session_id, run_id = self.extract_session_and_run_id(config)

        # Check if the last message is an AIMessage and has content
        combined_response = (
            state["messages"][-1]
            if isinstance(state["messages"][-1], AIMessage)
            and state["messages"][-1].content
            else None
        )

        checkpointhistorymessages = []
        for message in reversed(state["messages"]):
            if len(checkpointhistorymessages) >= self.chathistorycount and isinstance(
                message, HumanMessage
            ):
                checkpointhistorymessages.append(message)
                break
            checkpointhistorymessages.append(message)
        checkpointhistorymessages.reverse()
        chat_response_references = get_references_from_messages(
            checkpointhistorymessages
        )
        # Get artifacts and response artifacts from chat response references
        references, response_artifact_parts = get_references_response_artifact_parts(
            chat_response_references
        )

        self.logger.debug(f"Combined Response: {combined_response}")
        if combined_response is not None:
            # Check if decision.content is valid JSON
            try:
                llmresponse, is_json = extract_json_or_answer(combined_response.content)
            except Exception:
                is_json = False
            if is_json:
                parsed_output = get_output_parser().parse(llmresponse)
                aimessage = self.create_ai_message(
                    content=str(parsed_output.answer),
                    run_id=run_id if run_id is not None else "",
                    session_id=session_id if session_id is not None else "",
                    references=references,
                    response_artifact_parts=response_artifact_parts,
                    has_privatedata=self._has_private_data(parsed_output),
                )
            else:
                aimessage = self.create_ai_message(
                    content=str(combined_response.content),
                    run_id=run_id if run_id is not None else "",
                    session_id=session_id if session_id is not None else "",
                    references=references,
                    response_artifact_parts=response_artifact_parts,
                    has_privatedata=combined_response.additional_kwargs.get(
                        "redact", "false"
                    ),
                )
            state["messages"].append(aimessage)
            self.logger.info(
                "COMBINE_RESULTS_NODE Execution Done, updated state with AIMessage."
            )
        else:
            self.logger.error(
                "Combined response is None. Skipping AI message creation."
            )
        return state
    
    
    def update_chat_history_node(
        self, state: SharedAgentState, config: RunnableConfig
    ) -> SharedAgentState:
        """
        Updates chat history with message.

        Args:
            state(SharedAgentState) : current state in the graph

        Returns:
            state : current state in the graph
        """
        if not self.chat_history_client:
            self.logger.debug("Chat history node - Chat history disabled.")
            return state

        # Get last message
        last_human_message = self.get_last_human_message(state, config)

        if not last_human_message:
            self.logger.debug(
                "Chat history node - No last human found in state, routing to end."
            )
            return state

        # Update last human message to chat history
        # session_id = last_human_message.additional_kwargs.get("sessionid")

        # Fetch last AI message
        last_ai_message = (
            state["messages"][-1]
            if isinstance(state["messages"][-1], AIMessage)
            and state["messages"][-1].content
            else None
        )

        # Update last AI message to history
        if last_ai_message:
            self.chat_history_client.add_message(
                last_human_message.additional_kwargs["sessionid"], last_ai_message
            )
            self.logger.info("Chat history node - Updated ai message to chat history.")
        else:
            self.logger.info(
                "Chat history node - No last AI message found to udpate in chat history."
            )

        return state