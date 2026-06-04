import os
import json
from typing import Any, Sequence, Dict, Union
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from genai_core.agent.mcp_client_utils import StructuredToolToBaseToolAdapter
from genai_core.agent.shared_agent_state import SharedAgentState
from models.requestresponse import OrchestratorAgentResponse, \
    get_text_and_non_text_parts, get_text_from_parts, get_non_text_artifacts
from genai_core.logs.agent_logging import DKSAgentLogger
from genai_core.tools.tool_result import ToolResult
from copy import deepcopy
from langchain_core.runnables import RunnableConfig


class CustomToolExecutorNode:
    """
    Custom tool executor node to handle customized execution of tools. 
    """

    def __init__(self, tools: Sequence[BaseTool]):
        self.tools = tools
        self.logger = DKSAgentLogger.get_logger()
        sensitive_tools_env = os.getenv("DKS_SENSITIVE_TOOLS", "") # sentisitve tools are handled
        self.sensitive_tools = [tool.strip() for tool in sensitive_tools_env.split(",")] if sensitive_tools_env else []


    def get_decoded_structured_response(self, agent_response:str)->Any:
        """
        Get decoded strucuted response from agent response.
        """
        try:
            agent_response = json.loads(agent_response)
            self.logger.info('Agent responded with structured response.')
        except Exception:
            # Can not decode using json. It can be string response from agent
            # Do nothing
            self.logger.info('Agent responded with string response.')
        
        return agent_response
    
    def is_agent_result_of_type_model(self, result:Any, agent_response_model: Union[OrchestratorAgentResponse, ToolResult]) -> bool:
        """
        Test if agent result is of givem model type.
        """
        if not isinstance(result, dict):
            return False
        
        if agent_response_model == OrchestratorAgentResponse:
            return "parts" in result
        
        if agent_response_model == ToolResult:
            return "result" in result
        
        # Otherwise
        return False

    
    def parse_agent_result_for_model(self, result:Any, agent_response_model: Union[OrchestratorAgentResponse, ToolResult]) \
        -> Union[OrchestratorAgentResponse, ToolResult, None]:
        """
        Get parsed agent response as per given model.
        """
        if not isinstance(result, dict):
            return None
        
        # Agent tool response
        try:
            parsed_agent_response_model = agent_response_model.model_validate(result)
            self.logger.info(f'Model result is of type: {type(agent_response_model)}')
            return parsed_agent_response_model
        except Exception:
            self.logger.info(f'Model result is not of type: {type(agent_response_model)}')
            return None

    def get_tool_message_from_agent_response(self, 
                                             result: str, 
                                             executing_tool: BaseTool,
                                             tool_executed_with_success: bool, 
                                             tool_call: Dict[str, Any]) \
        -> ToolMessage:
        """
        Get tool message from agent response.
        Returns:
            ToolMessage: The constructed ToolMessage representing the tool execution result.
        """
        # Try to json decode agent response to find out 
        # If it is structured response from agent.
        result = self.get_decoded_structured_response(result)
        
        
        if self.is_agent_result_of_type_model(result, OrchestratorAgentResponse):            
            # Agent tool response
            agent_tool_response_data = self.parse_agent_result_for_model(result, OrchestratorAgentResponse)
            # extract file and data parts from parts                                                                 
            text_parts, non_text_parts = get_text_and_non_text_parts(agent_tool_response_data.parts)                            
            non_text_artifacts = get_non_text_artifacts(non_text_parts)
            artifacts_for_tool_message = agent_tool_response_data.artifacts + non_text_artifacts
            status_str = "success" if tool_executed_with_success else "error"
            tool_message = ToolMessage(content=get_text_from_parts(text_parts), 
                                        artifact=artifacts_for_tool_message, 
                                        tool_call_id=tool_call["id"], status=status_str)
            
        elif self.is_agent_result_of_type_model(result, ToolResult):
            # Tool result response
            tool_result_data = self.parse_agent_result_for_model(result, ToolResult)                            
            tool_message = ToolMessage(content=tool_result_data.result, 
                                        artifact=result, 
                                        tool_call_id = tool_call["id"])            
        else:
            # Otherwise, non json decodable output or string as agent result.
            status_str = "success" if tool_executed_with_success else "error"
            tool_message = ToolMessage(content=str(result), 
                                        artifact=result, 
                                        tool_call_id=tool_call["id"], 
                                        status=status_str)

        return tool_message         

    async def execute_tool(self, tooltoexecute: StructuredToolToBaseToolAdapter, tool_call:ToolCall) -> tuple:
        istoolsuccess = False                        
        try:
            result = await tooltoexecute._arun(tool_input=tool_call["args"])
            istoolsuccess = True
        except Exception as e:
            self.logger.exception(e)
            self.logger.error(f"Error executing tool {tooltoexecute.name}: {e}", exc_info=True)
            result = f"Error executing tool {tooltoexecute.name}: {e}"
        return result, istoolsuccess

    async def __call__(self, state: SharedAgentState, config: RunnableConfig) -> SharedAgentState:
        """
        Method Summary:
        Executes a sequence of tool calls based on the last AIMessage in the provided state, 
        updating the state with the results of the tool executions.
        Args:
            state (MessagesState): The current state containing a list of messages, 
                                   including potential tool calls to process.
        Returns:
            MessagesState: The updated state with results of executed tool calls appended 
                           to the messages list.
        Behavior:
        - Retrieves the "messages" list from the provided state.
        - Checks if the last message is an AIMessage and contains tool calls.
        - Iterates through each tool call, finds the corresponding tool, and executes it.
        - Passes shared tool inputs between tool calls using a shared dictionary.
        - Appends the results of each tool execution as a ToolMessage to the state's messages.
        """
        
        messages = state.get("messages", [])
        if messages:
            last_message = messages[-1]
            if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls") and last_message.tool_calls:
                # Find the last HumanMessage before the last AIMessage
                last_human_message = None
                for message in reversed(state["messages"]):
                    if isinstance(message, HumanMessage):
                        last_human_message = message
                        break
                for tool_call in last_message.tool_calls:
                    # Process each tool call as needed
                    tool_call_execute: ToolCall = deepcopy(tool_call)
                    tooltoexecute = tool_call["name"]
                    tooltoexecute = next((tool for tool in self.tools if tool.name == tooltoexecute), None)
                    if tooltoexecute:
                        tool_call_execute["args"]["auditcontext"] = {}
                        if last_human_message:
                            tool_call_execute["args"]["auditcontext"] = self.extract_auditcontext_from_config(tooltoexecute.name, config)
                        self.logger.debug(f"Executing tool {tooltoexecute.name} with {tool_call_execute['args']}")

                        # Execute tool with retry logic
                        result = None
                        istoolsuccess = False
                        max_retries = 3
                        for attempt in range(max_retries):
                            result, istoolsuccess = await self.execute_tool(tooltoexecute, tool_call_execute)
                            if istoolsuccess:
                                break
                            self.logger.warning(f"Tool execution attempt {attempt + 1}/{max_retries} failed for {tooltoexecute.name}")


                        # Get tool message from agent result
                        tool_message = self.get_tool_message_from_agent_response(
                            result=result, 
                            executing_tool=tooltoexecute, 
                            tool_executed_with_success=istoolsuccess, 
                            tool_call=tool_call_execute
                        )

                        # Update graph state for tool message                                            
                        state["messages"].append(tool_message)
                    else:
                        self.logger.error(f"Cannot find tool to Execute tool {tool_call_execute['name']}")
        return state
    
    def extract_auditcontext_from_config(self, toolname: str, config: RunnableConfig) -> dict:
        """
        Extracts the 'auditcontext' from the provided RunnableConfig if present.

        Args:
            config (RunnableConfig): The configuration object potentially containing 'auditcontext'.

        Returns:
            dict: The extracted 'auditcontext' dictionary, or an empty dict if not found.
        """
        auditcontext = {}
        if config is not None:
            auditcontext = deepcopy(config.get("configurable", {}).get("auditcontext", {}))
            # Remove 'additional_args' in 'auditcontext' if present
            if 'additional_args' in auditcontext:
                auditcontext.pop("additional_args", None)
        if toolname in self.sensitive_tools:
            auditcontext["redact"] = "true"
        return auditcontext