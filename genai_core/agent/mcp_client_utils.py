"""
Module for client side utility functions for communicating with MCP server.
It provides MCP adapter tool to convert mcp tool to core agent compatible tool.
"""
from typing import List, Any
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.sessions import StreamableHttpConnection
from langchain_core.tools.structured import StructuredTool
from langchain_core.tools import BaseTool


class StructuredToolToBaseToolAdapter(BaseTool):
    """
    Adapter tool to convert langchain structured tool to BaseTool
    """
    # Name of tool
    name: str = ""
    # Description of tool
    description: str = ""
    # Structured tool instance
    structured_tool: StructuredTool = None

    def __init__(self, structured_tool:StructuredTool, *args:tuple, **kwargs:dict) -> None:
        """
        Initialize tool details from structured tool.

        Args:
            *args(tuple): List of positional arguments passed for initialization.
            **kwargs(dict): Dictonary of keyword arguments passed for initialization.
        
        Returns:
            None
        """
        super().__init__(*args, **kwargs)
        self.structured_tool = structured_tool
        self.name = structured_tool.name
        self.description = structured_tool.description
        self.args_schema = structured_tool.args_schema
        self.response_format = structured_tool.response_format
        self.metadata = structured_tool.metadata

    @property
    def detailed_description(self) -> str:
        """
        Get detailed description of tool.

        Args:

        Return:
            str: Description of tool, set from structured tool.
        """        
        return self.description

    def _run(self) -> None:
        """
        Raise not implemented error for _run as it is not expected to be called.

        Args:

        Returns:
            None
        
        Raises:
            NotImplementedError: Raises as method not implemented.
        """
        raise NotImplementedError

    async def _arun(self, *args, **kwargs) -> str:
        """
        Async execute the structured tool coroutine method with identified arguments.

        Args:
            *args(tuple): List of positional arguments passed for tool calling.
            **kwargs(dict): Dictonary of keyword arguments passed for tool calling.

        Returns:
            str: Tool call result.
        """
        # @TODO: Test for *args case, where args has the tool call parameters.
        call_tool_result = await self.structured_tool.ainvoke(input=kwargs["tool_input"])
        return call_tool_result


def langchain_structured_tool_to_base_tool(structured_tools: List[StructuredTool]) -> List[StructuredToolToBaseToolAdapter]:
    """
    Convert langchain structured tool to tool extended from base tool.

    Args:
        structured_tools (List[StructuredTool]): List of langchain structured tool objects.
    
    Returns:
       List[StructuredToolToBaseToolAdapter]: List of StructuredToolToBaseToolAdapter objects which are extended from base tool.
    """
    tools = []
    # For each structured tool create tool extended form BaseTool
    for structured_tool in structured_tools:
        tool = StructuredToolToBaseToolAdapter(structured_tool)          
        tools.append(tool)        
    return tools


async def get_mcp_tools_with_server(connection: StreamableHttpConnection) -> List[StructuredToolToBaseToolAdapter]:
    """
    Get MCP tools from MCP server with server_url and transport.

    Args:
        server_url (str): URL of MCP server.
        taransport (str): Transport method for MCP server communication. Defaults to streamable_http. @TODO: Implement other transport methods.
    
    Returns:
        List[StructuredToolToBaseToolAdapter]: List of StructuredToolToBaseToolAdapter objects which are extended from base tool so that they can 
                                                  be executed with CoreReactAgent.

    """
    if connection is None:
        raise ValueError("Connection must not be None.")

    # Load MCP tools from langchain
    mcp_tools = await load_mcp_tools(session=None, connection=connection)

    # Convert langchain structured tool to dks tool for tool calling on core agent.
    dks_tools = langchain_structured_tool_to_base_tool(mcp_tools)
    return dks_tools
