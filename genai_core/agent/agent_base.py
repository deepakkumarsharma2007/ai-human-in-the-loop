from typing import Any, Dict, Optional
from abc import ABC, abstractmethod

class AgentBase(ABC):
    """
    Agent base class.
    Attributes:
        name (str): The name of the agent.
        description (str): A brief description of the agent.
        capabilities (str): The capabilities of the agent.
        typical_tasks (str): The typical tasks the agent can perform.
        llm: The language model used by the agent.
    Methods:
        construct_workflow() -> None:
            Construct the Graph workflow for the agent.
        async execute(user_query: str) -> Dict[str, Any]:
            Execute the query and return the result.
    """
    
    def __init__(self, name: str, description: str, capabilities: str,
                 typical_tasks: str, llm: Optional[Any] = None):
        self.name = name
        self.description = description
        self.capabilities = capabilities
        self.typical_tasks = typical_tasks
        self.llm = llm
        #self.conditions = conditions or []

    @abstractmethod
    def construct_workflow(self) -> None:
        """
        Construct the workflow for the agent.
        This method should be implemented by subclasses.
        """
        pass

    @abstractmethod
    async def async_execute(self, user_query: str, sessionid: Optional[str],
                       auditContext: Optional[dict]) -> Dict[str, Any]:
        """
        Execute the agent's logic based on the user query.
        This method should be implemented by subclasses.
        Args:
            state (Dict[str, Any]): The current state to be processed.
            user_query (str): The user query to guide the agent's execution.
        Returns:
            Dict[str, Any]: The updated state after processing.
        """
        pass