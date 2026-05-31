import os


from typing import Optional, Dict, Any
from langsmith import Client
from langchain_core.prompts import PromptTemplate
from genai_core.logs.agent_logging import DKSAgentLogger


logger = DKSAgentLogger.get_logger()


class PromptManager:
    """
    A class to manage prompts using LangSmith as the source of truth.
    Retrieves and manages prompts stored in LangSmith.
    """
    # Class attribute client to avoid reinstantiation 
    # LangSmith client
    client: Client = None
    # Class attribute prompt_cache to avoid reinstantiation
    # For cache fetched prompts
    prompt_cache : Dict[str, PromptTemplate] = {}
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the PromptManager with LangSmith credentials.
        
        Args:
            api_key (Optional[str]): LangSmith API key. If not provided, 
                                   will look for LANGSMITH_API_KEY in environment variables.
        """
        self.api_key = api_key or os.getenv("LANGSMITH_API_KEY")
        if not self.api_key:
            logger.warning('LangSmith API key not provided')
            raise ValueError("LangSmith API key must be provided or set as LANGSMITH_API_KEY environment variable")
        
        if not self.prompt_cache:    
            self.prompt_cache: Dict[str, PromptTemplate] = {}
        
    def get_prompt(self, prompt_name: str, refresh_cache: bool = False) -> PromptTemplate:
        """
        Retrieve a prompt from LangSmith by name.
        
        Args:
            prompt_name (str): Name of the prompt to retrieve
            refresh_cache (bool): Whether to force refresh the cache for this prompt
            
        Returns:
            PromptTemplate: The retrieved prompt template
            
        Raises:
            ValueError: If prompt is not found in LangSmith
        """
        if not refresh_cache and prompt_name in self.prompt_cache:
            return self.prompt_cache[prompt_name]
        
        try:
            # Fetch prompt from LangSmith
            prompt_data = self.client.pull_prompt(prompt_name)
            
            # Create PromptTemplate from the retrieved data
            template = prompt_data.template
            input_variables = [var.strip('{}') for var in prompt_data.input_variables]
            
            prompt_template = PromptTemplate(
                template=template,
                input_variables=input_variables
            )
            
            # Cache the prompt
            self.prompt_cache[prompt_name] = prompt_template
            return prompt_template
            
        except Exception as e:
            logger.error(e)
            raise ValueError(f"Failed to retrieve prompt '{prompt_name}' from LangSmith: {str(e)}")
    
    def get_formatted_prompt(self, prompt_name: str, **kwargs: Any) -> str:
        """
        Retrieve and format a prompt with provided variables.
        
        Args:
            prompt_name (str): Name of the prompt to retrieve and format
            **kwargs: Variables to format the prompt with
            
        Returns:
            str: The formatted prompt
        """
        prompt_template = self.get_prompt(prompt_name)
        return prompt_template.format(**kwargs)
    
    def refresh_cache(self) -> None:
        """Clear the entire prompt cache."""
        self.prompt_cache.clear()
    
    def list_available_prompts(self) -> list[str]:
        """
        List all available prompts in LangSmith.
        
        Returns:
            list[str]: List of prompt names
        """
        try:
            # prompts = self.client.list_prompts()
            return []
        except Exception as e:
            logger.error(e)
            raise ValueError(f"Failed to list prompts from LangSmith: {str(e)}")