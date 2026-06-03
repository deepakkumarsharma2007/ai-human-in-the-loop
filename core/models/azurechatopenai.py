from typing import Optional, List, Any
from dotenv import load_dotenv
import os

from langchain_core.messages import AIMessage
from langchain_openai import AzureChatOpenAI
from langchain_core.prompt_values import PromptValue
from langchain_core.messages import BaseMessage
from core.models.model_base import ModelBase
from langchain_core.tools import BaseTool
from core.guardrails.llm_guard_rail_service import LLMGuardRailService, GuardRailService
from genai_core.logs.agent_logging import DKSAgentLogger
from core.models.model_exception import LLMModelException


class AzureChatOpenAiModel(ModelBase):
    def __init__(self, 
        model_name: Optional[str] = None, 
        deployment_name: Optional[str] = None, 
        api_key: Optional[str] = None, 
        api_base: Optional[str] = None, 
        api_version: Optional[str]  = None,
        max_retries: Optional[int] = None):
        load_dotenv()  # Load variables from .env file

        self.model_name = model_name if model_name is not None else os.getenv("DKS_AZURE_OPENAI_MODEL_NAME")
        self.deployment_name = deployment_name if deployment_name is not None else os.getenv("DKS_AZURE_OPENAI_DEPLOYMENT_NAME")
        self.api_key = api_key if api_key is not None else os.getenv("DKS_AZURE_OPENAI_API_KEY")
        self.api_endpoint = api_base if api_base is not None else os.getenv("DKS_AZURE_OPENAI_ENDPOINT")
        self.api_version = api_version if api_version is not None else os.getenv("DKS_AZURE_OPENAI_API_VERSION")
        
        # Validate max_retries param
        max_retries = max_retries if max_retries is not None else int(os.environ.get("DKS_AZURE_OPENAI_API_MAX_RETRIES", 3))
        if not isinstance(max_retries, int):
            raise ValueError("max_retries must be a non-negative, non-zero integer.")        
        if max_retries <= 0:
                raise ValueError("max_retries must be a non-negative, non-zero integer.")                
        self.max_retries = max_retries
        
        if not all([self.model_name, self.deployment_name, self.api_key, self.api_endpoint, self.api_version]):
            raise ValueError("Missing required Azure OpenAI configuration. Please provide all necessary parameters or set them in the .env file.")

        self.isguardrailenabled = os.getenv("GUARDRAILENABLED", "false").lower() == "true"
        self.llmguardservice: GuardRailService = LLMGuardRailService()
        self.logger = DKSAgentLogger.get_logger()

        self.client = None
        self.toolenabledclient = None
        super().__init__(self.model_name)

    def load_model(self) -> AzureChatOpenAI:
        self.client = AzureChatOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.api_endpoint,
            azure_deployment=self.deployment_name,
            verbose=True,
            max_retries=self.max_retries
        )
        return self.client

    def generate(self, prompt: str, sysprompt: str, max_length: int = 1000) -> str:
        raise NotImplementedError("The generate method is not implemented.")  
    
    def guardrail_check_on_prompt(self, prompt: Any, failonguardrailscanning:bool = True, ignoreguardrailscan: bool = False) -> None:
        """
        Check guardrails on prompt if enabled.

        Args:
            prompt (Any): Input prompt to chec.
            failonguardrailscanning (Bool) : To raise exception if gaurdrail fails.
            ignoreguardrailscan (Bool) : To ignore guardrail scan or not.
        
        Returns:
            None
        
        Raises:
            LLMModelException : when guardrail validation fails.
            
        """
            
        if self.isguardrailenabled and not ignoreguardrailscan:
            if isinstance(prompt, PromptValue):
                prompt_string = prompt
                messagestoscan = prompt_string.to_messages()
                last_message = messagestoscan[-1] if messagestoscan else None
                if last_message and isinstance(last_message.content, str):
                    guardresponse = self.llmguardservice.analyzeprompt(last_message.content)
                    if not guardresponse.isvalid and failonguardrailscanning:
                        invalid_items = [scanner_result.type for scanner_result in guardresponse.scannerresults if scanner_result.value == 1]
                        self.logger.error(f"Guardrail validation failed for scans {invalid_items}.")
                        raise LLMModelException(f"Invalid Prompt. Does not comply to the organization policies. Guardrail validation failed for scans {invalid_items}.")
    
    def llmguard_process_response(self, response: BaseMessage, prompt: Any, ignoreguardrailscan: bool = False) -> BaseMessage:
        """
        Process response for guardrails.

        Args:
            response (BaseMessage): Response from llm invoke.
            prompt (Any): Input prompt
            ignoreguardrailscan (Bool): To disable guardrail scan or not.
        
        Returns:
            BaseMessage : Guardrail procssed response. 

        """
        # If guardrails enabled sanitize response
        if self.isguardrailenabled and not ignoreguardrailscan:
            self.logger.info("Guardrail is enabled. Validating llm output.")
            if response and isinstance(response.content, str) and response.content.strip():
                guardresponse = self.llmguardservice.analyzeoutput(str(prompt), response.content)
                sanitized_message = AIMessage(content=str(guardresponse.sanitizedstring), **response.additional_kwargs)
                return sanitized_message
        
        # Otherwise return unprocessed response.
        return response
    
    def generateresponse(self, prompt: Any, toolenabled: bool = False, failonguardrailscanning:bool = True,
                         ignoreguardrailscan: bool = False) -> BaseMessage:
        """
        Generate response for given prompt. 

        Args:
            promt (Any): Input prompt to generate response for.
            toolenabled (bool): Tool enabled client to generate response.
            failonguardrailscanning (bool): To disable/enable raising exception if guardrails fails.
            ignoreguardrailscan (bool): To disable/enable guardrails scan.
        
        Returns:
            BaseMessage: BaseMessage with generated llm response.
        """
        
        # Check guardrails on prompt
        self.guardrail_check_on_prompt(prompt, failonguardrailscanning, ignoreguardrailscan)

        # Load model if not loaded
        if not self.client:
            self.load_model()

        # Generate response as per toolenabled
        if (toolenabled):
            response = self.toolenabledclient.invoke(prompt)
        else:
            response = self.client.invoke(prompt)
        
        # Return processed response using llm guardrails
        return self.llmguard_process_response(response, prompt, ignoreguardrailscan)

    async def agenerateresponse(self, prompt: Any, toolenabled: bool = False, failonguardrailscanning:bool = True,
                         ignoreguardrailscan: bool = False) -> BaseMessage:
        """
        Aysnc - Generate response for given prompt. 

        Args:
            promt (Any): Input prompt to generate response for.
            toolenabled (bool): Tool enabled client to generate response.
            failonguardrailscanning (bool): To disable/enable raising exception if guardrails fails.
            ignoreguardrailscan (bool): To disable/enable guardrails scan.
        
        Returns:
            BaseMessage: BaseMessage with generated llm response.
        """
        
        # Check guardrails on prompt
        #    self.guardrail_check_on_prompt(prompt, failonguardrailscanning, ignoreguardrailscan)

        # Load model if not loaded
        if not self.client:
            self.load_model()

        # Generate response as per toolenabled
        if (toolenabled):
            response = await self.toolenabledclient.ainvoke(prompt)
        else:
            response = await self.client.ainvoke(prompt)
        
        return response
        # Return processed response using llm guardrails 
        # return self.llmguard_process_response(response, prompt, ignoreguardrailscan)


    def addtools(self, tools: List[BaseTool]) -> ModelBase:
        if not self.client:
            self.load_model()
        self.toolenabledclient = self.client.bind_tools(tools)
        return self
    
    def createembedding(self, query: str) -> list:
        raise NotImplementedError("The generate method is not implemented yet.")    

    def __call__(self, prompt: str, sysprompt: str, max_length: int = 1000) -> str:
        return self.generate(prompt, sysprompt, max_length)