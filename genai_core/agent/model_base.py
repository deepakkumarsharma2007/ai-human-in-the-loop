from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage

from abc import ABC, abstractmethod
from typing import List


class ModelBase(ABC):

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = self.load_model()


    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def generate(self):
        pass

    @abstractmethod
    def generateresponse(self, prompt, toolenabled: bool = False, 
                         failonguardrailscanning:bool = True,
                         ignoreguardrailscan: bool = False) -> BaseMessage:
        pass

    @abstractmethod
    def createembedding(self, query) -> list[float]:
        pass

    @abstractmethod
    def addtools(self, tools: List[BaseTool]) -> 'ModelBase':
        pass

    
    def __call__(self, prompt, sysprompt, max_length=1000):
        return self.generate(prompt, sysprompt, max_length)
    
    def __getembedding__(self, query):
        return self.createembedding(query)