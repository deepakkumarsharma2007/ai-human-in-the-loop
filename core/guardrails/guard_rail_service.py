from abc import ABC, abstractmethod
from core.guardrails.guard_rail_models import GuardRailsResponse

class GuardRailService(ABC):
    @abstractmethod
    def analyzeprompt(self, input_text: str) -> GuardRailsResponse:
        pass

    @abstractmethod
    def analyzeoutput(self, prompt: str, output: str) -> GuardRailsResponse:
        pass

    @abstractmethod
    def scanprompt(self, input_text: str) -> GuardRailsResponse:
        pass

    @abstractmethod
    def scanoutput(self, prompt: str, output: str) -> GuardRailsResponse:
        pass 