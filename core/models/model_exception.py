from typing import Any

class LLMModelException(Exception):
    """Custom exception for agent-related errors."""
    def __init__(self, prompt=None, *args: tuple[Any, ...]):
        super().__init__(*args)
        self.prompt = prompt
    def __str__(self):
        all_args = ', '.join(map(str, self.args))
        if self.prompt:
            return f"{self.__class__.__name__}: {all_args} (Prompt: {self.prompt})"
        return f"{self.__class__.__name__}: {all_args}"