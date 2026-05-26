from typing import Any

class AgentException(Exception):
    """Custom exception for agent-related errors."""
    def __init__(self, userquery=None, *args: tuple[Any, ...]):
        super().__init__(*args)
        self.userquery = userquery
        self.args = args

    def __str__(self):
        all_args = ', '.join(map(str, self.args))
        if self.userquery:
            return f"{self.__class__.__name__}: {all_args} (User Query: {self.userquery})"
        return f"{self.__class__.__name__}: {all_args}"