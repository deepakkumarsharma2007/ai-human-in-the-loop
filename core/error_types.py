"""
Module to define custom errors
"""


class AgentOrchestratorBaseError(Exception):
    """
    Base class for known errors.
    """
    def __init__(self, message:str, code:str, details:dict={}, status_code:int=500):
        super().__init__(message)
        self.status_code = status_code
        self.error = message
        self.code = code
        self.details = details


class OrchestratorConfigError(AgentOrchestratorBaseError):    
    """
    Orchestrator Configuration Error to be raised in cases where Orchestrator is having improper configuration.
    """
    def __init__(self, message, details = {}):        
        super().__init__(message, code="ServerConfigError", details=details, status_code=500)


class OrchestratorGraphError(AgentOrchestratorBaseError):    
    """
    Orchestrator Graph Error to be raised in cases where Orchestrator is having issues in processing the graph.
    """
    def __init__(self, message, details = {}):        
        super().__init__(message, code="OrchestratorGraphError", details=details, status_code=500)
