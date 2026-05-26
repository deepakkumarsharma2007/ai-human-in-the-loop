from langgraph.graph import MessagesState
from typing import Annotated
import operator

class SharedAgentState(MessagesState):
    shareddata: Annotated[list, operator.add]