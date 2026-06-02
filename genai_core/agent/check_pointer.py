from enum import Enum

class CheckPointer(Enum):
    REDIS = "Redis"
    MONGODB = "MongoDB"
    NONE = "None"