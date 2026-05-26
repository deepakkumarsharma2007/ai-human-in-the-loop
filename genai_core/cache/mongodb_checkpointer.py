from langgraph.checkpoint.mongodb import MongoDBSaver
import os
from redis import Redis

class MongoDBCheckPointer:
    def __init__(self):
        self.initialized = False
        self.mongodb_checkpointer = None
    
    def initialize(self) -> None:
        if not self.initialized:
            self.mongodb_checkpointer = self.set_mongodb_checkpointer()
            self.initialized = True

    def set_mongodb_checkpointer(self) -> MongoDBSaver:
        """
        Creates and returns a MongoDBSaver instance.

            Returns:
            MongoDBSaver: An instance of MongoDBSaver.
        """

        if self.initialized:
            return self.mongodb_checkpointer

        try:


            self.mongodb_checkpointer = MongoDBSaver.from_conn_string(
                 os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
                 ,db_name=os.getenv("MONGODB_CHECKPOINT_DB_NAME", "Checkpoints")
                 ,collection_name=os.getenv("MONGODB_CHECKPOINT_COLLECTION_NAME", "checkpoints")
                 , writes_collection_name=os.getenv("MONGODB_CHECKPOINT_WRITES_COLLECTION_NAME", "checkpoints_writes")
                                  )

            self.initialized = True
        
        except Exception as e:
            self.initialized = False
            raise RuntimeError(f"Failed to create MongoDBSaver instance for checkpointer: {e}") from e
            
