from langgraph.checkpoint.redis import RedisSaver
import os
from redis import Redis

class RedisCheckPointer:
    def __init__(self):
        self.initialized = False
        self.redis_checkpointer = None
    
    def initialize(self) -> None:
        if not self.initialized:
            self.redis_checkpointer = self.get_redis_checkpointer()
            self.initialized = True

    def set_redis_checkpointer(self) -> RedisSaver:
        """
        Creates and returns a RedisSaver instance.

            Returns:
            RedisSaver: An instance of RedisSaver.
        """

        if self.initialized:
            return self.redis_checkpointer

        try:
            REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
            REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
            REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
            REDIS_SSL = os.getenv("REDIS_SSL", "false").lower() == "true"
            # REDIS_URI = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"
            
            # RedisSaver.from_conn_string(REDIS_URI):
            client = Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                password=REDIS_PASSWORD,
                ssl=REDIS_SSL,
                ssl_cert_reqs="required" if REDIS_SSL else None,
                decode_responses=False
            )
            self.redis_checkpointer = RedisSaver(redis_client=client)
            self.redis_checkpointer.setup()
            self.initialized = True
        
        except Exception as e:
            self.initialized = False
            if(os.getenv("REDIS_DO_NOT_FAIL_IF_REDIS_UNAVAILABLE", "false").lower() == "true"):
                print(f"Warning: Failed to create RedisSaver instance for checkpointer: {e}. Continuing without Redis checkpointer.")
                return None
            raise RuntimeError(f"Failed to create RedisSaver instance: for checkpointer {e}") from e
            
