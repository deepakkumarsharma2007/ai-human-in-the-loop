"""
Module defines aegis agent cache class RedisSemCache.
"""
import os
import uuid
from dotenv import load_dotenv

from genai_core.logs.agent_logging import DKSAgentLogger
load_dotenv()


from abc import ABC, abstractmethod
from langchain_core.outputs.generation import Generation
from langchain_community import cache


logger = DKSAgentLogger.get_logger()


class BaseAgentCache(ABC):
    """
    Abstract base class for Agent Cache 

    Inheriting classes must implement the methods.

    """

    @abstractmethod
    def get(self, key) -> str | None:
        """
        To implement getting cache entry for given key.

        Args:
            key (str): Cache entry key
        
        Returns:
            (str | None): Returns string when cache entry found else None

        """
        pass

    @abstractmethod
    def set(self, key, value) -> None:
        """
        To implment setting cache entry for given key and value.
        @TODO: This should return bool True/False as per update status of cache entry, current implmentation does not support this.

        Args:
            key (str): Cache entry key
            value (str): Cache entry value
        
        Returns:
            None
        """
        pass

    @abstractmethod
    def delete(self, key) -> bool:
        """
        To implment deleting cache entry for given key.
        
        Args:
            key (str): Cache entry key

        Returns:
            (bool):True when cache entry deleted, False otherwise.

        """
        pass


class AgentRedisSemCache(BaseAgentCache):
    """
    Agent Redis Semantic Cache Implmentation class

    It initlizes with embeddings and redis endpoint, uses redis as store. 
    
    """
    def __init__(self, 
                 embeddings, 
                 llm_string='',                  
                 score_threshold=None, 
                 redis_endpoint=None, 
                 redis_user=None,                  
                 redis_pwd=None, 
                 ttl=None,
                 ) -> None:
        """
        Initializes Redis Semantic Cache.

        Args:
            embeddings (AzureOpenAIEmbeddings): Embeddings object to be used to create embeddings for similarity search.
            llm_string (Dict|str): User defined dictionary or string to isolate/group cache entries. This should be used to capture llm invocation parameters model name, temperature. 
            score_threshold (float): float value representing vector distance for similarity match.
            redis_endpoint (str): Redis instance endpoint with port e.g. example.redis.net:10000
            redis_user (str): Redis instance username            
            redis_pwd (str): Redis isntance password
            ttl (int): Cache expiry seconds since cache entry creation. 

        Returns:
            None

        """        
        # Redis endpoint: If not given pulls from env var SEM_CACHE_REDIS_ENDPOINT
        redis_endpoint = redis_endpoint if redis_endpoint else os.environ.get('SEM_CACHE_REDIS_ENDPOINT')

        # Redis user: If not given pulls from env var SEM_CACHE_REDIS_USER falls back to 'default'
        redis_user =  redis_user if redis_user else os.environ.get('SEM_CACHE_REDIS_USER', 'default')

        # Redis password : If not given pulls from env var  SEM_CACHE_REDIS_PASSWORD
        redis_pwd = redis_pwd if redis_pwd else os.environ.get('SEM_CACHE_REDIS_PASSWORD')
        redis_url = f"rediss://{redis_user}:{redis_pwd}@{redis_endpoint}"

        # Score / Distance threshold, defaults to 0.2 if not given
        score_threshold = score_threshold if score_threshold else round(float(os.environ.get('SEM_CACHE_SCORE_THRESHOLD', 0.2)), 2)

        # TTL - time to live until expiry, defaults to 24 hours if not given.
        ttl = ttl if ttl else int(os.environ.get('SEM_CACHE_TTL', 24*60*60))

        # Initialize RedisSemanticCache instance for querying cache
        redis_sem_cache = cache.RedisSemanticCache(
            redis_url = redis_url, 
            embeddings=embeddings, 
            distance_threshold=score_threshold, 
            ttl=ttl)            
        self.redis_sem_cache = redis_sem_cache

        # If LLM string not set, set to default one
        if not llm_string:
            llm_string = {
                'redis_endpoint': redis_endpoint,
                'score_threshold': score_threshold,
                'llm_string_key': os.environ.get('SEM_CACHE_LLM_STR_KEY', '')
            }

        self.llm_string = llm_string

    def get(self, key) -> str | None:
        """
        Gets the entry from cache for given key.

        Args:
            key (str): Cache key 
        
        Returns:
            (str | None): Response from cache or None incase of not found or error.

        Raises:
            RuntimeError: When semantic lookup call fails.
                        
        """
        # For empty strings or None values
        if not key:
            return None

        # Search cached entry into redis and return the response text
        try:
            cached = self.redis_sem_cache.lookup(prompt=key, llm_string=self.llm_string)        
            if not cached or len(cached) == 0:
                return None
                        
            cached_entry = cached[0]
            if not hasattr(cached_entry, 'text'):
                return None
            
            return cached_entry.text
        except Exception as e:
            error_msg = f"Error occurred while getting cache entry. Error: {e}"
            logger.error(error_msg)
            logger.exception(e)
            raise RuntimeError(error_msg)

    async def aget(self, key) -> str | None:
        """
        Gets the entry from cache for given key with async.

        Args:
            key (str): Cache key 
        
        Returns:
            (str | None): Response from cache or None incase of not found or error.

        Raises:
            RuntimeError: When semantic lookup call fails.
                        
        """
        # For empty strings or None values
        if not key:
            return None

        # Search cached entry into redis and return the response text
        try:
            cached = await self.redis_sem_cache.alookup(prompt=key, llm_string=self.llm_string)
            print("cached :", cached)        
            if not cached or len(cached) == 0:
                return None
                        
            cached_entry = cached[0]
            if not hasattr(cached_entry, 'text'):
                return None
            
            return cached_entry.text
        except Exception as e:
            error_msg = f"Error occurred while getting cache entry. Error: {e}"
            logger.error(error_msg)
            logger.exception(e)
            raise RuntimeError(error_msg)
        
    def set(self, key, value) -> None:
        """
        Sets the entry to cache with given key and value.

        Args:
            key (str): string to use as key in cache store.
            value (str): string to be later retrieved as response from cache. 
        
        Returns:
            None
        
        Raises:
            RuntimeError: When semantic update call fails.

        """
        try:        
            generation_info = Generation(generation_info={
                'custom_cache_id': f'custom_cache_id_{uuid.uuid4()}'
                }, text=value, type='Generation')                
            self.redis_sem_cache.update(prompt=key, 
                                        llm_string=self.llm_string, 
                                        return_val=[generation_info])
        except Exception as e:
            error_msg = f"Error occurred while updating cache entry. Error: {e}"
            logger.error(error_msg)
            logger.exception(e)
            raise RuntimeError(error_msg)


    async def aset(self, key, value) -> None:
        """
        Sets the entry to cache with given key and value.

        Args:
            key (str): string to use as key in cache store.
            value (str): string to be later retrieved as response from cache. 
        
        Returns:
            None
        
        Raises:
            RuntimeError: When semantic update call fails.

        """
        try:        
            generation_info = Generation(generation_info={
                'custom_cache_id': f'custom_cache_id_{uuid.uuid4()}'
                }, text=value, type='Generation')                
            await self.redis_sem_cache.aupdate(prompt=key, 
                                        llm_string=self.llm_string, 
                                        return_val=[generation_info])
        except Exception as e:
            error_msg = f"Error occurred while updating cache entry. Error: {e}"
            logger.error(error_msg)
            logger.exception(e)
            raise RuntimeError(error_msg)
                
    def delete(self, key) -> None:
        """
        Deletes entry from cache for given key.

        Args:
            key (str): string representing key to delete the corresponding entry.

        Raises:
            NotImplementedError : As not implemented in current scope.

        """
        raise NotImplementedError('Not implemented yet')



class AgentCacheManager:
    """
    To wrap/interface different implmentations of agent cache inherited from BaseAgentCache. 
    exploses strict methods and signatures.

    """
    def __init__(self, cache: BaseAgentCache):
        """
        Sets cache implementation to use.
        """
        self.cache = cache

    def get(self, key):
        """
        Wraps "get" method of cache implementation 
        """
        return self.cache.get(key)
    
    async def aget(self, key):
        """
        Wraps "get" method of cache implementation 
        """
        return await self.cache.aget(key)

    def set(self, key, value):
        """
        Wraps "set" method of cache implementation 
        """
        self.cache.set(key, value)

    async def aset(self, key, value):
        """
        Wraps "set" method of cache implementation 
        """
        await self.cache.aset(key, value)

    def delete(self, key):
        """
        Wraps "delete" method of cache implementation 
        """
        self.cache.delete(key)