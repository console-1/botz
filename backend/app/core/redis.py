import redis
from typing import Optional
from .config import settings


# Redis client for session management
redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)


class SessionManager:
    def __init__(self):
        self.redis = redis_client
        self.default_ttl = 3600  # 1 hour
    
    async def set_session(self, session_id: str, data: dict, ttl: Optional[int] = None) -> bool:
        """Store session data with optional TTL"""
        try:
            ttl = ttl or self.default_ttl
            return self.redis.setex(f"session:{session_id}", ttl, str(data))
        except Exception as e:
            print(f"Error setting session: {e}")
            return False
    
    async def get_session(self, session_id: str) -> Optional[dict]:
        """Retrieve session data"""
        try:
            data = self.redis.get(f"session:{session_id}")
            if data:
                return eval(data)  # Note: In production, use proper JSON serialization
            return None
        except Exception as e:
            print(f"Error getting session: {e}")
            return None
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete session data"""
        try:
            return bool(self.redis.delete(f"session:{session_id}"))
        except Exception as e:
            print(f"Error deleting session: {e}")
            return False
    
    async def extend_session(self, session_id: str, ttl: Optional[int] = None) -> bool:
        """Extend session TTL"""
        try:
            ttl = ttl or self.default_ttl
            return bool(self.redis.expire(f"session:{session_id}", ttl))
        except Exception as e:
            print(f"Error extending session: {e}")
            return False


class ConversationCache:
    def __init__(self):
        self.redis = redis_client
        self.default_ttl = 7200  # 2 hours
    
    async def set_conversation_context(self, conversation_id: str, context: dict, ttl: Optional[int] = None) -> bool:
        """Store conversation context"""
        try:
            ttl = ttl or self.default_ttl
            return self.redis.setex(f"conversation:{conversation_id}", ttl, str(context))
        except Exception as e:
            print(f"Error setting conversation context: {e}")
            return False
    
    async def get_conversation_context(self, conversation_id: str) -> Optional[dict]:
        """Retrieve conversation context"""
        try:
            data = self.redis.get(f"conversation:{conversation_id}")
            if data:
                return eval(data)  # Note: In production, use proper JSON serialization
            return None
        except Exception as e:
            print(f"Error getting conversation context: {e}")
            return None
    
    async def update_conversation_context(self, conversation_id: str, context: dict) -> bool:
        """Update conversation context while preserving TTL"""
        try:
            # Get current TTL
            ttl = self.redis.ttl(f"conversation:{conversation_id}")
            if ttl <= 0:
                ttl = self.default_ttl
            
            return self.redis.setex(f"conversation:{conversation_id}", ttl, str(context))
        except Exception as e:
            print(f"Error updating conversation context: {e}")
            return False


# Global instances
session_manager = SessionManager()
conversation_cache = ConversationCache()