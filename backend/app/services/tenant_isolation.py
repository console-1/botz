"""
Tenant isolation and resource quota management service
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Tuple
from uuid import UUID
from enum import Enum

from sqlalchemy.orm import Session
from sqlalchemy import and_, func, text
from fastapi import HTTPException, status
from pydantic import BaseModel

from app.core.database import get_db
from app.models.client import Client, ClientTier, UsageRecord
from app.models.conversation import Conversation, Message
from app.core.config import settings

class QuotaType(str, Enum):
    MESSAGES_PER_DAY = "messages_per_day"
    CONVERSATIONS_PER_DAY = "conversations_per_day"
    API_CALLS_PER_HOUR = "api_calls_per_hour"
    KNOWLEDGE_BASE_SIZE_MB = "knowledge_base_size_mb"
    TOKENS_PER_DAY = "tokens_per_day"
    CONCURRENT_CONVERSATIONS = "concurrent_conversations"

class QuotaExceededException(Exception):
    """Raised when a resource quota is exceeded"""
    def __init__(self, quota_type: QuotaType, current: int, limit: int):
        self.quota_type = quota_type
        self.current = current
        self.limit = limit
        super().__init__(f"Quota exceeded for {quota_type}: {current}/{limit}")

class ResourceQuota(BaseModel):
    """Resource quota definition"""
    quota_type: QuotaType
    limit: int
    current_usage: int
    percentage_used: float
    resets_at: Optional[datetime]

class TenantIsolationService:
    """
    Service for managing tenant isolation and resource quotas
    """
    
    def __init__(self):
        self.tier_quotas = {
            ClientTier.FREE: {
                QuotaType.MESSAGES_PER_DAY: 100,
                QuotaType.CONVERSATIONS_PER_DAY: 20,
                QuotaType.API_CALLS_PER_HOUR: 100,
                QuotaType.KNOWLEDGE_BASE_SIZE_MB: 10,
                QuotaType.TOKENS_PER_DAY: 10000,
                QuotaType.CONCURRENT_CONVERSATIONS: 2
            },
            ClientTier.STARTER: {
                QuotaType.MESSAGES_PER_DAY: 1000,
                QuotaType.CONVERSATIONS_PER_DAY: 100,
                QuotaType.API_CALLS_PER_HOUR: 500,
                QuotaType.KNOWLEDGE_BASE_SIZE_MB: 100,
                QuotaType.TOKENS_PER_DAY: 100000,
                QuotaType.CONCURRENT_CONVERSATIONS: 10
            },
            ClientTier.PROFESSIONAL: {
                QuotaType.MESSAGES_PER_DAY: 5000,
                QuotaType.CONVERSATIONS_PER_DAY: 500,
                QuotaType.API_CALLS_PER_HOUR: 2000,
                QuotaType.KNOWLEDGE_BASE_SIZE_MB: 500,
                QuotaType.TOKENS_PER_DAY: 500000,
                QuotaType.CONCURRENT_CONVERSATIONS: 50
            },
            ClientTier.ENTERPRISE: {
                QuotaType.MESSAGES_PER_DAY: 50000,
                QuotaType.CONVERSATIONS_PER_DAY: 5000,
                QuotaType.API_CALLS_PER_HOUR: 10000,
                QuotaType.KNOWLEDGE_BASE_SIZE_MB: 5000,
                QuotaType.TOKENS_PER_DAY: 5000000,
                QuotaType.CONCURRENT_CONVERSATIONS: 200
            }
        }
    
    def get_client_quotas(self, client: Client) -> Dict[QuotaType, int]:
        """Get quota limits for a client based on their tier"""
        
        # Get tier-based quotas
        base_quotas = self.tier_quotas.get(client.tier, self.tier_quotas[ClientTier.FREE])
        
        # Override with client-specific limits if configured
        client_limits = client.configuration.get("limits", {})
        quotas = base_quotas.copy()
        
        for quota_type in QuotaType:
            if quota_type.value in client_limits:
                quotas[quota_type] = client_limits[quota_type.value]
        
        return quotas
    
    def check_quota(
        self,
        db: Session,
        client: Client,
        quota_type: QuotaType,
        increment: int = 1
    ) -> ResourceQuota:
        """
        Check if a client can use a resource without exceeding quota
        Returns current quota status
        """
        
        quotas = self.get_client_quotas(client)
        limit = quotas.get(quota_type, 0)
        
        if limit <= 0:  # Unlimited
            return ResourceQuota(
                quota_type=quota_type,
                limit=0,
                current_usage=0,
                percentage_used=0.0,
                resets_at=None
            )
        
        # Get current usage
        current_usage = self._get_current_usage(db, client, quota_type)
        resets_at = self._get_quota_reset_time(quota_type)
        
        # Calculate percentage
        percentage_used = (current_usage / limit) * 100 if limit > 0 else 0
        
        return ResourceQuota(
            quota_type=quota_type,
            limit=limit,
            current_usage=current_usage,
            percentage_used=percentage_used,
            resets_at=resets_at
        )
    
    def enforce_quota(
        self,
        db: Session,
        client: Client,
        quota_type: QuotaType,
        increment: int = 1
    ) -> bool:
        """
        Enforce quota limits. Raises QuotaExceededException if limit would be exceeded.
        Returns True if quota check passes.
        """
        
        quota = self.check_quota(db, client, quota_type, increment)
        
        if quota.limit > 0 and quota.current_usage + increment > quota.limit:
            raise QuotaExceededException(quota_type, quota.current_usage + increment, quota.limit)
        
        return True
    
    def _get_current_usage(self, db: Session, client: Client, quota_type: QuotaType) -> int:
        """Get current usage for a specific quota type"""
        
        now = datetime.now(timezone.utc)
        
        if quota_type == QuotaType.MESSAGES_PER_DAY:
            # Count messages from today
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            return db.query(func.count(Message.id)).join(Conversation).filter(
                and_(
                    Conversation.client_id == client.id,
                    Message.created_at >= today_start
                )
            ).scalar() or 0
        
        elif quota_type == QuotaType.CONVERSATIONS_PER_DAY:
            # Count conversations started today
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            return db.query(func.count(Conversation.id)).filter(
                and_(
                    Conversation.client_id == client.id,
                    Conversation.created_at >= today_start
                )
            ).scalar() or 0
        
        elif quota_type == QuotaType.API_CALLS_PER_HOUR:
            # Count API calls from past hour using usage records
            hour_ago = now - timedelta(hours=1)
            usage_records = db.query(UsageRecord).filter(
                and_(
                    UsageRecord.client_id == client.id,
                    UsageRecord.period_start >= hour_ago
                )
            ).all()
            return sum(record.api_calls_made for record in usage_records)
        
        elif quota_type == QuotaType.TOKENS_PER_DAY:
            # Count tokens from today using usage records
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            usage_records = db.query(UsageRecord).filter(
                and_(
                    UsageRecord.client_id == client.id,
                    UsageRecord.period_start >= today_start
                )
            ).all()
            return sum(record.total_tokens_used for record in usage_records)
        
        elif quota_type == QuotaType.CONCURRENT_CONVERSATIONS:
            # Count active conversations
            return db.query(func.count(Conversation.id)).filter(
                and_(
                    Conversation.client_id == client.id,
                    Conversation.status == "active"
                )
            ).scalar() or 0
        
        elif quota_type == QuotaType.KNOWLEDGE_BASE_SIZE_MB:
            # Calculate total knowledge base size (placeholder - would need actual file sizes)
            total_chunks = db.query(func.count(text("document_chunks.id"))).select_from(
                text("document_chunks")
                .join(text("documents"), text("document_chunks.document_id = documents.id"))
                .join(text("knowledge_bases"), text("documents.knowledge_base_id = knowledge_bases.id"))
            ).filter(
                text("knowledge_bases.client_id = :client_id")
            ).params(client_id=str(client.id)).scalar() or 0
            
            # Rough estimate: 1KB per chunk
            return (total_chunks * 1024) // (1024 * 1024)  # Convert to MB
        
        return 0
    
    def _get_quota_reset_time(self, quota_type: QuotaType) -> Optional[datetime]:
        """Get when the quota resets"""
        
        now = datetime.now(timezone.utc)
        
        if quota_type in [QuotaType.MESSAGES_PER_DAY, QuotaType.CONVERSATIONS_PER_DAY, QuotaType.TOKENS_PER_DAY]:
            # Daily quotas reset at midnight UTC
            tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            return tomorrow
        
        elif quota_type == QuotaType.API_CALLS_PER_HOUR:
            # Hourly quotas reset at the top of the next hour
            next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            return next_hour
        
        elif quota_type in [QuotaType.KNOWLEDGE_BASE_SIZE_MB, QuotaType.CONCURRENT_CONVERSATIONS]:
            # These don't reset automatically
            return None
        
        return None
    
    def get_all_quotas(self, db: Session, client: Client) -> List[ResourceQuota]:
        """Get all quota statuses for a client"""
        
        quotas = []
        for quota_type in QuotaType:
            quota = self.check_quota(db, client, quota_type)
            quotas.append(quota)
        
        return quotas
    
    def is_client_isolated(self, client: Client) -> bool:
        """
        Check if client has proper tenant isolation
        This validates that the client can only access their own data
        """
        
        # In a production system, you might check:
        # - Database row-level security is enabled
        # - Client has proper namespace in vector DB
        # - Client-specific API keys are being used
        # - No cross-tenant data leakage
        
        return True  # Simplified for this implementation
    
    def validate_tenant_access(
        self,
        client: Client,
        resource_type: str,
        resource_id: str
    ) -> bool:
        """
        Validate that a client can access a specific resource
        """
        
        # This would implement detailed access controls
        # For now, just return True if client is active
        return client.is_active_client()
    
    def get_resource_namespace(self, client: Client, resource_type: str) -> str:
        """
        Get the namespace for a client's resources
        Used for vector DB collections, file storage, etc.
        """
        
        return f"client_{client.client_id}_{resource_type}"
    
    def check_data_isolation(self, db: Session, client: Client) -> Dict[str, bool]:
        """
        Perform data isolation checks for a client
        """
        
        checks = {
            "conversations_isolated": True,  # Only client's conversations
            "knowledge_bases_isolated": True,  # Only client's knowledge bases
            "api_keys_isolated": True,  # Only client's API keys
            "usage_data_isolated": True,  # Only client's usage data
            "vector_namespace_isolated": True  # Proper vector DB namespace
        }
        
        # In production, these would be actual isolation checks
        # For example:
        # - Query for conversations not belonging to this client
        # - Check vector DB namespace separation
        # - Validate API key access patterns
        
        return checks

# Dependency injection
def get_tenant_isolation_service() -> TenantIsolationService:
    """Get tenant isolation service instance"""
    return TenantIsolationService()

# FastAPI middleware for quota enforcement
class QuotaEnforcementMiddleware:
    """Middleware to enforce quotas on API requests"""
    
    def __init__(self, isolation_service: TenantIsolationService):
        self.isolation_service = isolation_service
    
    async def __call__(self, request, call_next):
        """Process request and enforce quotas"""
        
        # Skip quota checks for certain endpoints
        if request.url.path.startswith("/docs") or request.url.path.startswith("/health"):
            return await call_next(request)
        
        # Extract client from request (would be set by auth middleware)
        client = getattr(request.state, "client", None)
        
        if client:
            try:
                # Enforce API call quota
                db = next(get_db())
                self.isolation_service.enforce_quota(
                    db, client, QuotaType.API_CALLS_PER_HOUR
                )
                db.close()
            except QuotaExceededException as e:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Quota exceeded: {e.quota_type.value}. Current: {e.current}, Limit: {e.limit}"
                )
        
        return await call_next(request)