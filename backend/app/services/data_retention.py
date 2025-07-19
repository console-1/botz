"""
Data retention and deletion policies service for GDPR compliance
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from uuid import UUID
from enum import Enum
import asyncio
from dataclasses import dataclass

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, text, delete
from pydantic import BaseModel

from app.core.database import get_db
from app.models.client import Client, UsageRecord
from app.models.conversation import Conversation, Message, UsageMetric
from app.services.audit_logging import audit_logger, AuditEventType, AuditSeverity
from app.core.config import settings

class RetentionPolicyType(str, Enum):
    """Types of retention policies"""
    TIME_BASED = "time_based"           # Delete after X time
    COUNT_BASED = "count_based"         # Keep only last X records
    SIZE_BASED = "size_based"           # Keep only X MB of data
    CONDITIONAL = "conditional"         # Delete based on conditions
    IMMEDIATE = "immediate"             # Delete immediately
    LEGAL_HOLD = "legal_hold"          # Never delete (legal requirement)

class DataCategory(str, Enum):
    """Categories of data for retention policies"""
    CONVERSATION_DATA = "conversation_data"
    MESSAGE_DATA = "message_data"
    USAGE_ANALYTICS = "usage_analytics"
    AUDIT_LOGS = "audit_logs"
    CLIENT_DATA = "client_data"
    SESSION_DATA = "session_data"
    SYSTEM_LOGS = "system_logs"

@dataclass
class RetentionPolicy:
    """Data retention policy definition"""
    name: str
    data_category: DataCategory
    policy_type: RetentionPolicyType
    retention_period_days: int
    conditions: Optional[Dict[str, Any]] = None
    priority: int = 1  # Higher priority policies override lower ones
    enabled: bool = True
    compliance_basis: str = "GDPR Article 5(1)(e)"  # Legal basis for retention

class RetentionExecutionResult(BaseModel):
    """Result of retention policy execution"""
    policy_name: str
    execution_date: datetime
    records_evaluated: int
    records_deleted: int
    records_anonymized: int
    data_size_freed_mb: float
    execution_time_seconds: float
    errors: List[str] = []
    warnings: List[str] = []

class DataRetentionService:
    """
    Service for managing data retention and deletion policies
    """
    
    def __init__(self):
        self.policies = self._get_default_policies()
        self.dry_run_mode = False
    
    def _get_default_policies(self) -> List[RetentionPolicy]:
        """Get default data retention policies based on GDPR and business requirements"""
        return [
            # Conversation data - 1 year retention
            RetentionPolicy(
                name="conversation_data_retention",
                data_category=DataCategory.CONVERSATION_DATA,
                policy_type=RetentionPolicyType.TIME_BASED,
                retention_period_days=365,
                conditions={"status": "closed"},
                priority=1,
                compliance_basis="GDPR Article 5(1)(e) - Data minimization"
            ),
            
            # Message data - 1 year retention
            RetentionPolicy(
                name="message_data_retention", 
                data_category=DataCategory.MESSAGE_DATA,
                policy_type=RetentionPolicyType.TIME_BASED,
                retention_period_days=365,
                priority=1,
                compliance_basis="GDPR Article 5(1)(e) - Data minimization"
            ),
            
            # Usage analytics - 3 years for business analysis
            RetentionPolicy(
                name="usage_analytics_retention",
                data_category=DataCategory.USAGE_ANALYTICS,
                policy_type=RetentionPolicyType.TIME_BASED,
                retention_period_days=1095,  # 3 years
                priority=1,
                compliance_basis="Legitimate business interest"
            ),
            
            # Audit logs - 7 years for compliance
            RetentionPolicy(
                name="audit_logs_retention",
                data_category=DataCategory.AUDIT_LOGS,
                policy_type=RetentionPolicyType.TIME_BASED,
                retention_period_days=2555,  # 7 years
                priority=1,
                compliance_basis="Legal obligation - SOX compliance"
            ),
            
            # System logs - 90 days
            RetentionPolicy(
                name="system_logs_retention",
                data_category=DataCategory.SYSTEM_LOGS,
                policy_type=RetentionPolicyType.TIME_BASED,
                retention_period_days=90,
                priority=1,
                compliance_basis="Operational requirement"
            ),
            
            # Client data - Legal hold (never delete automatically)
            RetentionPolicy(
                name="client_data_legal_hold",
                data_category=DataCategory.CLIENT_DATA,
                policy_type=RetentionPolicyType.LEGAL_HOLD,
                retention_period_days=999999,  # Effectively never
                priority=1,
                compliance_basis="Contractual obligation"
            ),
            
            # Session data - 30 days
            RetentionPolicy(
                name="session_data_retention",
                data_category=DataCategory.SESSION_DATA,
                policy_type=RetentionPolicyType.TIME_BASED,
                retention_period_days=30,
                priority=1,
                compliance_basis="Security and operational requirement"
            )
        ]
    
    async def execute_retention_policies(
        self,
        db: Session,
        specific_policies: Optional[List[str]] = None,
        dry_run: bool = False
    ) -> List[RetentionExecutionResult]:
        """
        Execute data retention policies
        """
        self.dry_run_mode = dry_run
        results = []
        
        # Filter policies to execute
        policies_to_execute = self.policies
        if specific_policies:
            policies_to_execute = [p for p in self.policies if p.name in specific_policies]
        
        # Sort by priority (higher priority first)
        policies_to_execute.sort(key=lambda x: x.priority, reverse=True)
        
        for policy in policies_to_execute:
            if not policy.enabled:
                continue
                
            start_time = datetime.now(timezone.utc)
            
            try:
                result = await self._execute_single_policy(db, policy)
                result.execution_time_seconds = (
                    datetime.now(timezone.utc) - start_time
                ).total_seconds()
                
                # Log the retention policy execution
                await audit_logger.log_event({
                    "event_type": AuditEventType.DATA_RETENTION_APPLIED,
                    "severity": AuditSeverity.MEDIUM,
                    "actor_type": "system",
                    "actor_id": "retention_service",
                    "action": f"policy_executed_{policy.name}",
                    "description": f"Executed retention policy: {policy.name}",
                    "event_data": {
                        "policy_name": policy.name,
                        "records_deleted": result.records_deleted,
                        "dry_run": dry_run
                    },
                    "compliance_relevant": True
                })
                
                results.append(result)
                
            except Exception as e:
                error_result = RetentionExecutionResult(
                    policy_name=policy.name,
                    execution_date=start_time,
                    records_evaluated=0,
                    records_deleted=0,
                    records_anonymized=0,
                    data_size_freed_mb=0.0,
                    execution_time_seconds=(
                        datetime.now(timezone.utc) - start_time
                    ).total_seconds(),
                    errors=[f"Policy execution failed: {str(e)}"]
                )
                results.append(error_result)
                
                # Log the error
                await audit_logger.log_system_error(
                    error_type="retention_policy_failure",
                    error_message=str(e),
                    component="data_retention_service",
                    severity=AuditSeverity.HIGH,
                    policy_name=policy.name
                )
        
        return results
    
    async def _execute_single_policy(
        self,
        db: Session,
        policy: RetentionPolicy
    ) -> RetentionExecutionResult:
        """Execute a single retention policy"""
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=policy.retention_period_days)
        
        result = RetentionExecutionResult(
            policy_name=policy.name,
            execution_date=datetime.now(timezone.utc),
            records_evaluated=0,
            records_deleted=0,
            records_anonymized=0,
            data_size_freed_mb=0.0,
            execution_time_seconds=0.0
        )
        
        if policy.data_category == DataCategory.CONVERSATION_DATA:
            return await self._apply_conversation_retention(db, policy, cutoff_date, result)
        elif policy.data_category == DataCategory.MESSAGE_DATA:
            return await self._apply_message_retention(db, policy, cutoff_date, result)
        elif policy.data_category == DataCategory.USAGE_ANALYTICS:
            return await self._apply_usage_analytics_retention(db, policy, cutoff_date, result)
        elif policy.data_category == DataCategory.AUDIT_LOGS:
            return await self._apply_audit_logs_retention(db, policy, cutoff_date, result)
        elif policy.data_category == DataCategory.SESSION_DATA:
            return await self._apply_session_data_retention(db, policy, cutoff_date, result)
        else:
            result.warnings.append(f"No handler for data category: {policy.data_category}")
            return result
    
    async def _apply_conversation_retention(
        self,
        db: Session,
        policy: RetentionPolicy,
        cutoff_date: datetime,
        result: RetentionExecutionResult
    ) -> RetentionExecutionResult:
        """Apply retention policy to conversation data"""
        
        # Build query for conversations to delete
        query = db.query(Conversation).filter(Conversation.created_at < cutoff_date)
        
        # Apply conditions if specified
        if policy.conditions:
            for key, value in policy.conditions.items():
                if hasattr(Conversation, key):
                    query = query.filter(getattr(Conversation, key) == value)
        
        # Get conversations to delete
        conversations_to_delete = query.all()
        result.records_evaluated = len(conversations_to_delete)
        
        if not self.dry_run_mode:
            for conversation in conversations_to_delete:
                # Delete associated messages first
                message_count = db.query(func.count(Message.id)).filter(
                    Message.conversation_id == conversation.conversation_id
                ).scalar()
                
                db.query(Message).filter(
                    Message.conversation_id == conversation.conversation_id
                ).delete()
                
                # Delete the conversation
                db.delete(conversation)
                result.records_deleted += 1 + (message_count or 0)
            
            db.commit()
        else:
            # In dry run, just count what would be deleted
            for conversation in conversations_to_delete:
                message_count = db.query(func.count(Message.id)).filter(
                    Message.conversation_id == conversation.conversation_id
                ).scalar()
                result.records_deleted += 1 + (message_count or 0)
        
        # Estimate data size freed (rough calculation)
        result.data_size_freed_mb = result.records_deleted * 0.01  # ~10KB per record
        
        return result
    
    async def _apply_message_retention(
        self,
        db: Session,
        policy: RetentionPolicy,
        cutoff_date: datetime,
        result: RetentionExecutionResult
    ) -> RetentionExecutionResult:
        """Apply retention policy to message data"""
        
        # Find orphaned messages (messages without conversations)
        orphaned_messages = db.query(Message).outerjoin(Conversation).filter(
            and_(
                Message.created_at < cutoff_date,
                Conversation.id.is_(None)
            )
        ).all()
        
        result.records_evaluated = len(orphaned_messages)
        
        if not self.dry_run_mode:
            for message in orphaned_messages:
                db.delete(message)
                result.records_deleted += 1
            
            db.commit()
        else:
            result.records_deleted = len(orphaned_messages)
        
        result.data_size_freed_mb = result.records_deleted * 0.005  # ~5KB per message
        
        return result
    
    async def _apply_usage_analytics_retention(
        self,
        db: Session,
        policy: RetentionPolicy,
        cutoff_date: datetime,
        result: RetentionExecutionResult
    ) -> RetentionExecutionResult:
        """Apply retention policy to usage analytics data"""
        
        # Delete old usage records
        old_usage_records = db.query(UsageRecord).filter(
            UsageRecord.created_at < cutoff_date
        ).all()
        
        result.records_evaluated = len(old_usage_records)
        
        if not self.dry_run_mode:
            for record in old_usage_records:
                db.delete(record)
                result.records_deleted += 1
            
            # Also delete old usage metrics
            old_usage_metrics = db.query(UsageMetric).filter(
                UsageMetric.created_at < cutoff_date
            ).all()
            
            for metric in old_usage_metrics:
                db.delete(metric)
                result.records_deleted += 1
            
            db.commit()
        else:
            result.records_deleted = len(old_usage_records)
            old_metrics_count = db.query(func.count(UsageMetric.id)).filter(
                UsageMetric.created_at < cutoff_date
            ).scalar()
            result.records_deleted += old_metrics_count or 0
        
        result.data_size_freed_mb = result.records_deleted * 0.002  # ~2KB per record
        
        return result
    
    async def _apply_audit_logs_retention(
        self,
        db: Session,
        policy: RetentionPolicy,
        cutoff_date: datetime,
        result: RetentionExecutionResult
    ) -> RetentionExecutionResult:
        """Apply retention policy to audit logs"""
        
        # For audit logs, we might want to archive rather than delete
        # or apply different retention periods based on severity
        
        from app.services.audit_logging import AuditLog
        
        # Only delete low-severity audit logs older than retention period
        old_audit_logs = db.query(AuditLog).filter(
            and_(
                AuditLog.timestamp < cutoff_date,
                AuditLog.severity == "low",
                AuditLog.compliance_relevant == False
            )
        ).all()
        
        result.records_evaluated = len(old_audit_logs)
        
        if not self.dry_run_mode:
            for log in old_audit_logs:
                db.delete(log)
                result.records_deleted += 1
            
            db.commit()
        else:
            result.records_deleted = len(old_audit_logs)
        
        result.data_size_freed_mb = result.records_deleted * 0.003  # ~3KB per log
        
        return result
    
    async def _apply_session_data_retention(
        self,
        db: Session,
        policy: RetentionPolicy,
        cutoff_date: datetime,
        result: RetentionExecutionResult
    ) -> RetentionExecutionResult:
        """Apply retention policy to session data"""
        
        # This would apply to Redis session data in production
        # For now, we'll simulate it
        
        result.records_evaluated = 0  # Would query Redis
        result.records_deleted = 0    # Would delete from Redis
        result.data_size_freed_mb = 0.0
        result.warnings.append("Session data retention not implemented (Redis integration required)")
        
        return result
    
    def add_custom_policy(self, policy: RetentionPolicy):
        """Add a custom retention policy"""
        self.policies.append(policy)
    
    def remove_policy(self, policy_name: str) -> bool:
        """Remove a retention policy by name"""
        for i, policy in enumerate(self.policies):
            if policy.name == policy_name:
                del self.policies[i]
                return True
        return False
    
    def get_policy(self, policy_name: str) -> Optional[RetentionPolicy]:
        """Get a retention policy by name"""
        for policy in self.policies:
            if policy.name == policy_name:
                return policy
        return None
    
    def list_policies(self) -> List[RetentionPolicy]:
        """List all retention policies"""
        return self.policies.copy()
    
    async def estimate_storage_savings(self, db: Session) -> Dict[str, Any]:
        """Estimate storage savings from applying retention policies"""
        
        total_savings_mb = 0.0
        policy_estimates = {}
        
        for policy in self.policies:
            if not policy.enabled:
                continue
            
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=policy.retention_period_days)
            
            # Dry run to estimate savings
            self.dry_run_mode = True
            result = await self._execute_single_policy(db, policy)
            
            policy_estimates[policy.name] = {
                "records_to_delete": result.records_deleted,
                "estimated_savings_mb": result.data_size_freed_mb,
                "data_category": policy.data_category.value
            }
            
            total_savings_mb += result.data_size_freed_mb
        
        return {
            "total_estimated_savings_mb": total_savings_mb,
            "total_estimated_savings_gb": total_savings_mb / 1024,
            "policy_breakdown": policy_estimates,
            "estimation_date": datetime.now(timezone.utc).isoformat()
        }
    
    async def create_retention_schedule(self) -> Dict[str, Any]:
        """Create a retention schedule showing when data will be deleted"""
        
        schedule = {
            "schedule_created": datetime.now(timezone.utc).isoformat(),
            "policies": []
        }
        
        for policy in self.policies:
            if not policy.enabled:
                continue
            
            next_execution = datetime.now(timezone.utc) + timedelta(days=1)  # Daily execution
            cutoff_date = next_execution - timedelta(days=policy.retention_period_days)
            
            schedule["policies"].append({
                "policy_name": policy.name,
                "data_category": policy.data_category.value,
                "retention_period_days": policy.retention_period_days,
                "next_execution": next_execution.isoformat(),
                "cutoff_date": cutoff_date.isoformat(),
                "compliance_basis": policy.compliance_basis
            })
        
        return schedule

# Global data retention service instance
data_retention_service = DataRetentionService()

def get_data_retention_service() -> DataRetentionService:
    """Get data retention service instance"""
    return data_retention_service

# Scheduled task function for automated retention
async def run_scheduled_retention():
    """Run retention policies on schedule (daily)"""
    try:
        db = next(get_db())
        results = await data_retention_service.execute_retention_policies(db, dry_run=False)
        
        total_deleted = sum(r.records_deleted for r in results)
        total_freed_mb = sum(r.data_size_freed_mb for r in results)
        
        await audit_logger.log_event({
            "event_type": AuditEventType.DATA_RETENTION_APPLIED,
            "severity": AuditSeverity.MEDIUM,
            "actor_type": "system",
            "actor_id": "scheduled_retention",
            "action": "scheduled_retention_executed",
            "description": f"Scheduled retention deleted {total_deleted} records, freed {total_freed_mb:.2f} MB",
            "event_data": {
                "total_records_deleted": total_deleted,
                "total_mb_freed": total_freed_mb,
                "policies_executed": len(results)
            },
            "compliance_relevant": True
        })
        
        db.close()
        
    except Exception as e:
        await audit_logger.log_system_error(
            error_type="scheduled_retention_failure",
            error_message=str(e),
            component="data_retention_service",
            severity=AuditSeverity.CRITICAL
        )