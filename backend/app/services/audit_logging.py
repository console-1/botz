"""
Comprehensive audit logging service for compliance and security monitoring
"""

import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4
from enum import Enum
import asyncio
from contextlib import contextmanager

from sqlalchemy import Column, String, DateTime, Text, JSON, Boolean, Integer, Index
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Session
from pydantic import BaseModel
import structlog

from app.core.database import Base, get_db
from app.core.config import settings

class AuditEventType(str, Enum):
    """Types of audit events"""
    # Authentication and Authorization
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    
    # Data Access and Modification
    DATA_ACCESS = "data_access"
    DATA_CREATION = "data_creation"
    DATA_MODIFICATION = "data_modification"
    DATA_DELETION = "data_deletion"
    DATA_EXPORT = "data_export"
    
    # GDPR and Privacy
    GDPR_REQUEST_SUBMITTED = "gdpr_request_submitted"
    GDPR_REQUEST_PROCESSED = "gdpr_request_processed"
    DATA_ANONYMIZATION = "data_anonymization"
    DATA_RETENTION_APPLIED = "data_retention_applied"
    CONSENT_UPDATED = "consent_updated"
    
    # System Events
    SYSTEM_START = "system_start"
    SYSTEM_SHUTDOWN = "system_shutdown"
    DATABASE_MIGRATION = "database_migration"
    CONFIGURATION_CHANGE = "configuration_change"
    
    # Business Events
    CLIENT_CREATED = "client_created"
    CLIENT_UPDATED = "client_updated"
    CLIENT_SUSPENDED = "client_suspended"
    CONVERSATION_STARTED = "conversation_started"
    MESSAGE_SENT = "message_sent"
    ESCALATION_TRIGGERED = "escalation_triggered"
    
    # Security Events
    SECURITY_ALERT = "security_alert"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    DATA_BREACH_DETECTED = "data_breach_detected"
    
    # Error Events
    SYSTEM_ERROR = "system_error"
    API_ERROR = "api_error"
    DATABASE_ERROR = "database_error"
    EXTERNAL_SERVICE_ERROR = "external_service_error"

class AuditSeverity(str, Enum):
    """Severity levels for audit events"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AuditLog(Base):
    """Audit log table for storing all system events"""
    __tablename__ = "audit_logs"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    event_id = Column(String(64), unique=True, nullable=False, index=True)
    
    # Event metadata
    event_type = Column(String(50), nullable=False, index=True)
    severity = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # Actor information
    actor_type = Column(String(20))  # user, system, api_key, admin
    actor_id = Column(String(255), index=True)
    actor_ip = Column(String(45))  # IPv6 support
    user_agent = Column(Text)
    
    # Resource information
    resource_type = Column(String(50))  # conversation, message, client, etc.
    resource_id = Column(String(255), index=True)
    client_id = Column(PGUUID(as_uuid=True), index=True)
    
    # Event details
    action = Column(String(100), nullable=False)
    description = Column(Text)
    event_data = Column(JSON)
    
    # Context and correlation
    session_id = Column(String(255), index=True)
    request_id = Column(String(255), index=True)
    correlation_id = Column(String(255), index=True)
    
    # Compliance and security
    is_sensitive = Column(Boolean, default=False)
    compliance_relevant = Column(Boolean, default=False)
    retention_period_days = Column(Integer, default=2555)  # 7 years default
    
    # Data integrity
    event_hash = Column(String(64))  # SHA-256 hash for integrity
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_audit_timestamp_type', 'timestamp', 'event_type'),
        Index('idx_audit_client_timestamp', 'client_id', 'timestamp'),
        Index('idx_audit_actor_timestamp', 'actor_id', 'timestamp'),
        Index('idx_audit_resource', 'resource_type', 'resource_id'),
        Index('idx_audit_compliance', 'compliance_relevant', 'timestamp'),
    )

class AuditEvent(BaseModel):
    """Audit event data structure"""
    event_type: AuditEventType
    severity: AuditSeverity
    actor_type: Optional[str] = None
    actor_id: Optional[str] = None
    actor_ip: Optional[str] = None
    user_agent: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    client_id: Optional[UUID] = None
    action: str
    description: Optional[str] = None
    event_data: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None
    is_sensitive: bool = False
    compliance_relevant: bool = False

class AuditLogger:
    """
    Comprehensive audit logging service for compliance and security
    """
    
    def __init__(self):
        self.logger = structlog.get_logger("audit")
        self.batch_size = 100
        self.event_buffer: List[AuditEvent] = []
        self.buffer_lock = asyncio.Lock()
    
    async def log_event(self, event: AuditEvent, db: Optional[Session] = None) -> str:
        """
        Log an audit event with comprehensive metadata
        """
        event_id = self._generate_event_id(event)
        
        # Create audit log record
        audit_record = AuditLog(
            event_id=event_id,
            event_type=event.event_type.value,
            severity=event.severity.value,
            timestamp=datetime.now(timezone.utc),
            actor_type=event.actor_type,
            actor_id=event.actor_id,
            actor_ip=event.actor_ip,
            user_agent=event.user_agent,
            resource_type=event.resource_type,
            resource_id=event.resource_id,
            client_id=event.client_id,
            action=event.action,
            description=event.description,
            event_data=event.event_data or {},
            session_id=event.session_id,
            request_id=event.request_id,
            correlation_id=event.correlation_id,
            is_sensitive=event.is_sensitive,
            compliance_relevant=event.compliance_relevant
        )
        
        # Calculate integrity hash
        audit_record.event_hash = self._calculate_event_hash(audit_record)
        
        # Store in database
        if db:
            db.add(audit_record)
            try:
                db.commit()
            except Exception as e:
                db.rollback()
                self.logger.error("Failed to store audit event", error=str(e), event_id=event_id)
        else:
            # Buffer for batch processing
            async with self.buffer_lock:
                self.event_buffer.append(event)
                if len(self.event_buffer) >= self.batch_size:
                    await self._flush_buffer()
        
        # Also log to structured logger
        self.logger.info(
            "audit_event",
            event_id=event_id,
            event_type=event.event_type.value,
            severity=event.severity.value,
            actor_id=event.actor_id,
            resource_id=event.resource_id,
            action=event.action,
            client_id=str(event.client_id) if event.client_id else None
        )
        
        return event_id
    
    def log_authentication_success(
        self,
        actor_id: str,
        actor_ip: str,
        method: str = "api_key",
        client_id: Optional[UUID] = None,
        **kwargs
    ):
        """Log successful authentication"""
        event = AuditEvent(
            event_type=AuditEventType.LOGIN_SUCCESS,
            severity=AuditSeverity.LOW,
            actor_type="api_key" if method == "api_key" else "user",
            actor_id=actor_id,
            actor_ip=actor_ip,
            client_id=client_id,
            action=f"authentication_success_{method}",
            description=f"Successful authentication using {method}",
            event_data={"method": method, **kwargs},
            compliance_relevant=True
        )
        return asyncio.create_task(self.log_event(event))
    
    def log_authentication_failure(
        self,
        actor_id: Optional[str],
        actor_ip: str,
        reason: str,
        method: str = "api_key",
        **kwargs
    ):
        """Log failed authentication attempt"""
        event = AuditEvent(
            event_type=AuditEventType.LOGIN_FAILURE,
            severity=AuditSeverity.MEDIUM,
            actor_type="unknown",
            actor_id=actor_id or "unknown",
            actor_ip=actor_ip,
            action=f"authentication_failure_{method}",
            description=f"Authentication failed: {reason}",
            event_data={"reason": reason, "method": method, **kwargs},
            compliance_relevant=True
        )
        return asyncio.create_task(self.log_event(event))
    
    def log_data_access(
        self,
        actor_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        client_id: Optional[UUID] = None,
        sensitive: bool = False,
        **kwargs
    ):
        """Log data access events"""
        event = AuditEvent(
            event_type=AuditEventType.DATA_ACCESS,
            severity=AuditSeverity.MEDIUM if sensitive else AuditSeverity.LOW,
            actor_type="api_key",
            actor_id=actor_id,
            resource_type=resource_type,
            resource_id=resource_id,
            client_id=client_id,
            action=action,
            description=f"Data access: {action} on {resource_type}",
            event_data=kwargs,
            is_sensitive=sensitive,
            compliance_relevant=True
        )
        return asyncio.create_task(self.log_event(event))
    
    def log_gdpr_request(
        self,
        request_type: str,
        data_subject_id: str,
        status: str,
        details: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Log GDPR data subject requests"""
        event = AuditEvent(
            event_type=AuditEventType.GDPR_REQUEST_SUBMITTED if status == "submitted" 
                      else AuditEventType.GDPR_REQUEST_PROCESSED,
            severity=AuditSeverity.HIGH,
            actor_type="data_subject",
            actor_id=data_subject_id,
            resource_type="personal_data",
            resource_id=data_subject_id,
            action=f"gdpr_{request_type}_{status}",
            description=f"GDPR {request_type} request {status}",
            event_data={"request_type": request_type, "details": details, **kwargs},
            is_sensitive=True,
            compliance_relevant=True
        )
        return asyncio.create_task(self.log_event(event))
    
    def log_system_error(
        self,
        error_type: str,
        error_message: str,
        component: str,
        severity: AuditSeverity = AuditSeverity.MEDIUM,
        **kwargs
    ):
        """Log system errors and exceptions"""
        event = AuditEvent(
            event_type=AuditEventType.SYSTEM_ERROR,
            severity=severity,
            actor_type="system",
            actor_id="system",
            resource_type="system",
            resource_id=component,
            action=f"error_{error_type}",
            description=f"System error in {component}: {error_message}",
            event_data={"error_type": error_type, "component": component, **kwargs}
        )
        return asyncio.create_task(self.log_event(event))
    
    def log_security_event(
        self,
        security_event_type: str,
        description: str,
        actor_ip: Optional[str] = None,
        severity: AuditSeverity = AuditSeverity.HIGH,
        **kwargs
    ):
        """Log security-related events"""
        event = AuditEvent(
            event_type=AuditEventType.SECURITY_ALERT,
            severity=severity,
            actor_type="unknown",
            actor_id="unknown",
            actor_ip=actor_ip,
            action=f"security_{security_event_type}",
            description=description,
            event_data={"security_event_type": security_event_type, **kwargs},
            compliance_relevant=True
        )
        return asyncio.create_task(self.log_event(event))
    
    def log_client_activity(
        self,
        client_id: UUID,
        activity_type: str,
        details: Dict[str, Any],
        actor_id: Optional[str] = None
    ):
        """Log client-specific activities"""
        event = AuditEvent(
            event_type=AuditEventType.CLIENT_UPDATED if "update" in activity_type 
                      else AuditEventType.CLIENT_CREATED,
            severity=AuditSeverity.MEDIUM,
            actor_type="admin" if actor_id else "system",
            actor_id=actor_id or "system",
            resource_type="client",
            resource_id=str(client_id),
            client_id=client_id,
            action=activity_type,
            description=f"Client activity: {activity_type}",
            event_data=details,
            compliance_relevant=True
        )
        return asyncio.create_task(self.log_event(event))
    
    async def _flush_buffer(self):
        """Flush buffered events to database"""
        if not self.event_buffer:
            return
        
        try:
            db = next(get_db())
            events_to_process = self.event_buffer.copy()
            self.event_buffer.clear()
            
            for event in events_to_process:
                await self.log_event(event, db)
            
            db.close()
        except Exception as e:
            self.logger.error("Failed to flush audit event buffer", error=str(e))
    
    def _generate_event_id(self, event: AuditEvent) -> str:
        """Generate unique event ID"""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        event_type_short = event.event_type.value[:10]
        return f"{event_type_short}_{timestamp}"
    
    def _calculate_event_hash(self, audit_record: AuditLog) -> str:
        """Calculate SHA-256 hash for event integrity"""
        hash_data = {
            "event_type": audit_record.event_type,
            "timestamp": audit_record.timestamp.isoformat(),
            "actor_id": audit_record.actor_id,
            "resource_id": audit_record.resource_id,
            "action": audit_record.action,
            "event_data": audit_record.event_data
        }
        
        hash_string = json.dumps(hash_data, sort_keys=True, default=str)
        return hashlib.sha256(f"{hash_string}{settings.secret_key}".encode()).hexdigest()
    
    async def search_audit_logs(
        self,
        db: Session,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_types: Optional[List[str]] = None,
        actor_id: Optional[str] = None,
        client_id: Optional[UUID] = None,
        severity: Optional[str] = None,
        limit: int = 1000
    ) -> List[AuditLog]:
        """Search audit logs with filters"""
        query = db.query(AuditLog)
        
        if start_date:
            query = query.filter(AuditLog.timestamp >= start_date)
        if end_date:
            query = query.filter(AuditLog.timestamp <= end_date)
        if event_types:
            query = query.filter(AuditLog.event_type.in_(event_types))
        if actor_id:
            query = query.filter(AuditLog.actor_id == actor_id)
        if client_id:
            query = query.filter(AuditLog.client_id == client_id)
        if severity:
            query = query.filter(AuditLog.severity == severity)
        
        return query.order_by(AuditLog.timestamp.desc()).limit(limit).all()
    
    def generate_audit_report(
        self,
        db: Session,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate comprehensive audit report"""
        from sqlalchemy import func
        
        # Event type distribution
        event_type_stats = db.query(
            AuditLog.event_type,
            func.count(AuditLog.id).label('count')
        ).filter(
            AuditLog.timestamp.between(start_date, end_date)
        ).group_by(AuditLog.event_type).all()
        
        # Severity distribution
        severity_stats = db.query(
            AuditLog.severity,
            func.count(AuditLog.id).label('count')
        ).filter(
            AuditLog.timestamp.between(start_date, end_date)
        ).group_by(AuditLog.severity).all()
        
        # Top actors
        actor_stats = db.query(
            AuditLog.actor_id,
            func.count(AuditLog.id).label('count')
        ).filter(
            AuditLog.timestamp.between(start_date, end_date)
        ).group_by(AuditLog.actor_id).order_by(
            func.count(AuditLog.id).desc()
        ).limit(10).all()
        
        # Compliance events
        compliance_events = db.query(func.count(AuditLog.id)).filter(
            AuditLog.timestamp.between(start_date, end_date),
            AuditLog.compliance_relevant == True
        ).scalar()
        
        return {
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "summary": {
                "total_events": sum(stat.count for stat in event_type_stats),
                "compliance_events": compliance_events,
                "unique_actors": len(actor_stats)
            },
            "event_types": {stat.event_type: stat.count for stat in event_type_stats},
            "severity_distribution": {stat.severity: stat.count for stat in severity_stats},
            "top_actors": [{"actor_id": stat.actor_id, "event_count": stat.count} 
                          for stat in actor_stats]
        }

# Global audit logger instance
audit_logger = AuditLogger()

def get_audit_logger() -> AuditLogger:
    """Get audit logger instance"""
    return audit_logger

# Context manager for audit logging
@contextmanager
def audit_context(correlation_id: str, session_id: Optional[str] = None):
    """Context manager for audit correlation"""
    # This would be implemented with contextvars in production
    # to maintain audit context across async operations
    try:
        yield {
            "correlation_id": correlation_id,
            "session_id": session_id
        }
    finally:
        pass