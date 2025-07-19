"""
GDPR compliance service for data protection and privacy
"""

from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Set
from uuid import UUID
from enum import Enum
import json
import hashlib

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, text
from pydantic import BaseModel, Field

from app.core.database import get_db
from app.models.client import Client
from app.models.conversation import Conversation, Message
from app.core.config import settings

class DataCategory(str, Enum):
    """Categories of personal data as defined by GDPR"""
    PERSONAL_IDENTIFIERS = "personal_identifiers"  # Names, emails, IDs
    CONTACT_INFORMATION = "contact_information"   # Phone, address
    CONVERSATION_DATA = "conversation_data"       # Chat messages, history
    USAGE_ANALYTICS = "usage_analytics"          # Behavior, preferences
    TECHNICAL_DATA = "technical_data"            # IP addresses, device info
    BIOMETRIC_DATA = "biometric_data"            # Voice, behavioral patterns

class LegalBasis(str, Enum):
    """Legal basis for data processing under GDPR Article 6"""
    CONSENT = "consent"                          # Article 6(1)(a)
    CONTRACT = "contract"                        # Article 6(1)(b)
    LEGAL_OBLIGATION = "legal_obligation"        # Article 6(1)(c)
    VITAL_INTERESTS = "vital_interests"          # Article 6(1)(d)
    PUBLIC_TASK = "public_task"                  # Article 6(1)(e)
    LEGITIMATE_INTERESTS = "legitimate_interests" # Article 6(1)(f)

class DataRetentionPeriod(str, Enum):
    """Standard data retention periods"""
    IMMEDIATE = "immediate"                      # Delete immediately
    THIRTY_DAYS = "30_days"                     # 30 days
    ONE_YEAR = "1_year"                         # 1 year
    THREE_YEARS = "3_years"                     # 3 years
    SEVEN_YEARS = "7_years"                     # 7 years (legal requirement)
    INDEFINITE = "indefinite"                   # Keep indefinitely (with consent)

class DataProcessingRecord(BaseModel):
    """Record of data processing activity for GDPR compliance"""
    processing_id: str
    data_subject_id: str
    data_categories: List[DataCategory]
    legal_basis: LegalBasis
    purpose: str
    processing_date: datetime
    retention_period: DataRetentionPeriod
    deletion_date: Optional[datetime]
    consent_given: bool
    consent_date: Optional[datetime]
    metadata: Dict[str, Any] = {}

class DataSubjectRequest(BaseModel):
    """GDPR data subject request"""
    request_id: str
    data_subject_id: str
    request_type: str  # access, rectification, erasure, portability, restriction
    status: str       # pending, in_progress, completed, rejected
    submitted_date: datetime
    completed_date: Optional[datetime]
    verification_method: str
    details: Dict[str, Any] = {}

class PersonalDataInventory(BaseModel):
    """Inventory of personal data held for a data subject"""
    data_subject_id: str
    data_categories: Dict[DataCategory, List[Dict[str, Any]]]
    total_records: int
    earliest_record: datetime
    latest_record: datetime
    retention_schedule: Dict[DataCategory, datetime]

class GDPRComplianceService:
    """
    Service for GDPR compliance including data protection, subject rights, and audit trails
    """
    
    def __init__(self):
        self.retention_periods = {
            DataCategory.CONVERSATION_DATA: DataRetentionPeriod.ONE_YEAR,
            DataCategory.USAGE_ANALYTICS: DataRetentionPeriod.THREE_YEARS,
            DataCategory.PERSONAL_IDENTIFIERS: DataRetentionPeriod.SEVEN_YEARS,
            DataCategory.CONTACT_INFORMATION: DataRetentionPeriod.SEVEN_YEARS,
            DataCategory.TECHNICAL_DATA: DataRetentionPeriod.THIRTY_DAYS,
            DataCategory.BIOMETRIC_DATA: DataRetentionPeriod.ONE_YEAR
        }
        
        self.pii_fields = {
            "email", "phone", "name", "address", "ip_address", 
            "user_id", "session_id", "device_id"
        }
    
    def anonymize_pii(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anonymize personally identifiable information in data
        """
        anonymized = data.copy()
        
        for field in self.pii_fields:
            if field in anonymized:
                if isinstance(anonymized[field], str):
                    # Hash the value for anonymization
                    hash_value = hashlib.sha256(
                        f"{anonymized[field]}{settings.secret_key}".encode()
                    ).hexdigest()[:12]
                    anonymized[field] = f"anon_{hash_value}"
                else:
                    anonymized[field] = None
        
        return anonymized
    
    def pseudonymize_data(self, data: Dict[str, Any], data_subject_id: str) -> Dict[str, Any]:
        """
        Pseudonymize data while maintaining referential integrity
        """
        pseudonymized = data.copy()
        
        # Create consistent pseudonym for the data subject
        pseudonym = hashlib.sha256(
            f"{data_subject_id}{settings.secret_key}".encode()
        ).hexdigest()[:16]
        
        # Replace direct identifiers with pseudonym
        for field in ["user_id", "email", "name"]:
            if field in pseudonymized:
                pseudonymized[field] = f"pseudo_{pseudonym}"
        
        return pseudonymized
    
    def create_personal_data_inventory(
        self, 
        db: Session, 
        data_subject_id: str
    ) -> PersonalDataInventory:
        """
        Create comprehensive inventory of personal data for a data subject
        """
        inventory_data = {}
        total_records = 0
        earliest_record = None
        latest_record = None
        
        # Conversations and messages
        conversations = db.query(Conversation).filter(
            or_(
                Conversation.user_id == data_subject_id,
                Conversation.session_id == data_subject_id
            )
        ).all()
        
        conversation_records = []
        for conv in conversations:
            messages = db.query(Message).filter(
                Message.conversation_id == conv.conversation_id
            ).all()
            
            conversation_records.append({
                "conversation_id": conv.conversation_id,
                "created_at": conv.created_at,
                "message_count": len(messages),
                "status": conv.status,
                "metadata": conv.metadata
            })
            
            # Track date ranges
            if earliest_record is None or conv.created_at < earliest_record:
                earliest_record = conv.created_at
            if latest_record is None or conv.created_at > latest_record:
                latest_record = conv.created_at
        
        inventory_data[DataCategory.CONVERSATION_DATA] = conversation_records
        total_records += len(conversation_records)
        
        # Calculate retention schedule
        retention_schedule = {}
        for category, period in self.retention_periods.items():
            retention_date = self._calculate_retention_date(period, latest_record or datetime.now(timezone.utc))
            retention_schedule[category] = retention_date
        
        return PersonalDataInventory(
            data_subject_id=data_subject_id,
            data_categories=inventory_data,
            total_records=total_records,
            earliest_record=earliest_record or datetime.now(timezone.utc),
            latest_record=latest_record or datetime.now(timezone.utc),
            retention_schedule=retention_schedule
        )
    
    def process_data_subject_access_request(
        self, 
        db: Session, 
        data_subject_id: str,
        verification_token: str
    ) -> Dict[str, Any]:
        """
        Process GDPR Article 15 - Right of access request
        """
        # Verify the request (simplified - in production, implement proper verification)
        if not self._verify_data_subject_identity(data_subject_id, verification_token):
            raise ValueError("Identity verification failed")
        
        inventory = self.create_personal_data_inventory(db, data_subject_id)
        
        # Collect all personal data
        personal_data = {
            "data_subject_id": data_subject_id,
            "request_date": datetime.now(timezone.utc),
            "data_categories": {},
            "processing_purposes": {},
            "retention_periods": {},
            "third_party_recipients": [],
            "data_source": "Customer service chat interactions"
        }
        
        # Add conversation data
        for category, records in inventory.data_categories.items():
            personal_data["data_categories"][category.value] = records
            personal_data["processing_purposes"][category.value] = self._get_processing_purpose(category)
            personal_data["retention_periods"][category.value] = str(inventory.retention_schedule.get(category))
        
        return personal_data
    
    def process_data_erasure_request(
        self,
        db: Session,
        data_subject_id: str,
        verification_token: str,
        erasure_reason: str = "withdrawal_of_consent"
    ) -> Dict[str, Any]:
        """
        Process GDPR Article 17 - Right to erasure (Right to be forgotten)
        """
        if not self._verify_data_subject_identity(data_subject_id, verification_token):
            raise ValueError("Identity verification failed")
        
        erasure_log = {
            "data_subject_id": data_subject_id,
            "erasure_date": datetime.now(timezone.utc),
            "reason": erasure_reason,
            "records_deleted": {},
            "anonymized_records": {}
        }
        
        # Delete conversations and messages
        conversations = db.query(Conversation).filter(
            or_(
                Conversation.user_id == data_subject_id,
                Conversation.session_id == data_subject_id
            )
        ).all()
        
        conversations_deleted = 0
        messages_deleted = 0
        
        for conv in conversations:
            # Delete messages first
            messages = db.query(Message).filter(
                Message.conversation_id == conv.conversation_id
            ).all()
            
            for message in messages:
                db.delete(message)
                messages_deleted += 1
            
            # Delete conversation
            db.delete(conv)
            conversations_deleted += 1
        
        erasure_log["records_deleted"]["conversations"] = conversations_deleted
        erasure_log["records_deleted"]["messages"] = messages_deleted
        
        db.commit()
        
        return erasure_log
    
    def process_data_portability_request(
        self,
        db: Session,
        data_subject_id: str,
        verification_token: str,
        export_format: str = "json"
    ) -> Dict[str, Any]:
        """
        Process GDPR Article 20 - Right to data portability
        """
        if not self._verify_data_subject_identity(data_subject_id, verification_token):
            raise ValueError("Identity verification failed")
        
        # Get personal data in structured format
        personal_data = self.process_data_subject_access_request(db, data_subject_id, verification_token)
        
        # Format for portability
        portable_data = {
            "export_date": datetime.now(timezone.utc).isoformat(),
            "data_subject_id": data_subject_id,
            "format": export_format,
            "data": personal_data,
            "schema_version": "1.0",
            "export_method": "gdpr_portability_request"
        }
        
        return portable_data
    
    def apply_data_retention_policies(self, db: Session) -> Dict[str, Any]:
        """
        Apply data retention policies and delete expired data
        """
        now = datetime.now(timezone.utc)
        deletion_log = {
            "execution_date": now,
            "policies_applied": {},
            "records_processed": 0,
            "records_deleted": 0,
            "errors": []
        }
        
        try:
            # Delete expired conversations (older than retention period)
            retention_date = now - timedelta(days=365)  # 1 year retention
            
            expired_conversations = db.query(Conversation).filter(
                Conversation.created_at < retention_date
            ).all()
            
            conversations_deleted = 0
            messages_deleted = 0
            
            for conv in expired_conversations:
                # Delete associated messages
                messages = db.query(Message).filter(
                    Message.conversation_id == conv.conversation_id
                ).all()
                
                for message in messages:
                    db.delete(message)
                    messages_deleted += 1
                
                # Delete conversation
                db.delete(conv)
                conversations_deleted += 1
            
            deletion_log["policies_applied"]["conversation_retention"] = {
                "retention_period_days": 365,
                "cutoff_date": retention_date.isoformat(),
                "conversations_deleted": conversations_deleted,
                "messages_deleted": messages_deleted
            }
            
            deletion_log["records_deleted"] = conversations_deleted + messages_deleted
            
            db.commit()
            
        except Exception as e:
            deletion_log["errors"].append(f"Error applying retention policies: {str(e)}")
            db.rollback()
        
        return deletion_log
    
    def _verify_data_subject_identity(self, data_subject_id: str, verification_token: str) -> bool:
        """
        Verify the identity of the data subject making the request
        In production, this would implement proper identity verification
        """
        # Simplified verification - in production, implement:
        # - Email verification
        # - Multi-factor authentication
        # - Identity document verification
        # - Biometric verification
        
        expected_token = hashlib.sha256(
            f"{data_subject_id}{settings.secret_key}".encode()
        ).hexdigest()[:16]
        
        return verification_token == expected_token
    
    def _get_processing_purpose(self, category: DataCategory) -> str:
        """Get the processing purpose for a data category"""
        purposes = {
            DataCategory.CONVERSATION_DATA: "Providing customer service and support",
            DataCategory.USAGE_ANALYTICS: "Service improvement and analytics",
            DataCategory.PERSONAL_IDENTIFIERS: "User identification and account management",
            DataCategory.CONTACT_INFORMATION: "Communication and service delivery",
            DataCategory.TECHNICAL_DATA: "System security and performance monitoring",
            DataCategory.BIOMETRIC_DATA: "Voice interaction and authentication"
        }
        return purposes.get(category, "General business operations")
    
    def _calculate_retention_date(self, period: DataRetentionPeriod, reference_date: datetime) -> datetime:
        """Calculate the retention date based on the retention period"""
        if period == DataRetentionPeriod.IMMEDIATE:
            return reference_date
        elif period == DataRetentionPeriod.THIRTY_DAYS:
            return reference_date + timedelta(days=30)
        elif period == DataRetentionPeriod.ONE_YEAR:
            return reference_date + timedelta(days=365)
        elif period == DataRetentionPeriod.THREE_YEARS:
            return reference_date + timedelta(days=1095)
        elif period == DataRetentionPeriod.SEVEN_YEARS:
            return reference_date + timedelta(days=2555)
        else:  # INDEFINITE
            return reference_date + timedelta(days=36500)  # 100 years
    
    def generate_privacy_notice(self, client: Client) -> Dict[str, Any]:
        """
        Generate GDPR-compliant privacy notice for a client
        """
        return {
            "data_controller": {
                "name": client.name,
                "contact": client.email,
                "dpo_contact": client.configuration.get("dpo_email", client.email)
            },
            "data_categories_collected": [
                {
                    "category": "Personal identifiers",
                    "examples": "Name, email address, user ID",
                    "legal_basis": "Legitimate interests",
                    "purpose": "User identification and service provision"
                },
                {
                    "category": "Conversation data",
                    "examples": "Chat messages, queries, responses",
                    "legal_basis": "Legitimate interests",
                    "purpose": "Providing customer service and support"
                },
                {
                    "category": "Usage analytics",
                    "examples": "Response times, satisfaction scores",
                    "legal_basis": "Legitimate interests", 
                    "purpose": "Service improvement and quality assurance"
                }
            ],
            "retention_periods": {
                "Conversation data": "1 year from last interaction",
                "Usage analytics": "3 years for improvement purposes",
                "Contact information": "7 years for legal compliance"
            },
            "data_subject_rights": [
                "Right of access (Article 15)",
                "Right to rectification (Article 16)", 
                "Right to erasure (Article 17)",
                "Right to data portability (Article 20)",
                "Right to object (Article 21)"
            ],
            "third_party_sharing": [
                "LLM providers (OpenAI, Anthropic, etc.) for response generation",
                "Cloud infrastructure providers for data hosting",
                "Analytics services for performance monitoring"
            ],
            "data_protection_measures": [
                "End-to-end encryption in transit (TLS 1.3)",
                "Encryption at rest (AES-256)",
                "Access controls and authentication",
                "Regular security audits and monitoring",
                "Data minimization and pseudonymization"
            ]
        }

# Dependency injection
def get_gdpr_service() -> GDPRComplianceService:
    """Get GDPR compliance service instance"""
    return GDPRComplianceService()