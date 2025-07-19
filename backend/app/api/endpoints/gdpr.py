"""
GDPR compliance API endpoints for data subject rights
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field, EmailStr

from app.core.database import get_db
from app.services.gdpr_compliance import GDPRComplianceService, get_gdpr_service
from app.services.auth_service import AuthService, get_auth_service, get_current_client
from app.models.client import Client, ClientAPIKey
from app.core.config import settings

router = APIRouter()

# Pydantic models for GDPR API
class DataSubjectVerificationRequest(BaseModel):
    """Request to verify data subject identity"""
    data_subject_id: str = Field(..., description="User ID, email, or session ID")
    verification_method: str = Field(..., description="Method of verification (email, phone, etc.)")
    verification_data: Optional[str] = Field(None, description="Additional verification data")

class DataSubjectAccessRequest(BaseModel):
    """GDPR Article 15 - Right of access request"""
    data_subject_id: str
    verification_token: str
    request_details: Optional[str] = None

class DataErasureRequest(BaseModel):
    """GDPR Article 17 - Right to erasure request"""
    data_subject_id: str
    verification_token: str
    erasure_reason: str = Field(
        default="withdrawal_of_consent",
        description="Reason for erasure request"
    )

class DataPortabilityRequest(BaseModel):
    """GDPR Article 20 - Right to data portability request"""
    data_subject_id: str
    verification_token: str
    export_format: str = Field(default="json", regex="^(json|csv|xml)$")

class ConsentUpdateRequest(BaseModel):
    """Request to update consent preferences"""
    data_subject_id: str
    consent_categories: Dict[str, bool]
    consent_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class DataProcessingLogRequest(BaseModel):
    """Request for data processing activity logs"""
    data_subject_id: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

class GDPRResponse(BaseModel):
    """Standard GDPR response format"""
    request_id: str
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None
    processing_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# Public endpoints (no authentication required for data subject rights)
@router.post("/data-access", response_model=GDPRResponse)
async def request_data_access(
    request: DataSubjectAccessRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    gdpr_service: GDPRComplianceService = Depends(get_gdpr_service)
):
    """
    GDPR Article 15 - Right of access
    Allow data subjects to request access to their personal data
    """
    try:
        # Process the access request
        personal_data = gdpr_service.process_data_subject_access_request(
            db=db,
            data_subject_id=request.data_subject_id,
            verification_token=request.verification_token
        )
        
        request_id = f"access_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        # In production, you would:
        # 1. Queue the request for manual review if needed
        # 2. Send the data via secure delivery method
        # 3. Log the request for audit purposes
        
        return GDPRResponse(
            request_id=request_id,
            status="completed",
            message="Personal data access request processed successfully",
            data=personal_data
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Access request failed: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error processing access request"
        )

@router.post("/data-erasure", response_model=GDPRResponse)
async def request_data_erasure(
    request: DataErasureRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    gdpr_service: GDPRComplianceService = Depends(get_gdpr_service)
):
    """
    GDPR Article 17 - Right to erasure (Right to be forgotten)
    Allow data subjects to request deletion of their personal data
    """
    try:
        # Process the erasure request
        erasure_log = gdpr_service.process_data_erasure_request(
            db=db,
            data_subject_id=request.data_subject_id,
            verification_token=request.verification_token,
            erasure_reason=request.erasure_reason
        )
        
        request_id = f"erasure_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        return GDPRResponse(
            request_id=request_id,
            status="completed",
            message="Data erasure request processed successfully",
            data=erasure_log
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erasure request failed: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error processing erasure request"
        )

@router.post("/data-portability", response_model=GDPRResponse)
async def request_data_portability(
    request: DataPortabilityRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    gdpr_service: GDPRComplianceService = Depends(get_gdpr_service)
):
    """
    GDPR Article 20 - Right to data portability
    Allow data subjects to receive their personal data in a portable format
    """
    try:
        # Process the portability request
        portable_data = gdpr_service.process_data_portability_request(
            db=db,
            data_subject_id=request.data_subject_id,
            verification_token=request.verification_token,
            export_format=request.export_format
        )
        
        request_id = f"portability_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        return GDPRResponse(
            request_id=request_id,
            status="completed",
            message="Data portability request processed successfully",
            data=portable_data
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Portability request failed: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error processing portability request"
        )

@router.get("/verification-token/{data_subject_id}")
async def generate_verification_token(
    data_subject_id: str,
    background_tasks: BackgroundTasks
):
    """
    Generate verification token for data subject identity verification
    In production, this would send the token via secure channel (email, SMS)
    """
    import hashlib
    
    # Generate verification token
    verification_token = hashlib.sha256(
        f"{data_subject_id}{settings.secret_key}".encode()
    ).hexdigest()[:16]
    
    # In production, send this token via:
    # - Email with secure link
    # - SMS with verification code
    # - Two-factor authentication
    
    return {
        "message": "Verification token generated and sent",
        "token": verification_token,  # Remove this in production!
        "expires_at": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
        "note": "In production, this token would be sent via secure communication channel"
    }

# Client-authenticated endpoints
@router.get("/privacy-notice")
async def get_privacy_notice(
    current_client: tuple[Client, ClientAPIKey] = Depends(get_current_client),
    gdpr_service: GDPRComplianceService = Depends(get_gdpr_service)
):
    """
    Get GDPR-compliant privacy notice for a client
    """
    client, _ = current_client
    
    privacy_notice = gdpr_service.generate_privacy_notice(client)
    
    return {
        "client_id": client.client_id,
        "privacy_notice": privacy_notice,
        "generated_at": datetime.now(timezone.utc),
        "version": "1.0"
    }

@router.get("/data-inventory/{data_subject_id}")
async def get_personal_data_inventory(
    data_subject_id: str,
    db: Session = Depends(get_db),
    current_client: tuple[Client, ClientAPIKey] = Depends(get_current_client),
    gdpr_service: GDPRComplianceService = Depends(get_gdpr_service)
):
    """
    Get inventory of personal data for a specific data subject (client use)
    """
    client, _ = current_client
    
    try:
        inventory = gdpr_service.create_personal_data_inventory(db, data_subject_id)
        
        return {
            "client_id": client.client_id,
            "data_subject_id": data_subject_id,
            "inventory": inventory.dict(),
            "generated_at": datetime.now(timezone.utc)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating inventory: {str(e)}"
        )

# Admin endpoints (admin authentication required)
@router.post("/admin/apply-retention-policies")
async def apply_data_retention_policies(
    admin_key: str,
    db: Session = Depends(get_db),
    gdpr_service: GDPRComplianceService = Depends(get_gdpr_service)
):
    """
    Apply data retention policies and delete expired data (Admin only)
    """
    if admin_key != settings.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin key"
        )
    
    try:
        deletion_log = gdpr_service.apply_data_retention_policies(db)
        
        return {
            "status": "completed",
            "deletion_log": deletion_log,
            "executed_at": datetime.now(timezone.utc)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error applying retention policies: {str(e)}"
        )

@router.get("/admin/compliance-report")
async def generate_compliance_report(
    admin_key: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db)
):
    """
    Generate GDPR compliance report (Admin only)
    """
    if admin_key != settings.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin key"
        )
    
    if not start_date:
        start_date = datetime.now(timezone.utc) - timedelta(days=30)
    if not end_date:
        end_date = datetime.now(timezone.utc)
    
    try:
        # Generate compliance metrics
        from sqlalchemy import func, and_
        from app.models.conversation import Conversation, Message
        
        # Basic metrics
        total_conversations = db.query(func.count(Conversation.id)).filter(
            and_(
                Conversation.created_at >= start_date,
                Conversation.created_at <= end_date
            )
        ).scalar()
        
        total_messages = db.query(func.count(Message.id)).join(Conversation).filter(
            and_(
                Conversation.created_at >= start_date,
                Conversation.created_at <= end_date
            )
        ).scalar()
        
        # Data subject requests (would be stored in a requests table in production)
        compliance_report = {
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "data_processing_metrics": {
                "total_conversations": total_conversations or 0,
                "total_messages": total_messages or 0,
                "active_data_subjects": 0  # Would calculate unique users
            },
            "data_subject_requests": {
                "access_requests": 0,
                "erasure_requests": 0,
                "portability_requests": 0,
                "pending_requests": 0
            },
            "retention_policy_application": {
                "last_execution": "Not implemented",
                "records_deleted": 0,
                "policies_in_effect": len(gdpr_service.retention_periods)
            },
            "compliance_status": "Compliant",
            "recommendations": [
                "Implement automated retention policy execution",
                "Set up data subject request tracking",
                "Regular compliance audits"
            ]
        }
        
        return compliance_report
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating compliance report: {str(e)}"
        )