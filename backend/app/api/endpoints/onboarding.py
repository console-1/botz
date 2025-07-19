"""
Client onboarding API endpoints
"""

import secrets
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field, EmailStr, validator

from app.core.database import get_db
from app.models.clients import ClientInvitation, Client, ClientTier, ClientStatus
from app.services.auth_service import AuthService, get_auth_service
from app.core.config import settings

router = APIRouter()

# Pydantic models
class OnboardingRequest(BaseModel):
    """Initial onboarding request from potential client"""
    company_name: str = Field(..., min_length=1, max_length=200)
    email: EmailStr
    contact_name: str = Field(..., min_length=1, max_length=100)
    phone: Optional[str] = Field(None, max_length=20)
    website: Optional[str] = Field(None, max_length=255)
    use_case: Optional[str] = Field(None, max_length=500)
    expected_volume: Optional[str] = None  # low, medium, high
    
    @validator('website')
    def validate_website(cls, v):
        if v and not (v.startswith('http://') or v.startswith('https://')):
            v = f"https://{v}"
        return v

class InvitationCreateRequest(BaseModel):
    """Admin request to create invitation"""
    email: EmailStr
    client_name: str = Field(..., min_length=1, max_length=200)
    client_id: str = Field(..., min_length=3, max_length=50)
    invited_by: str = Field(..., max_length=255)
    initial_tier: ClientTier = ClientTier.TRIAL
    expires_hours: int = Field(168, ge=1, le=720)  # Default 7 days, max 30 days
    initial_configuration: Optional[Dict[str, Any]] = None

class InvitationAcceptRequest(BaseModel):
    """Request to accept invitation and complete onboarding"""
    token: str
    contact_name: str = Field(..., min_length=1, max_length=100)
    phone: Optional[str] = Field(None, max_length=20)
    website: Optional[str] = Field(None, max_length=255)
    password: str = Field(..., min_length=8, max_length=128)
    company_configuration: Optional[Dict[str, Any]] = None

class OnboardingResponse(BaseModel):
    """Response to onboarding request"""
    status: str
    message: str
    request_id: Optional[str] = None

class InvitationResponse(BaseModel):
    """Response to invitation creation"""
    invitation_id: str
    token: str
    email: str
    client_id: str
    expires_at: datetime
    invitation_url: str

class ClientSetupResponse(BaseModel):
    """Response after successful client setup"""
    client_id: str
    api_key: str
    dashboard_url: str
    widget_embed_code: str
    next_steps: List[str]

async def send_onboarding_email(email: str, company_name: str, request_id: str):
    """Send onboarding confirmation email (placeholder)"""
    # In production, integrate with email service
    print(f"Sending onboarding email to {email} for {company_name} (ID: {request_id})")

async def send_invitation_email(email: str, invitation_url: str, client_name: str):
    """Send invitation email (placeholder)"""
    # In production, integrate with email service
    print(f"Sending invitation to {email} for {client_name}: {invitation_url}")

@router.post("/request", response_model=OnboardingResponse)
async def submit_onboarding_request(
    request: OnboardingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Submit initial onboarding request
    This creates a lead that admins can review and convert to invitations
    """
    
    # Generate unique request ID
    request_id = f"onb_{secrets.token_hex(8)}"
    
    # Store onboarding request (you might want a separate table for this)
    # For now, we'll just log it and send confirmation email
    
    # Send confirmation email
    background_tasks.add_task(
        send_onboarding_email,
        request.email,
        request.company_name,
        request_id
    )
    
    return OnboardingResponse(
        status="submitted",
        message="Thank you for your interest! We'll review your request and get back to you within 24 hours.",
        request_id=request_id
    )

@router.post("/invitations", response_model=InvitationResponse)
async def create_invitation(
    request: InvitationCreateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Create an invitation for a new client (Admin only)
    """
    
    # Check if client_id already exists
    existing_client = auth_service.get_client_by_id(db, request.client_id)
    if existing_client:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Client ID '{request.client_id}' already exists"
        )
    
    # Check if there's already a pending invitation for this email
    existing_invitation = db.query(ClientInvitation).filter(
        ClientInvitation.email == request.email,
        ClientInvitation.status == "pending"
    ).first()
    
    if existing_invitation and not existing_invitation.is_expired():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="There is already a pending invitation for this email"
        )
    
    # Generate secure token
    token = secrets.token_urlsafe(32)
    
    # Create invitation
    invitation = ClientInvitation(
        email=request.email,
        client_name=request.client_name,
        client_id=request.client_id,
        invited_by=request.invited_by,
        token=token,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=request.expires_hours),
        initial_configuration=request.initial_configuration or {},
        initial_tier=request.initial_tier
    )
    
    db.add(invitation)
    db.commit()
    db.refresh(invitation)
    
    # Generate invitation URL
    invitation_url = f"{settings.frontend_url}/onboarding/accept?token={token}"
    
    # Send invitation email
    background_tasks.add_task(
        send_invitation_email,
        request.email,
        invitation_url,
        request.client_name
    )
    
    return InvitationResponse(
        invitation_id=str(invitation.id),
        token=token,
        email=request.email,
        client_id=request.client_id,
        expires_at=invitation.expires_at,
        invitation_url=invitation_url
    )

@router.get("/invitations/{token}")
async def get_invitation_details(
    token: str,
    db: Session = Depends(get_db)
):
    """
    Get invitation details by token (for the acceptance form)
    """
    
    invitation = db.query(ClientInvitation).filter(
        ClientInvitation.token == token
    ).first()
    
    if not invitation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invitation not found"
        )
    
    if not invitation.is_valid():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invitation has expired or is no longer valid"
        )
    
    return {
        "client_name": invitation.client_name,
        "client_id": invitation.client_id,
        "email": invitation.email,
        "expires_at": invitation.expires_at,
        "initial_tier": invitation.initial_tier
    }

@router.post("/accept", response_model=ClientSetupResponse)
async def accept_invitation(
    request: InvitationAcceptRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Accept invitation and complete client onboarding
    """
    
    # Find and validate invitation
    invitation = db.query(ClientInvitation).filter(
        ClientInvitation.token == request.token
    ).first()
    
    if not invitation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invitation not found"
        )
    
    if not invitation.is_valid():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invitation has expired or is no longer valid"
        )
    
    # Check if client already exists
    existing_client = auth_service.get_client_by_id(db, invitation.client_id)
    if existing_client:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Client already exists"
        )
    
    try:
        # Create client
        client_config = invitation.initial_configuration.copy()
        if request.company_configuration:
            client_config.update(request.company_configuration)
        
        client = auth_service.create_client(
            db=db,
            client_id=invitation.client_id,
            name=invitation.client_name,
            email=invitation.email,
            tier=invitation.initial_tier,
            configuration=client_config
        )
        
        # Update additional client info
        client.contact_name = request.contact_name
        client.phone = request.phone
        client.website = request.website
        
        # Create initial API key
        api_key, full_api_key = auth_service.create_api_key(
            db=db,
            client_id=client.id,
            name="Default API Key",
            description="Initial API key created during onboarding",
            scopes=["chat:send", "chat:receive", "knowledge:read", "knowledge:write"]
        )
        
        # Mark invitation as accepted
        invitation.status = "accepted"
        invitation.accepted_at = datetime.now(timezone.utc)
        
        db.commit()
        
        # Generate widget embed code
        widget_embed_code = f'''<script 
  src="{settings.cdn_url}/chat-widget/embed.js"
  data-client-id="{client.client_id}"
  data-api-url="{settings.api_url}"
  data-company-name="{client.name}"
  data-primary-color="#007bff"
  data-welcome-message="Hi! How can we help you today?"
  async
></script>'''
        
        # Generate dashboard URL
        dashboard_url = f"{settings.frontend_url}/dashboard?client={client.client_id}"
        
        return ClientSetupResponse(
            client_id=client.client_id,
            api_key=full_api_key,
            dashboard_url=dashboard_url,
            widget_embed_code=widget_embed_code,
            next_steps=[
                "Add the widget embed code to your website",
                "Upload your knowledge base documents",
                "Configure your bot's tone and branding",
                "Test the chat widget functionality",
                "Set up escalation rules and workflows"
            ]
        )
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create client: {str(e)}"
        )

@router.get("/status/{client_id}")
async def get_onboarding_status(
    client_id: str,
    db: Session = Depends(get_db),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Get onboarding status for a client
    """
    
    client = auth_service.get_client_by_id(db, client_id)
    if not client:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Client not found"
        )
    
    # Check onboarding completion steps
    onboarding_steps = {
        "client_created": True,
        "api_key_generated": len(client.api_keys) > 0,
        "knowledge_base_uploaded": len(client.knowledge_bases) > 0,
        "widget_configured": bool(client.configuration.get("branding")),
        "first_conversation": len(client.conversations) > 0
    }
    
    completion_percentage = (sum(onboarding_steps.values()) / len(onboarding_steps)) * 100
    
    return {
        "client_id": client_id,
        "status": client.status,
        "tier": client.tier,
        "onboarding_steps": onboarding_steps,
        "completion_percentage": completion_percentage,
        "is_trial_expired": client.is_trial_expired(),
        "trial_ends_at": client.trial_ends_at,
        "created_at": client.created_at
    }

@router.post("/complete/{client_id}")
async def complete_onboarding(
    client_id: str,
    db: Session = Depends(get_db),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Mark onboarding as complete and activate client
    """
    
    client = auth_service.get_client_by_id(db, client_id)
    if not client:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Client not found"
        )
    
    # Check if basic requirements are met
    if not client.api_keys:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Client must have at least one API key"
        )
    
    # Activate client if still in trial
    if client.status == ClientStatus.TRIAL:
        client.status = ClientStatus.ACTIVE
        db.commit()
    
    return {
        "status": "completed",
        "message": "Onboarding completed successfully",
        "client_status": client.status
    }