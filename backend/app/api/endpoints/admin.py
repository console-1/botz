"""
Admin dashboard API endpoints for client management
"""

from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc
from pydantic import BaseModel, Field

from app.core.database import get_db
from app.models.client import Client, ClientAPIKey, UsageRecord, ClientStatus, ClientTier
from app.models.conversation import Conversation, Message, UsageMetric
from app.services.auth_service import AuthService, get_auth_service
from app.core.config import settings

router = APIRouter()

# Pydantic models for API
class ClientSummary(BaseModel):
    id: str
    client_id: str
    name: str
    email: str
    status: str
    tier: str
    created_at: datetime
    last_active_at: Optional[datetime]
    trial_ends_at: Optional[datetime]
    is_trial_expired: bool
    total_conversations: int
    total_messages: int
    api_keys_count: int

class ClientDetail(BaseModel):
    id: str
    client_id: str
    name: str
    email: str
    contact_name: Optional[str]
    phone: Optional[str]
    website: Optional[str]
    status: str
    tier: str
    configuration: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    last_active_at: Optional[datetime]
    trial_ends_at: Optional[datetime]
    is_trial_expired: bool

class ClientCreateRequest(BaseModel):
    client_id: str = Field(..., min_length=3, max_length=50)
    name: str = Field(..., min_length=1, max_length=200)
    email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
    contact_name: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    tier: ClientTier = ClientTier.TRIAL
    configuration: Optional[Dict[str, Any]] = None

class ClientUpdateRequest(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    contact_name: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    status: Optional[ClientStatus] = None
    tier: Optional[ClientTier] = None
    configuration: Optional[Dict[str, Any]] = None

class APIKeyCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    scopes: List[str] = ["chat:send", "chat:receive", "knowledge:read"]
    rate_limit: Optional[int] = 1000
    expires_days: Optional[int] = None

class APIKeyResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    key_prefix: str
    scopes: List[str]
    rate_limit: int
    is_active: bool
    created_at: datetime
    expires_at: Optional[datetime]
    last_used_at: Optional[datetime]

class UsageStats(BaseModel):
    total_clients: int
    active_clients: int
    trial_clients: int
    total_conversations: int
    total_messages: int
    total_api_calls: int
    avg_response_time_ms: float
    escalation_rate: float

class ClientUsageStats(BaseModel):
    client_id: str
    period_start: datetime
    period_end: datetime
    conversations: int
    messages: int
    tokens_used: int
    api_calls: int
    avg_response_time_ms: int
    escalations: int
    estimated_cost_usd: float

# Admin authentication (for now, just a simple admin key check)
def verify_admin_key(admin_key: str = Query(..., alias="admin_key")):
    """Verify admin API key"""
    if admin_key != settings.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin key"
        )
    return True

@router.get("/stats", response_model=UsageStats)
async def get_usage_stats(
    db: Session = Depends(get_db),
    _: bool = Depends(verify_admin_key)
):
    """Get overall usage statistics"""
    
    # Client statistics
    total_clients = db.query(func.count(Client.id)).scalar()
    active_clients = db.query(func.count(Client.id)).filter(
        Client.status == ClientStatus.ACTIVE
    ).scalar()
    trial_clients = db.query(func.count(Client.id)).filter(
        Client.status == ClientStatus.TRIAL
    ).scalar()
    
    # Conversation statistics
    total_conversations = db.query(func.count(Conversation.id)).scalar()
    total_messages = db.query(func.count(Message.id)).scalar()
    
    # API call statistics (from usage records)
    api_calls_result = db.query(func.sum(UsageRecord.api_calls_made)).scalar()
    total_api_calls = api_calls_result or 0
    
    # Performance statistics
    avg_response_time = db.query(func.avg(Message.processing_time_ms)).scalar()
    avg_response_time_ms = float(avg_response_time) if avg_response_time else 0.0
    
    # Escalation rate
    total_escalations = db.query(func.count(Conversation.id)).filter(
        Conversation.is_escalated == True
    ).scalar()
    escalation_rate = (total_escalations / total_conversations) if total_conversations > 0 else 0.0
    
    return UsageStats(
        total_clients=total_clients,
        active_clients=active_clients,
        trial_clients=trial_clients,
        total_conversations=total_conversations,
        total_messages=total_messages,
        total_api_calls=total_api_calls,
        avg_response_time_ms=avg_response_time_ms,
        escalation_rate=escalation_rate
    )

@router.get("/clients", response_model=List[ClientSummary])
async def list_clients(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[ClientStatus] = None,
    tier: Optional[ClientTier] = None,
    search: Optional[str] = None,
    db: Session = Depends(get_db),
    _: bool = Depends(verify_admin_key)
):
    """List all clients with filtering and pagination"""
    
    query = db.query(Client)
    
    # Apply filters
    if status:
        query = query.filter(Client.status == status)
    if tier:
        query = query.filter(Client.tier == tier)
    if search:
        search_term = f"%{search}%"
        query = query.filter(
            or_(
                Client.name.ilike(search_term),
                Client.email.ilike(search_term),
                Client.client_id.ilike(search_term)
            )
        )
    
    # Get clients with pagination
    clients = query.order_by(desc(Client.created_at)).offset(skip).limit(limit).all()
    
    # Build response with aggregated data
    client_summaries = []
    for client in clients:
        # Count conversations and messages
        total_conversations = db.query(func.count(Conversation.id)).filter(
            Conversation.client_id == client.id
        ).scalar()
        
        total_messages = db.query(func.count(Message.id)).join(Conversation).filter(
            Conversation.client_id == client.id
        ).scalar()
        
        # Count API keys
        api_keys_count = db.query(func.count(ClientAPIKey.id)).filter(
            ClientAPIKey.client_id == client.id
        ).scalar()
        
        client_summaries.append(ClientSummary(
            id=str(client.id),
            client_id=client.client_id,
            name=client.name,
            email=client.email,
            status=client.status,
            tier=client.tier,
            created_at=client.created_at,
            last_active_at=client.last_active_at,
            trial_ends_at=client.trial_ends_at,
            is_trial_expired=client.is_trial_expired(),
            total_conversations=total_conversations or 0,
            total_messages=total_messages or 0,
            api_keys_count=api_keys_count or 0
        ))
    
    return client_summaries

@router.get("/clients/{client_id}", response_model=ClientDetail)
async def get_client(
    client_id: str,
    db: Session = Depends(get_db),
    _: bool = Depends(verify_admin_key)
):
    """Get detailed client information"""
    
    client = db.query(Client).filter(Client.client_id == client_id).first()
    if not client:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Client not found"
        )
    
    return ClientDetail(
        id=str(client.id),
        client_id=client.client_id,
        name=client.name,
        email=client.email,
        contact_name=client.contact_name,
        phone=client.phone,
        website=client.website,
        status=client.status,
        tier=client.tier,
        configuration=client.configuration,
        created_at=client.created_at,
        updated_at=client.updated_at,
        last_active_at=client.last_active_at,
        trial_ends_at=client.trial_ends_at,
        is_trial_expired=client.is_trial_expired()
    )

@router.post("/clients", response_model=ClientDetail)
async def create_client(
    request: ClientCreateRequest,
    db: Session = Depends(get_db),
    auth_service: AuthService = Depends(get_auth_service),
    _: bool = Depends(verify_admin_key)
):
    """Create a new client"""
    
    try:
        client = auth_service.create_client(
            db=db,
            client_id=request.client_id,
            name=request.name,
            email=request.email,
            tier=request.tier,
            configuration=request.configuration
        )
        
        # Set additional fields
        if request.contact_name:
            client.contact_name = request.contact_name
        if request.phone:
            client.phone = request.phone
        if request.website:
            client.website = request.website
        
        db.commit()
        db.refresh(client)
        
        return ClientDetail(
            id=str(client.id),
            client_id=client.client_id,
            name=client.name,
            email=client.email,
            contact_name=client.contact_name,
            phone=client.phone,
            website=client.website,
            status=client.status,
            tier=client.tier,
            configuration=client.configuration,
            created_at=client.created_at,
            updated_at=client.updated_at,
            last_active_at=client.last_active_at,
            trial_ends_at=client.trial_ends_at,
            is_trial_expired=client.is_trial_expired()
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.put("/clients/{client_id}", response_model=ClientDetail)
async def update_client(
    client_id: str,
    request: ClientUpdateRequest,
    db: Session = Depends(get_db),
    _: bool = Depends(verify_admin_key)
):
    """Update client information"""
    
    client = db.query(Client).filter(Client.client_id == client_id).first()
    if not client:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Client not found"
        )
    
    # Update fields
    update_data = request.dict(exclude_unset=True)
    for field, value in update_data.items():
        if field == "configuration" and value:
            # Merge configuration
            client.configuration = {**client.configuration, **value}
        else:
            setattr(client, field, value)
    
    db.commit()
    db.refresh(client)
    
    return ClientDetail(
        id=str(client.id),
        client_id=client.client_id,
        name=client.name,
        email=client.email,
        contact_name=client.contact_name,
        phone=client.phone,
        website=client.website,
        status=client.status,
        tier=client.tier,
        configuration=client.configuration,
        created_at=client.created_at,
        updated_at=client.updated_at,
        last_active_at=client.last_active_at,
        trial_ends_at=client.trial_ends_at,
        is_trial_expired=client.is_trial_expired()
    )

@router.get("/clients/{client_id}/api-keys", response_model=List[APIKeyResponse])
async def list_client_api_keys(
    client_id: str,
    db: Session = Depends(get_db),
    auth_service: AuthService = Depends(get_auth_service),
    _: bool = Depends(verify_admin_key)
):
    """List all API keys for a client"""
    
    client = db.query(Client).filter(Client.client_id == client_id).first()
    if not client:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Client not found"
        )
    
    api_keys = auth_service.list_client_api_keys(db, client.id)
    
    return [
        APIKeyResponse(
            id=str(key.id),
            name=key.name,
            description=key.description,
            key_prefix=key.key_prefix,
            scopes=key.scopes,
            rate_limit=key.rate_limit,
            is_active=key.is_active,
            created_at=key.created_at,
            expires_at=key.expires_at,
            last_used_at=key.last_used_at
        )
        for key in api_keys
    ]

@router.post("/clients/{client_id}/api-keys")
async def create_client_api_key(
    client_id: str,
    request: APIKeyCreateRequest,
    db: Session = Depends(get_db),
    auth_service: AuthService = Depends(get_auth_service),
    _: bool = Depends(verify_admin_key)
):
    """Create a new API key for a client"""
    
    client = db.query(Client).filter(Client.client_id == client_id).first()
    if not client:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Client not found"
        )
    
    try:
        api_key, full_key = auth_service.create_api_key(
            db=db,
            client_id=client.id,
            name=request.name,
            description=request.description,
            scopes=request.scopes,
            rate_limit=request.rate_limit,
            expires_days=request.expires_days
        )
        
        return {
            "api_key": APIKeyResponse(
                id=str(api_key.id),
                name=api_key.name,
                description=api_key.description,
                key_prefix=api_key.key_prefix,
                scopes=api_key.scopes,
                rate_limit=api_key.rate_limit,
                is_active=api_key.is_active,
                created_at=api_key.created_at,
                expires_at=api_key.expires_at,
                last_used_at=api_key.last_used_at
            ),
            "full_key": full_key  # Only returned once
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.delete("/clients/{client_id}/api-keys/{key_id}")
async def revoke_client_api_key(
    client_id: str,
    key_id: str,
    db: Session = Depends(get_db),
    auth_service: AuthService = Depends(get_auth_service),
    _: bool = Depends(verify_admin_key)
):
    """Revoke (deactivate) a client API key"""
    
    client = db.query(Client).filter(Client.client_id == client_id).first()
    if not client:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Client not found"
        )
    
    try:
        success = auth_service.revoke_api_key(db, client.id, UUID(key_id))
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found"
            )
        
        return {"success": True}
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/clients/{client_id}/usage", response_model=List[ClientUsageStats])
async def get_client_usage(
    client_id: str,
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db),
    _: bool = Depends(verify_admin_key)
):
    """Get usage statistics for a client"""
    
    client = db.query(Client).filter(Client.client_id == client_id).first()
    if not client:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Client not found"
        )
    
    # Get usage records for the specified period
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)
    
    usage_records = db.query(UsageRecord).filter(
        and_(
            UsageRecord.client_id == client.id,
            UsageRecord.period_start >= start_date,
            UsageRecord.period_end <= end_date
        )
    ).order_by(UsageRecord.period_start).all()
    
    return [
        ClientUsageStats(
            client_id=client_id,
            period_start=record.period_start,
            period_end=record.period_end,
            conversations=record.total_conversations,
            messages=record.total_messages,
            tokens_used=record.total_tokens_used,
            api_calls=record.api_calls_made,
            avg_response_time_ms=record.avg_response_time_ms,
            escalations=record.escalations_triggered,
            estimated_cost_usd=record.estimated_cost_usd / 100.0  # Convert from cents
        )
        for record in usage_records
    ]