from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from ...core.database import get_db
from ...schemas.client import (
    ClientCreateRequest, 
    ClientUpdateRequest, 
    ClientResponse, 
    ClientListResponse
)
from ...models.client import Client
from ...core.vector_db import ClientVectorManager

router = APIRouter()


@router.post("/", response_model=ClientResponse)
async def create_client(
    client_request: ClientCreateRequest,
    db: Session = Depends(get_db)
):
    """Create a new client"""
    
    # Check if client_id already exists
    existing_client = db.query(Client).filter(Client.client_id == client_request.client_id).first()
    if existing_client:
        raise HTTPException(status_code=400, detail="Client ID already exists")
    
    # Create client
    client = Client(
        client_id=client_request.client_id,
        name=client_request.name,
        email=client_request.email,
        config=client_request.config.dict(),
        branding=client_request.branding.dict(),
        features=client_request.features.dict()
    )
    
    db.add(client)
    db.commit()
    db.refresh(client)
    
    # Initialize vector collection for client
    vector_manager = ClientVectorManager(client_request.client_id)
    vector_manager.initialize_client_collection()
    
    return client


@router.get("/", response_model=ClientListResponse)
async def list_clients(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """List all clients with pagination"""
    
    clients = db.query(Client).offset(skip).limit(limit).all()
    total = db.query(Client).count()
    
    return ClientListResponse(
        clients=clients,
        total=total,
        page=skip // limit + 1,
        per_page=limit
    )


@router.get("/{client_id}", response_model=ClientResponse)
async def get_client(
    client_id: str,
    db: Session = Depends(get_db)
):
    """Get client by ID"""
    
    client = db.query(Client).filter(Client.client_id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    return client


@router.put("/{client_id}", response_model=ClientResponse)
async def update_client(
    client_id: str,
    client_update: ClientUpdateRequest,
    db: Session = Depends(get_db)
):
    """Update client configuration"""
    
    client = db.query(Client).filter(Client.client_id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    # Update fields
    if client_update.name is not None:
        client.name = client_update.name
    
    if client_update.email is not None:
        client.email = client_update.email
    
    if client_update.config is not None:
        client.config = client_update.config.dict()
    
    if client_update.branding is not None:
        client.branding = client_update.branding.dict()
    
    if client_update.features is not None:
        client.features = client_update.features.dict()
    
    if client_update.is_active is not None:
        client.is_active = client_update.is_active
    
    if client_update.is_whitelabel is not None:
        client.is_whitelabel = client_update.is_whitelabel
    
    db.commit()
    db.refresh(client)
    
    return client


@router.delete("/{client_id}")
async def delete_client(
    client_id: str,
    db: Session = Depends(get_db)
):
    """Delete a client and all associated data"""
    
    client = db.query(Client).filter(Client.client_id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    # Delete vector collection
    vector_manager = ClientVectorManager(client_id)
    vector_manager.vector_db.delete_collection(vector_manager.collection_name)
    
    # Delete client (cascades to knowledge bases, conversations, etc.)
    db.delete(client)
    db.commit()
    
    return {"message": "Client deleted successfully"}


@router.get("/{client_id}/stats")
async def get_client_stats(
    client_id: str,
    db: Session = Depends(get_db)
):
    """Get client statistics"""
    
    client = db.query(Client).filter(Client.client_id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    from ...models.client import KnowledgeBase, Document
    from ...models.conversation import Conversation, Message
    
    # Get statistics
    knowledge_bases_count = db.query(KnowledgeBase).filter(
        KnowledgeBase.client_id == client_id
    ).count()
    
    documents_count = db.query(Document).join(KnowledgeBase).filter(
        KnowledgeBase.client_id == client_id
    ).count()
    
    conversations_count = db.query(Conversation).filter(
        Conversation.client_id == client_id
    ).count()
    
    messages_count = db.query(Message).join(Conversation).filter(
        Conversation.client_id == client_id
    ).count()
    
    # Get vector stats
    vector_manager = ClientVectorManager(client_id)
    vector_stats = vector_manager.get_stats()
    
    return {
        "client_id": client_id,
        "knowledge_bases": knowledge_bases_count,
        "documents": documents_count,
        "conversations": conversations_count,
        "messages": messages_count,
        "vector_points": vector_stats.get("points_count", 0),
        "is_active": client.is_active,
        "created_at": client.created_at
    }


@router.post("/{client_id}/activate")
async def activate_client(
    client_id: str,
    db: Session = Depends(get_db)
):
    """Activate a client"""
    
    client = db.query(Client).filter(Client.client_id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    client.is_active = True
    db.commit()
    
    return {"message": "Client activated successfully"}


@router.post("/{client_id}/deactivate")
async def deactivate_client(
    client_id: str,
    db: Session = Depends(get_db)
):
    """Deactivate a client"""
    
    client = db.query(Client).filter(Client.client_id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    client.is_active = False
    db.commit()
    
    return {"message": "Client deactivated successfully"}