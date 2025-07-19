"""
Authentication and authorization service for multi-tenant API access
"""

import hashlib
import secrets
import string
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Tuple
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from fastapi import HTTPException, status
from passlib.context import CryptContext

from app.models.clients import Client, ClientAPIKey, ClientStatus, ClientTier
from app.core.database import get_db
from app.core.config import settings

class AuthenticationError(Exception):
    """Raised when authentication fails"""
    pass

class AuthorizationError(Exception):
    """Raised when authorization fails"""
    pass

class AuthService:
    """
    Service for handling client authentication and authorization
    """
    
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.key_prefix_length = 12
        self.key_length = 64
    
    def generate_api_key(self) -> Tuple[str, str, str]:
        """
        Generate a new API key
        Returns: (full_key, key_prefix, key_hash)
        """
        # Generate random key
        alphabet = string.ascii_letters + string.digits
        key = ''.join(secrets.choice(alphabet) for _ in range(self.key_length))
        
        # Add prefix for identification
        prefix = f"cs_{secrets.token_hex(4)}_"
        full_key = f"{prefix}{key}"
        
        # Create hash for storage
        key_hash = self.pwd_context.hash(full_key)
        
        # Extract prefix for identification
        key_prefix = full_key[:self.key_prefix_length]
        
        return full_key, key_prefix, key_hash
    
    def verify_api_key(self, key: str, key_hash: str) -> bool:
        """Verify API key against stored hash"""
        return self.pwd_context.verify(key, key_hash)
    
    def create_api_key(
        self,
        db: Session,
        client_id: UUID,
        name: str,
        description: Optional[str] = None,
        scopes: Optional[List[str]] = None,
        rate_limit: Optional[int] = None,
        expires_days: Optional[int] = None
    ) -> Tuple[ClientAPIKey, str]:
        """
        Create a new API key for a client
        Returns: (api_key_record, full_key)
        """
        # Verify client exists and is active
        client = db.query(Client).filter(Client.id == client_id).first()
        if not client:
            raise ValueError("Client not found")
        
        if not client.is_active():
            raise ValueError("Client is not active")
        
        # Generate API key
        full_key, key_prefix, key_hash = self.generate_api_key()
        
        # Set default scopes if none provided
        if scopes is None:
            scopes = ["chat:send", "chat:receive", "knowledge:read"]
        
        # Set expiration if specified
        expires_at = None
        if expires_days:
            expires_at = datetime.now(timezone.utc) + timedelta(days=expires_days)
        
        # Create API key record
        api_key = ClientAPIKey(
            client_id=client_id,
            key_prefix=key_prefix,
            key_hash=key_hash,
            name=name,
            description=description,
            scopes=scopes,
            rate_limit=rate_limit or 1000,
            expires_at=expires_at
        )
        
        db.add(api_key)
        db.commit()
        db.refresh(api_key)
        
        return api_key, full_key
    
    def authenticate_api_key(self, db: Session, api_key: str) -> Optional[Tuple[Client, ClientAPIKey]]:
        """
        Authenticate API key and return client and key info
        Returns None if authentication fails
        """
        if not api_key or len(api_key) < self.key_prefix_length:
            return None
        
        # Extract prefix for efficient lookup
        key_prefix = api_key[:self.key_prefix_length]
        
        # Find potential matching keys
        potential_keys = db.query(ClientAPIKey).filter(
            and_(
                ClientAPIKey.key_prefix == key_prefix,
                ClientAPIKey.is_active == True
            )
        ).all()
        
        # Verify key hash
        for key_record in potential_keys:
            if self.verify_api_key(api_key, key_record.key_hash):
                # Check if key is still valid
                if not key_record.is_valid():
                    continue
                
                # Get client
                client = db.query(Client).filter(Client.id == key_record.client_id).first()
                if not client or not client.is_active():
                    continue
                
                # Update last used timestamps
                key_record.update_last_used()
                client.update_last_active()
                db.commit()
                
                return client, key_record
        
        return None
    
    def authorize_scope(self, api_key: ClientAPIKey, required_scope: str) -> bool:
        """
        Check if API key has required scope
        """
        return required_scope in api_key.scopes or "admin" in api_key.scopes
    
    def check_rate_limit(self, db: Session, api_key: ClientAPIKey) -> bool:
        """
        Check if API key is within rate limits
        This is a simplified implementation - in production, use Redis for rate limiting
        """
        # For now, just return True - implement proper rate limiting with Redis
        return True
    
    def revoke_api_key(self, db: Session, client_id: UUID, api_key_id: UUID) -> bool:
        """
        Revoke (deactivate) an API key
        """
        api_key = db.query(ClientAPIKey).filter(
            and_(
                ClientAPIKey.id == api_key_id,
                ClientAPIKey.client_id == client_id
            )
        ).first()
        
        if not api_key:
            return False
        
        api_key.is_active = False
        db.commit()
        return True
    
    def list_client_api_keys(self, db: Session, client_id: UUID) -> List[ClientAPIKey]:
        """
        List all API keys for a client (excluding sensitive data)
        """
        return db.query(ClientAPIKey).filter(
            ClientAPIKey.client_id == client_id
        ).order_by(ClientAPIKey.created_at.desc()).all()
    
    def get_client_by_id(self, db: Session, client_id: str) -> Optional[Client]:
        """
        Get client by client_id
        """
        return db.query(Client).filter(Client.client_id == client_id).first()
    
    def create_client(
        self,
        db: Session,
        client_id: str,
        name: str,
        email: str,
        tier: ClientTier = ClientTier.TRIAL,
        configuration: Optional[Dict[str, Any]] = None
    ) -> Client:
        """
        Create a new client
        """
        # Check if client_id already exists
        existing = self.get_client_by_id(db, client_id)
        if existing:
            raise ValueError(f"Client with ID '{client_id}' already exists")
        
        # Create client with default configuration
        client = Client(
            client_id=client_id,
            name=name,
            email=email,
            tier=tier,
            status=ClientStatus.TRIAL if tier == ClientTier.TRIAL else ClientStatus.ACTIVE,
            configuration=configuration or {}
        )
        
        # Set default configuration
        if not client.configuration:
            client.configuration = client.get_default_configuration()
        
        # Set trial period for trial accounts
        if tier == ClientTier.TRIAL:
            client.trial_ends_at = datetime.now(timezone.utc) + timedelta(days=14)
        
        db.add(client)
        db.commit()
        db.refresh(client)
        
        return client
    
    def update_client_configuration(
        self,
        db: Session,
        client_id: UUID,
        configuration: Dict[str, Any]
    ) -> Optional[Client]:
        """
        Update client configuration
        """
        client = db.query(Client).filter(Client.id == client_id).first()
        if not client:
            return None
        
        # Merge configuration
        client.configuration = {**client.configuration, **configuration}
        db.commit()
        db.refresh(client)
        
        return client

# Dependency injection
def get_auth_service() -> AuthService:
    """Get authentication service instance"""
    return AuthService()

# FastAPI dependencies
async def get_current_client(
    api_key: str,
    db: Session = next(get_db()),
    auth_service: AuthService = get_auth_service()
) -> Tuple[Client, ClientAPIKey]:
    """
    FastAPI dependency to get current authenticated client
    """
    result = auth_service.authenticate_api_key(db, api_key)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return result

def require_scope(required_scope: str):
    """
    FastAPI dependency factory to require specific scope
    """
    def check_scope(
        current_client: Tuple[Client, ClientAPIKey] = get_current_client,
        auth_service: AuthService = get_auth_service()
    ):
        client, api_key = current_client
        if not auth_service.authorize_scope(api_key, required_scope):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required scope: {required_scope}"
            )
        return current_client
    
    return check_scope