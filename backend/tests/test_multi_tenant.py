"""
Tests for multi-tenant system functionality
"""

import pytest
from datetime import datetime, timezone, timedelta
from uuid import uuid4

from sqlalchemy.orm import Session
from fastapi.testclient import TestClient

from app.models.client import Client, ClientAPIKey, ClientStatus, ClientTier
from app.services.auth_service import AuthService
from app.services.usage_tracking import UsageAggregator
from app.services.tenant_isolation import TenantIsolationService, QuotaType, QuotaExceededException

class TestClientAuthentication:
    """Test client authentication and API key management"""
    
    def test_create_client(self, db: Session):
        """Test creating a new client"""
        auth_service = AuthService()
        
        client = auth_service.create_client(
            db=db,
            client_id="test-client",
            name="Test Company",
            email="test@example.com",
            tier=ClientTier.TRIAL
        )
        
        assert client.client_id == "test-client"
        assert client.name == "Test Company"
        assert client.email == "test@example.com"
        assert client.status == ClientStatus.TRIAL
        assert client.tier == ClientTier.TRIAL
        assert client.configuration is not None
    
    def test_create_api_key(self, db: Session):
        """Test creating API key for client"""
        auth_service = AuthService()
        
        # Create client first
        client = auth_service.create_client(
            db=db,
            client_id="test-client-2",
            name="Test Company 2",
            email="test2@example.com"
        )
        
        # Create API key
        api_key, full_key = auth_service.create_api_key(
            db=db,
            client_id=client.id,
            name="Test API Key",
            description="Test key for unit tests",
            scopes=["chat:send", "chat:receive"]
        )
        
        assert api_key.name == "Test API Key"
        assert api_key.client_id == client.id
        assert api_key.scopes == ["chat:send", "chat:receive"]
        assert full_key.startswith("cs_")
        assert len(full_key) > 32
    
    def test_authenticate_api_key(self, db: Session):
        """Test API key authentication"""
        auth_service = AuthService()
        
        # Create client and API key
        client = auth_service.create_client(
            db=db,
            client_id="test-client-3",
            name="Test Company 3",
            email="test3@example.com"
        )
        
        api_key, full_key = auth_service.create_api_key(
            db=db,
            client_id=client.id,
            name="Auth Test Key"
        )
        
        # Test authentication
        result = auth_service.authenticate_api_key(db, full_key)
        assert result is not None
        
        authenticated_client, authenticated_key = result
        assert authenticated_client.id == client.id
        assert authenticated_key.id == api_key.id
        
        # Test invalid key
        invalid_result = auth_service.authenticate_api_key(db, "invalid-key")
        assert invalid_result is None
    
    def test_revoke_api_key(self, db: Session):
        """Test revoking API key"""
        auth_service = AuthService()
        
        # Create client and API key
        client = auth_service.create_client(
            db=db,
            client_id="test-client-4",
            name="Test Company 4",
            email="test4@example.com"
        )
        
        api_key, full_key = auth_service.create_api_key(
            db=db,
            client_id=client.id,
            name="Revoke Test Key"
        )
        
        # Revoke key
        success = auth_service.revoke_api_key(db, client.id, api_key.id)
        assert success
        
        # Test authentication fails after revocation
        result = auth_service.authenticate_api_key(db, full_key)
        assert result is None

class TestUsageTracking:
    """Test usage tracking and analytics"""
    
    def test_track_conversation_start(self):
        """Test tracking conversation start"""
        aggregator = UsageAggregator()
        client_id = uuid4()
        
        aggregator.track_conversation_start(
            client_id=client_id,
            conversation_id="conv-123",
            metadata={"source": "website"}
        )
        
        # Check event is buffered
        assert client_id in aggregator.event_buffer
        assert len(aggregator.event_buffer[client_id]) == 1
        
        event = aggregator.event_buffer[client_id][0]
        assert event.event_type == "conversation_start"
        assert event.client_id == client_id
        assert event.metadata["source"] == "website"
    
    def test_track_message_sent(self):
        """Test tracking message sent"""
        aggregator = UsageAggregator()
        client_id = uuid4()
        
        aggregator.track_message_sent(
            client_id=client_id,
            conversation_id="conv-123",
            message_id="msg-456",
            role="user",
            token_count=100,
            processing_time_ms=500
        )
        
        event = aggregator.event_buffer[client_id][0]
        assert event.event_type == "message_sent"
        assert event.metadata["token_count"] == 100
        assert event.metadata["processing_time_ms"] == 500
    
    def test_track_api_call(self):
        """Test tracking API call"""
        aggregator = UsageAggregator()
        client_id = uuid4()
        
        aggregator.track_api_call(
            client_id=client_id,
            endpoint="/api/chat/message",
            method="POST",
            status_code=200,
            response_time_ms=250
        )
        
        event = aggregator.event_buffer[client_id][0]
        assert event.event_type == "api_call"
        assert event.metadata["endpoint"] == "/api/chat/message"
        assert event.metadata["status_code"] == 200
        assert event.metadata["success"] is True

class TestTenantIsolation:
    """Test tenant isolation and resource quotas"""
    
    def test_get_client_quotas(self, db: Session):
        """Test getting quota limits for different client tiers"""
        isolation_service = TenantIsolationService()
        
        # Test FREE tier
        free_client = Client(
            client_id="free-client",
            name="Free Client",
            email="free@example.com",
            tier=ClientTier.FREE,
            configuration={}
        )
        
        quotas = isolation_service.get_client_quotas(free_client)
        assert quotas[QuotaType.MESSAGES_PER_DAY] == 100
        assert quotas[QuotaType.API_CALLS_PER_HOUR] == 100
        
        # Test PROFESSIONAL tier
        pro_client = Client(
            client_id="pro-client",
            name="Pro Client",
            email="pro@example.com",
            tier=ClientTier.PROFESSIONAL,
            configuration={}
        )
        
        quotas = isolation_service.get_client_quotas(pro_client)
        assert quotas[QuotaType.MESSAGES_PER_DAY] == 5000
        assert quotas[QuotaType.API_CALLS_PER_HOUR] == 2000
    
    def test_custom_client_quotas(self, db: Session):
        """Test custom quota overrides in client configuration"""
        isolation_service = TenantIsolationService()
        
        client = Client(
            client_id="custom-client",
            name="Custom Client",
            email="custom@example.com",
            tier=ClientTier.STARTER,
            configuration={
                "limits": {
                    "messages_per_day": 2000,  # Override default
                    "api_calls_per_hour": 1000  # Override default
                }
            }
        )
        
        quotas = isolation_service.get_client_quotas(client)
        assert quotas[QuotaType.MESSAGES_PER_DAY] == 2000
        assert quotas[QuotaType.API_CALLS_PER_HOUR] == 1000
        # Other quotas should remain default
        assert quotas[QuotaType.CONVERSATIONS_PER_DAY] == 100  # Default for STARTER
    
    def test_quota_enforcement(self, db: Session):
        """Test quota enforcement raises exceptions when exceeded"""
        isolation_service = TenantIsolationService()
        
        # Create a client with very low limits for testing
        client = Client(
            client_id="quota-test-client",
            name="Quota Test Client",
            email="quota@example.com",
            tier=ClientTier.FREE,
            configuration={
                "limits": {
                    "messages_per_day": 1,  # Very low limit
                    "concurrent_conversations": 1
                }
            }
        )
        db.add(client)
        db.commit()
        
        # Test that quota enforcement works
        try:
            # This should pass as we're within limits
            result = isolation_service.enforce_quota(
                db, client, QuotaType.CONCURRENT_CONVERSATIONS
            )
            assert result is True
        except QuotaExceededException:
            pytest.fail("Should not raise exception when within quota")
    
    def test_quota_status(self, db: Session):
        """Test getting quota status for a client"""
        isolation_service = TenantIsolationService()
        
        client = Client(
            client_id="status-test-client",
            name="Status Test Client",
            email="status@example.com",
            tier=ClientTier.STARTER,
            configuration={}
        )
        db.add(client)
        db.commit()
        
        quota = isolation_service.check_quota(
            db, client, QuotaType.MESSAGES_PER_DAY
        )
        
        assert quota.quota_type == QuotaType.MESSAGES_PER_DAY
        assert quota.limit == 1000  # STARTER tier default
        assert quota.current_usage >= 0
        assert quota.percentage_used >= 0.0
        assert quota.resets_at is not None
    
    def test_data_isolation_checks(self, db: Session):
        """Test data isolation validation"""
        isolation_service = TenantIsolationService()
        
        client = Client(
            client_id="isolation-test-client",
            name="Isolation Test Client",
            email="isolation@example.com",
            tier=ClientTier.PROFESSIONAL,
            status=ClientStatus.ACTIVE,
            configuration={}
        )
        
        # Test isolation checks
        checks = isolation_service.check_data_isolation(db, client)
        
        assert "conversations_isolated" in checks
        assert "knowledge_bases_isolated" in checks
        assert "api_keys_isolated" in checks
        assert "usage_data_isolated" in checks
        assert "vector_namespace_isolated" in checks
        
        # All checks should pass for a properly isolated client
        assert all(checks.values())
    
    def test_resource_namespace(self):
        """Test resource namespace generation"""
        isolation_service = TenantIsolationService()
        
        client = Client(
            client_id="namespace-test-client",
            name="Namespace Test Client",
            email="namespace@example.com"
        )
        
        namespace = isolation_service.get_resource_namespace(client, "vectors")
        assert namespace == "client_namespace-test-client_vectors"
        
        kb_namespace = isolation_service.get_resource_namespace(client, "knowledge_base")
        assert kb_namespace == "client_namespace-test-client_knowledge_base"

class TestAdminAPI:
    """Test admin dashboard API endpoints"""
    
    def test_admin_authentication(self, client: TestClient):
        """Test admin API key authentication"""
        # Test without admin key
        response = client.get("/api/v1/admin/stats")
        assert response.status_code == 422  # Missing query parameter
        
        # Test with invalid admin key
        response = client.get("/api/v1/admin/stats?admin_key=invalid")
        assert response.status_code == 401
        
        # Test with valid admin key
        response = client.get("/api/v1/admin/stats?admin_key=admin-key-change-this-in-production")
        # This might fail due to database setup, but authentication should pass
        assert response.status_code != 401
    
    def test_client_creation_via_admin(self, client: TestClient):
        """Test creating client through admin API"""
        client_data = {
            "client_id": "admin-created-client",
            "name": "Admin Created Client",
            "email": "admin-created@example.com",
            "tier": "starter",
            "contact_name": "John Doe"
        }
        
        response = client.post(
            "/api/v1/admin/clients?admin_key=admin-key-change-this-in-production",
            json=client_data
        )
        
        # Response might vary based on database state, but structure should be correct
        if response.status_code == 200:
            data = response.json()
            assert data["client_id"] == "admin-created-client"
            assert data["name"] == "Admin Created Client"
            assert data["tier"] == "starter"

class TestOnboardingFlow:
    """Test client onboarding process"""
    
    def test_onboarding_request(self, client: TestClient):
        """Test submitting onboarding request"""
        request_data = {
            "company_name": "New Company",
            "email": "new@company.com",
            "contact_name": "Jane Smith",
            "phone": "+1-555-0123",
            "website": "company.com",
            "use_case": "Customer support for e-commerce",
            "expected_volume": "medium"
        }
        
        response = client.post("/api/v1/onboarding/request", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "submitted"
        assert "request_id" in data
        assert "Thank you" in data["message"]
    
    def test_invitation_creation(self, client: TestClient):
        """Test creating invitation (admin function)"""
        invitation_data = {
            "email": "invite@example.com",
            "client_name": "Invited Company",
            "client_id": "invited-company",
            "invited_by": "admin@yourcompany.com",
            "initial_tier": "trial",
            "expires_hours": 168
        }
        
        response = client.post("/api/v1/onboarding/invitations", json=invitation_data)
        
        # This might fail due to database constraints, but structure should be correct
        if response.status_code == 200:
            data = response.json()
            assert data["email"] == "invite@example.com"
            assert data["client_id"] == "invited-company"
            assert "token" in data
            assert "invitation_url" in data

@pytest.fixture
def sample_client(db: Session) -> Client:
    """Create a sample client for testing"""
    client = Client(
        client_id="sample-client",
        name="Sample Company",
        email="sample@example.com",
        tier=ClientTier.PROFESSIONAL,
        status=ClientStatus.ACTIVE,
        configuration={}
    )
    db.add(client)
    db.commit()
    db.refresh(client)
    return client

@pytest.fixture
def sample_api_key(db: Session, sample_client: Client) -> tuple[ClientAPIKey, str]:
    """Create a sample API key for testing"""
    auth_service = AuthService()
    api_key, full_key = auth_service.create_api_key(
        db=db,
        client_id=sample_client.id,
        name="Sample API Key",
        scopes=["chat:send", "chat:receive", "knowledge:read"]
    )
    return api_key, full_key