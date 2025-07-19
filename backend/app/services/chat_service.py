"""
Unified Chat Service

Orchestrates the complete customer service bot conversation flow by integrating:
- Conversation management
- Knowledge base search  
- Response generation with tone injection
- Quality control and escalation
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone

from .conversation_service import get_conversation_manager, ConversationManager, ConversationContext
from .response_service import get_response_generator, ResponseGenerator, ResponseContext, ClientToneConfig, GeneratedResponse
from .escalation_service import get_escalation_manager, EscalationManager, EscalationEvent
from .llm_service import get_llm_service, LLMService
from ..services.search_service import get_search_service
from ..services.knowledge_base_service import get_knowledge_base_service
from ..core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ChatRequest:
    """Complete chat request with all context"""
    message: str
    client_id: str
    session_id: str
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ChatResponse:
    """Complete chat response with all metadata"""
    message: str
    conversation_id: str
    response_type: str
    confidence_score: float
    should_escalate: bool
    escalation_info: Optional[Dict[str, Any]] = None
    processing_time_ms: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.escalation_info is None and self.should_escalate:
            self.escalation_info = {"reason": "unknown", "priority": "medium"}


@dataclass  
class ClientConfiguration:
    """Client-specific configuration for chat behavior"""
    client_id: str
    tone_config: ClientToneConfig
    knowledge_base_ids: List[str] = None
    escalation_config: Dict[str, Any] = None
    rate_limits: Dict[str, int] = None
    custom_prompts: Dict[str, str] = None
    
    def __post_init__(self):
        if self.knowledge_base_ids is None:
            self.knowledge_base_ids = []
        if self.escalation_config is None:
            self.escalation_config = {}
        if self.rate_limits is None:
            self.rate_limits = {"messages_per_hour": 100}
        if self.custom_prompts is None:
            self.custom_prompts = {}


class ChatService:
    """Main chat service orchestrating all components"""
    
    def __init__(
        self,
        conversation_manager: Optional[ConversationManager] = None,
        response_generator: Optional[ResponseGenerator] = None,
        escalation_manager: Optional[EscalationManager] = None,
        llm_service: Optional[LLMService] = None
    ):
        self.conversation_manager = conversation_manager or get_conversation_manager()
        self.response_generator = response_generator or get_response_generator()
        self.escalation_manager = escalation_manager or get_escalation_manager()
        self.llm_service = llm_service or get_llm_service()
        
        # Services for knowledge retrieval
        self.search_service = get_search_service()
        self.knowledge_base_service = get_knowledge_base_service()
        
        # Client configurations cache
        self.client_configs: Dict[str, ClientConfiguration] = {}
        
        # Default client configuration
        self.default_client_config = ClientConfiguration(
            client_id="default",
            tone_config=ClientToneConfig()
        )
    
    async def process_chat_message(self, request: ChatRequest) -> ChatResponse:
        """Process a complete chat message through the full pipeline"""
        start_time = datetime.now(timezone.utc)
        
        try:
            # Get or create conversation
            conversation = await self._get_or_create_conversation(request)
            
            # Get client configuration
            client_config = await self._get_client_configuration(request.client_id)
            
            # Add user message to conversation
            await self.conversation_manager.add_user_message(
                conversation.conversation_id,
                request.message,
                request.metadata
            )
            
            # Search knowledge base for relevant information
            knowledge_context = await self._search_knowledge_base(
                request.message,
                client_config.knowledge_base_ids
            )
            
            # Get conversation history for LLM context
            conversation_history = await self.conversation_manager.get_context_for_llm(
                conversation.conversation_id
            )
            
            # Generate response
            response_context = ResponseContext(
                conversation_id=conversation.conversation_id,
                client_id=request.client_id,
                user_message=request.message,
                conversation_history=conversation_history,
                client_config=client_config.tone_config,
                knowledge_context=knowledge_context,
                metadata=request.metadata
            )
            
            generated_response = await self.response_generator.generate_response(response_context)
            
            # Add assistant response to conversation
            await self.conversation_manager.add_assistant_message(
                conversation.conversation_id,
                generated_response.content,
                {
                    "confidence_score": generated_response.confidence_score,
                    "response_type": generated_response.response_type.value,
                    "tone_applied": generated_response.tone_applied.value,
                    "tokens_used": generated_response.tokens_used,
                    "cost_usd": generated_response.cost_usd
                }
            )
            
            # Evaluate escalation need
            escalation_event = await self.escalation_manager.evaluate_escalation(
                conversation.conversation_id,
                generated_response,
                request.message,
                client_config.escalation_config
            )
            
            # Calculate total processing time
            end_time = datetime.now(timezone.utc)
            total_processing_time = int((end_time - start_time).total_seconds() * 1000)
            
            # Build chat response
            chat_response = ChatResponse(
                message=generated_response.content,
                conversation_id=conversation.conversation_id,
                response_type=generated_response.response_type.value,
                confidence_score=generated_response.confidence_score,
                should_escalate=generated_response.should_escalate or escalation_event is not None,
                escalation_info=escalation_event.to_dict() if escalation_event else None,
                processing_time_ms=total_processing_time,
                tokens_used=generated_response.tokens_used,
                cost_usd=generated_response.cost_usd,
                metadata={
                    "knowledge_context_found": knowledge_context is not None,
                    "conversation_length": len(conversation.messages),
                    "tone_applied": generated_response.tone_applied.value,
                    "llm_processing_time_ms": generated_response.processing_time_ms,
                    "model_used": generated_response.metadata.get("model_used"),
                    "provider_used": generated_response.metadata.get("provider_used")
                }
            )
            
            logger.info(
                f"Processed chat message for {request.client_id}: "
                f"confidence={generated_response.confidence_score:.2f}, "
                f"escalation={chat_response.should_escalate}, "
                f"time={total_processing_time}ms"
            )
            
            return chat_response
            
        except Exception as e:
            logger.error(f"Error processing chat message: {e}")
            
            # Return error response
            end_time = datetime.now(timezone.utc)
            error_processing_time = int((end_time - start_time).total_seconds() * 1000)
            
            return ChatResponse(
                message="I apologize, but I'm experiencing technical difficulties. Please try again or contact our support team.",
                conversation_id=request.conversation_id or "error",
                response_type="error",
                confidence_score=0.0,
                should_escalate=True,
                escalation_info={
                    "reason": "system_error",
                    "priority": "high",
                    "error_details": str(e)
                },
                processing_time_ms=error_processing_time,
                metadata={"error": str(e)}
            )
    
    async def start_conversation(self, request: ChatRequest) -> ChatResponse:
        """Start a new conversation with greeting"""
        try:
            # Get client configuration
            client_config = await self._get_client_configuration(request.client_id)
            
            # Create new conversation
            conversation = await self.conversation_manager.create_conversation(
                request.client_id,
                request.session_id,
                request.metadata
            )
            
            # Generate greeting
            greeting = await self.response_generator.generate_greeting(client_config.tone_config)
            
            # Add greeting to conversation
            await self.conversation_manager.add_assistant_message(
                conversation.conversation_id,
                greeting,
                {"message_type": "greeting"}
            )
            
            return ChatResponse(
                message=greeting,
                conversation_id=conversation.conversation_id,
                response_type="greeting",
                confidence_score=1.0,
                should_escalate=False,
                processing_time_ms=0,
                tokens_used=0,
                cost_usd=0.0,
                metadata={"conversation_started": True}
            )
            
        except Exception as e:
            logger.error(f"Error starting conversation: {e}")
            raise
    
    async def end_conversation(self, conversation_id: str, client_id: str) -> ChatResponse:
        """End conversation with goodbye"""
        try:
            # Get client configuration
            client_config = await self._get_client_configuration(client_id)
            
            # Generate goodbye
            goodbye = await self.response_generator.generate_goodbye(client_config.tone_config)
            
            # Add goodbye to conversation
            await self.conversation_manager.add_assistant_message(
                conversation_id,
                goodbye,
                {"message_type": "goodbye"}
            )
            
            # Archive conversation
            await self.conversation_manager.archive_conversation(conversation_id)
            
            return ChatResponse(
                message=goodbye,
                conversation_id=conversation_id,
                response_type="goodbye",
                confidence_score=1.0,
                should_escalate=False,
                processing_time_ms=0,
                tokens_used=0,
                cost_usd=0.0,
                metadata={"conversation_ended": True}
            )
            
        except Exception as e:
            logger.error(f"Error ending conversation: {e}")
            raise
    
    async def _get_or_create_conversation(self, request: ChatRequest) -> ConversationContext:
        """Get existing conversation or create new one"""
        if request.conversation_id:
            conversation = await self.conversation_manager.get_conversation(request.conversation_id)
            if conversation:
                return conversation
        
        # Create new conversation
        return await self.conversation_manager.create_conversation(
            request.client_id,
            request.session_id,
            request.metadata
        )
    
    async def _get_client_configuration(self, client_id: str) -> ClientConfiguration:
        """Get client configuration (cached or from database)"""
        if client_id in self.client_configs:
            return self.client_configs[client_id]
        
        # In production, this would load from database
        # For now, return default configuration
        config = ClientConfiguration(
            client_id=client_id,
            tone_config=ClientToneConfig(
                primary_tone="professional",
                persona_description=f"A helpful customer service representative for {client_id}"
            )
        )
        
        # Cache the configuration
        self.client_configs[client_id] = config
        return config
    
    async def _search_knowledge_base(
        self, 
        query: str, 
        knowledge_base_ids: List[str]
    ) -> Optional[str]:
        """Search knowledge base for relevant information"""
        if not knowledge_base_ids:
            return None
        
        try:
            # Use the first knowledge base for now
            # In production, search across all relevant knowledge bases
            primary_kb_id = knowledge_base_ids[0] if knowledge_base_ids else "default"
            
            # Perform hybrid search
            search_results = await self.search_service.hybrid_search(
                query=query,
                knowledge_base_id=primary_kb_id,
                limit=3,
                confidence_threshold=0.6
            )
            
            if search_results and search_results.get("results"):
                # Combine top results into context
                contexts = []
                for result in search_results["results"][:2]:  # Top 2 results
                    contexts.append(result.get("content", ""))
                
                return " ".join(contexts)
            
        except Exception as e:
            logger.error(f"Knowledge base search failed: {e}")
        
        return None
    
    async def update_client_configuration(
        self, 
        client_id: str, 
        config: ClientConfiguration
    ):
        """Update client configuration"""
        self.client_configs[client_id] = config
        logger.info(f"Updated configuration for client {client_id}")
    
    async def get_conversation_history(
        self, 
        conversation_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get conversation history"""
        conversation = await self.conversation_manager.get_conversation(conversation_id)
        if not conversation:
            return []
        
        messages = conversation.get_recent_messages(limit) if limit else conversation.messages
        
        return [
            {
                "role": msg.role.value,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "metadata": msg.metadata
            }
            for msg in messages
        ]
    
    async def get_active_escalations(
        self, 
        client_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get active escalations for client"""
        escalations = self.escalation_manager.get_active_escalations(client_id=client_id)
        return [escalation.to_dict() for escalation in escalations]
    
    async def resolve_escalation(
        self, 
        escalation_id: str, 
        resolution_notes: str,
        agent_id: Optional[str] = None
    ):
        """Resolve an escalation"""
        await self.escalation_manager.resolve_escalation(
            escalation_id, 
            resolution_notes, 
            agent_id
        )
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics"""
        return {
            "conversation_stats": self.conversation_manager.get_conversation_stats(),
            "escalation_stats": self.escalation_manager.get_escalation_stats(),
            "llm_metrics": self.llm_service.get_metrics(),
            "active_clients": len(self.client_configs),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# Global chat service instance
_chat_service = None

def get_chat_service() -> ChatService:
    """Get or create global chat service instance"""
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service