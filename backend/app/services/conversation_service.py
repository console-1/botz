"""
Conversation Service

Manages conversation context, history, and memory for customer service bot interactions.
Provides context window optimization, conversation summarization, and session management.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import json
import hashlib
from collections import deque

from sqlalchemy.orm import Session
from ..models.database import get_db
from ..models.conversation import Conversation, ConversationMessage
from ..services.llm_service import get_llm_service, LLMService
from ..core.config import settings

logger = logging.getLogger(__name__)


class MessageRole(str, Enum):
    """Message roles in conversation"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


class ConversationState(str, Enum):
    """Conversation states"""
    ACTIVE = "active"
    WAITING = "waiting"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    ARCHIVED = "archived"


@dataclass
class ConversationMessage:
    """Structured conversation message"""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    tokens: Optional[int] = None
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to LLM API format"""
        return {
            "role": self.role.value,
            "content": self.content
        }


@dataclass
class ConversationContext:
    """Complete conversation context"""
    conversation_id: str
    client_id: str
    session_id: str
    messages: List[ConversationMessage] = field(default_factory=list)
    state: ConversationState = ConversationState.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    summary: Optional[str] = None
    total_tokens: int = 0
    
    def add_message(self, message: ConversationMessage):
        """Add message to conversation"""
        self.messages.append(message)
        self.updated_at = datetime.now(timezone.utc)
        if message.tokens:
            self.total_tokens += message.tokens
    
    def get_recent_messages(self, limit: int = 10) -> List[ConversationMessage]:
        """Get recent messages"""
        return self.messages[-limit:] if len(self.messages) > limit else self.messages
    
    def get_token_count(self) -> int:
        """Get total token count for conversation"""
        return sum(msg.tokens or 0 for msg in self.messages)


class ConversationManager:
    """Manages conversation context and memory"""
    
    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or get_llm_service()
        self.active_conversations: Dict[str, ConversationContext] = {}
        self.max_context_tokens = settings.max_context_tokens
        self.conversation_timeout = timedelta(hours=2)  # Auto-archive after 2 hours
        self.max_active_conversations = 1000  # Memory limit
        
    async def create_conversation(
        self, 
        client_id: str, 
        session_id: str,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> ConversationContext:
        """Create new conversation"""
        conversation_id = self._generate_conversation_id(client_id, session_id)
        
        context = ConversationContext(
            conversation_id=conversation_id,
            client_id=client_id,
            session_id=session_id,
            metadata=initial_context or {}
        )
        
        # Add system message if client has custom instructions
        system_message = await self._get_system_message(client_id)
        if system_message:
            context.add_message(ConversationMessage(
                role=MessageRole.SYSTEM,
                content=system_message,
                metadata={"type": "system_instructions"}
            ))
        
        self.active_conversations[conversation_id] = context
        
        # Cleanup old conversations if needed
        await self._cleanup_old_conversations()
        
        logger.info(f"Created conversation {conversation_id} for client {client_id}")
        return context
    
    async def get_conversation(self, conversation_id: str) -> Optional[ConversationContext]:
        """Get existing conversation"""
        if conversation_id in self.active_conversations:
            context = self.active_conversations[conversation_id]
            
            # Check if conversation is too old
            if datetime.now(timezone.utc) - context.updated_at > self.conversation_timeout:
                await self.archive_conversation(conversation_id)
                return None
            
            return context
        
        # Try to load from database
        return await self._load_conversation_from_db(conversation_id)
    
    async def add_user_message(
        self, 
        conversation_id: str, 
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationContext:
        """Add user message to conversation"""
        context = await self.get_conversation(conversation_id)
        if not context:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        # Estimate tokens for the message
        tokens = await self._estimate_tokens(content)
        
        message = ConversationMessage(
            role=MessageRole.USER,
            content=content,
            metadata=metadata or {},
            tokens=tokens
        )
        
        context.add_message(message)
        
        # Check if we need to summarize or truncate
        await self._manage_context_window(context)
        
        logger.info(f"Added user message to {conversation_id} ({tokens} tokens)")
        return context
    
    async def add_assistant_message(
        self, 
        conversation_id: str, 
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationContext:
        """Add assistant message to conversation"""
        context = await self.get_conversation(conversation_id)
        if not context:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        # Estimate tokens for the message
        tokens = await self._estimate_tokens(content)
        
        message = ConversationMessage(
            role=MessageRole.ASSISTANT,
            content=content,
            metadata=metadata or {},
            tokens=tokens
        )
        
        context.add_message(message)
        
        logger.info(f"Added assistant message to {conversation_id} ({tokens} tokens)")
        return context
    
    async def get_context_for_llm(
        self, 
        conversation_id: str,
        include_system: bool = True
    ) -> List[Dict[str, str]]:
        """Get conversation context formatted for LLM API"""
        context = await self.get_conversation(conversation_id)
        if not context:
            return []
        
        messages = []
        
        # Add system message if requested and available
        if include_system:
            system_messages = [msg for msg in context.messages if msg.role == MessageRole.SYSTEM]
            if system_messages:
                # Use the most recent system message
                messages.append(system_messages[-1].to_dict())
        
        # Add conversation summary if available
        if context.summary:
            messages.append({
                "role": "system",
                "content": f"Previous conversation summary: {context.summary}"
            })
        
        # Add recent conversation messages (excluding system messages)
        conversation_messages = [
            msg for msg in context.messages 
            if msg.role != MessageRole.SYSTEM
        ]
        
        # Get messages that fit in context window
        selected_messages = await self._select_messages_for_context(
            conversation_messages, 
            self.max_context_tokens - sum(await self._estimate_tokens(msg["content"]) for msg in messages)
        )
        
        messages.extend([msg.to_dict() for msg in selected_messages])
        
        return messages
    
    async def summarize_conversation(self, conversation_id: str) -> str:
        """Generate conversation summary"""
        context = await self.get_conversation(conversation_id)
        if not context or len(context.messages) < 3:
            return ""
        
        # Get conversation messages (excluding system)
        conversation_messages = [
            msg for msg in context.messages 
            if msg.role != MessageRole.SYSTEM
        ]
        
        if not conversation_messages:
            return ""
        
        # Format messages for summarization
        conversation_text = "\n".join([
            f"{msg.role.value}: {msg.content}" 
            for msg in conversation_messages
        ])
        
        # Generate summary using LLM
        try:
            summary_prompt = [
                {
                    "role": "system",
                    "content": (
                        "Summarize this customer service conversation concisely. "
                        "Include key issues, decisions made, and current status. "
                        "Keep it under 200 words."
                    )
                },
                {
                    "role": "user",
                    "content": f"Conversation to summarize:\n{conversation_text}"
                }
            ]
            
            response = await self.llm_service.generate_response(
                messages=summary_prompt,
                max_tokens=300,
                temperature=0.3
            )
            
            summary = response.content.strip()
            context.summary = summary
            
            logger.info(f"Generated summary for {conversation_id}: {len(summary)} chars")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate summary for {conversation_id}: {e}")
            return ""
    
    async def archive_conversation(self, conversation_id: str):
        """Archive conversation to database"""
        if conversation_id not in self.active_conversations:
            return
        
        context = self.active_conversations[conversation_id]
        context.state = ConversationState.ARCHIVED
        
        # Save to database
        await self._save_conversation_to_db(context)
        
        # Remove from active conversations
        del self.active_conversations[conversation_id]
        
        logger.info(f"Archived conversation {conversation_id}")
    
    async def escalate_conversation(
        self, 
        conversation_id: str, 
        reason: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Mark conversation as escalated"""
        context = await self.get_conversation(conversation_id)
        if not context:
            return
        
        context.state = ConversationState.ESCALATED
        context.metadata.update({
            "escalation_reason": reason,
            "escalated_at": datetime.now(timezone.utc).isoformat(),
            "escalation_metadata": metadata or {}
        })
        
        logger.info(f"Escalated conversation {conversation_id}: {reason}")
    
    async def _manage_context_window(self, context: ConversationContext):
        """Manage conversation context window size"""
        current_tokens = context.get_token_count()
        
        if current_tokens <= self.max_context_tokens:
            return
        
        logger.info(f"Context window exceeded ({current_tokens} > {self.max_context_tokens}), managing context")
        
        # First, try to summarize if we haven't already
        if not context.summary and len(context.messages) > 5:
            await self.summarize_conversation(context.conversation_id)
        
        # If still too large, truncate old messages (keep system + recent messages)
        if context.get_token_count() > self.max_context_tokens:
            await self._truncate_old_messages(context)
    
    async def _truncate_old_messages(self, context: ConversationContext):
        """Truncate old messages to fit context window"""
        system_messages = [msg for msg in context.messages if msg.role == MessageRole.SYSTEM]
        other_messages = [msg for msg in context.messages if msg.role != MessageRole.SYSTEM]
        
        # Keep system messages and work backwards from recent messages
        keep_messages = system_messages.copy()
        current_tokens = sum(msg.tokens or 0 for msg in keep_messages)
        
        # Add recent messages until we hit the limit
        for message in reversed(other_messages):
            message_tokens = message.tokens or 0
            if current_tokens + message_tokens <= self.max_context_tokens * 0.8:  # Leave some buffer
                keep_messages.insert(-len(system_messages) if system_messages else 0, message)
                current_tokens += message_tokens
            else:
                break
        
        # Update conversation with truncated messages
        removed_count = len(context.messages) - len(keep_messages)
        context.messages = keep_messages
        context.total_tokens = current_tokens
        
        logger.info(f"Truncated {removed_count} old messages from {context.conversation_id}")
    
    async def _select_messages_for_context(
        self, 
        messages: List[ConversationMessage], 
        max_tokens: int
    ) -> List[ConversationMessage]:
        """Select messages that fit within token limit"""
        selected = []
        current_tokens = 0
        
        # Work backwards from most recent
        for message in reversed(messages):
            message_tokens = message.tokens or await self._estimate_tokens(message.content)
            if current_tokens + message_tokens <= max_tokens:
                selected.insert(0, message)
                current_tokens += message_tokens
            else:
                break
        
        return selected
    
    async def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        # Simple estimation: ~4 characters per token
        # In production, use tiktoken or similar for accurate counting
        return len(text) // 4 + 1
    
    def _generate_conversation_id(self, client_id: str, session_id: str) -> str:
        """Generate unique conversation ID"""
        timestamp = datetime.now(timezone.utc).isoformat()
        content = f"{client_id}:{session_id}:{timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    async def _get_system_message(self, client_id: str) -> Optional[str]:
        """Get system message for client"""
        # This would typically load from client configuration
        # For now, return a default message
        return (
            "You are a helpful customer service assistant. "
            "Be polite, professional, and try to resolve customer issues efficiently. "
            "If you cannot help with something, offer to escalate to a human agent."
        )
    
    async def _cleanup_old_conversations(self):
        """Clean up old conversations from memory"""
        if len(self.active_conversations) <= self.max_active_conversations:
            return
        
        # Find conversations to archive (oldest first)
        conversations_to_archive = []
        now = datetime.now(timezone.utc)
        
        for conv_id, context in self.active_conversations.items():
            age = now - context.updated_at
            if age > self.conversation_timeout:
                conversations_to_archive.append(conv_id)
        
        # Archive old conversations
        for conv_id in conversations_to_archive:
            await self.archive_conversation(conv_id)
        
        # If still too many, archive oldest
        if len(self.active_conversations) > self.max_active_conversations:
            sorted_conversations = sorted(
                self.active_conversations.items(),
                key=lambda x: x[1].updated_at
            )
            
            excess_count = len(self.active_conversations) - self.max_active_conversations + 100  # Buffer
            for conv_id, _ in sorted_conversations[:excess_count]:
                await self.archive_conversation(conv_id)
        
        logger.info(f"Cleaned up {len(conversations_to_archive)} old conversations")
    
    async def _save_conversation_to_db(self, context: ConversationContext):
        """Save conversation to database"""
        # This would implement database persistence
        # For now, just log
        logger.info(f"Saving conversation {context.conversation_id} to database")
        pass
    
    async def _load_conversation_from_db(self, conversation_id: str) -> Optional[ConversationContext]:
        """Load conversation from database"""
        # This would implement database loading
        # For now, return None
        logger.info(f"Attempting to load conversation {conversation_id} from database")
        return None
    
    def get_active_conversation_count(self) -> int:
        """Get number of active conversations"""
        return len(self.active_conversations)
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        total_messages = sum(len(ctx.messages) for ctx in self.active_conversations.values())
        total_tokens = sum(ctx.total_tokens for ctx in self.active_conversations.values())
        
        states = {}
        for context in self.active_conversations.values():
            state = context.state.value
            states[state] = states.get(state, 0) + 1
        
        return {
            "active_conversations": len(self.active_conversations),
            "total_messages": total_messages,
            "total_tokens": total_tokens,
            "conversation_states": states,
            "max_context_tokens": self.max_context_tokens
        }


# Global conversation manager instance
_conversation_manager = None

def get_conversation_manager() -> ConversationManager:
    """Get or create global conversation manager instance"""
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = ConversationManager()
    return _conversation_manager