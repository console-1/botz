"""
Escalation Service

Handles quality control, confidence scoring, and escalation logic for customer service interactions.
Provides intelligent escalation triggers and human handoff management.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone, timedelta
import json
import re

from .conversation_service import get_conversation_manager, ConversationManager, ConversationContext
from .response_service import GeneratedResponse, ResponseType, ToneStyle
from ..core.config import settings

logger = logging.getLogger(__name__)


class EscalationReason(str, Enum):
    """Reasons for escalation"""
    LOW_CONFIDENCE = "low_confidence"
    USER_REQUEST = "user_request"
    COMPLEX_ISSUE = "complex_issue"
    SENTIMENT_NEGATIVE = "sentiment_negative"
    TECHNICAL_ISSUE = "technical_issue"
    POLICY_QUESTION = "policy_question"
    BILLING_ISSUE = "billing_issue"
    SAFETY_CONCERN = "safety_concern"
    SYSTEM_ERROR = "system_error"
    REPEAT_ISSUE = "repeat_issue"


class EscalationPriority(str, Enum):
    """Escalation priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class SentimentScore(str, Enum):
    """Sentiment analysis scores"""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


@dataclass
class EscalationTrigger:
    """Configuration for escalation triggers"""
    keywords: List[str] = field(default_factory=list)
    confidence_threshold: float = 0.7
    sentiment_threshold: SentimentScore = SentimentScore.NEGATIVE
    max_conversation_turns: int = 10
    repeat_issue_threshold: int = 3
    enabled: bool = True
    priority: EscalationPriority = EscalationPriority.MEDIUM


@dataclass
class EscalationEvent:
    """Record of an escalation event"""
    escalation_id: str
    conversation_id: str
    client_id: str
    reason: EscalationReason
    priority: EscalationPriority
    confidence_score: float
    sentiment_score: SentimentScore
    user_message: str
    bot_response: str
    escalation_context: Dict[str, Any]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    human_agent_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/API"""
        return {
            "escalation_id": self.escalation_id,
            "conversation_id": self.conversation_id,
            "client_id": self.client_id,
            "reason": self.reason.value,
            "priority": self.priority.value,
            "confidence_score": self.confidence_score,
            "sentiment_score": self.sentiment_score.value,
            "user_message": self.user_message,
            "bot_response": self.bot_response,
            "escalation_context": self.escalation_context,
            "created_at": self.created_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolution_notes": self.resolution_notes,
            "human_agent_id": self.human_agent_id
        }


class QualityScorer:
    """Scores response quality and determines escalation needs"""
    
    def __init__(self):
        # Keyword patterns for different escalation reasons
        self.escalation_patterns = {
            EscalationReason.USER_REQUEST: [
                r"\b(human|agent|representative|person|manager|supervisor)\b",
                r"\b(speak to|talk to|connect me|transfer me)\b",
                r"\b(escalate|escalation)\b"
            ],
            EscalationReason.BILLING_ISSUE: [
                r"\b(bill|billing|charge|payment|invoice|refund|credit)\b",
                r"\b(overcharge|dispute|incorrect charge)\b",
                r"\b(cancel subscription|billing error)\b"
            ],
            EscalationReason.TECHNICAL_ISSUE: [
                r"\b(broken|not working|error|bug|glitch|crash)\b",
                r"\b(technical support|tech support|IT)\b",
                r"\b(server|database|system down)\b"
            ],
            EscalationReason.POLICY_QUESTION: [
                r"\b(policy|terms|conditions|legal|compliance)\b",
                r"\b(privacy|data protection|GDPR)\b",
                r"\b(terms of service|user agreement)\b"
            ],
            EscalationReason.SAFETY_CONCERN: [
                r"\b(safety|danger|harm|risk|threat)\b",
                r"\b(emergency|urgent|critical|immediate)\b",
                r"\b(security|breach|hack|unauthorized)\b"
            ]
        }
        
        # Sentiment analysis keywords
        self.sentiment_keywords = {
            SentimentScore.VERY_NEGATIVE: [
                "terrible", "awful", "horrible", "disgusting", "hate", "furious", 
                "outraged", "unacceptable", "ridiculous", "pathetic"
            ],
            SentimentScore.NEGATIVE: [
                "bad", "poor", "disappointing", "frustrated", "annoyed", "upset",
                "dissatisfied", "unhappy", "angry", "worried"
            ],
            SentimentScore.NEUTRAL: [
                "okay", "fine", "average", "normal", "standard", "regular"
            ],
            SentimentScore.POSITIVE: [
                "good", "nice", "helpful", "satisfied", "pleased", "happy",
                "thanks", "appreciate", "excellent"
            ],
            SentimentScore.VERY_POSITIVE: [
                "amazing", "fantastic", "wonderful", "perfect", "outstanding",
                "exceptional", "brilliant", "love", "impressed"
            ]
        }
    
    async def analyze_conversation_quality(
        self, 
        conversation_context: ConversationContext,
        generated_response: GeneratedResponse,
        user_message: str
    ) -> Tuple[float, SentimentScore, List[EscalationReason]]:
        """Analyze conversation quality and return metrics"""
        
        # Calculate overall confidence
        confidence_score = await self._calculate_overall_confidence(
            conversation_context, generated_response, user_message
        )
        
        # Analyze sentiment
        sentiment_score = await self._analyze_sentiment(user_message)
        
        # Identify escalation reasons
        escalation_reasons = await self._identify_escalation_reasons(
            conversation_context, generated_response, user_message, sentiment_score
        )
        
        return confidence_score, sentiment_score, escalation_reasons
    
    async def _calculate_overall_confidence(
        self,
        conversation_context: ConversationContext,
        generated_response: GeneratedResponse,
        user_message: str
    ) -> float:
        """Calculate overall confidence score"""
        base_confidence = generated_response.confidence_score
        
        # Adjust based on conversation length
        conversation_length = len(conversation_context.messages)
        if conversation_length > 8:  # Long conversations may indicate complexity
            base_confidence *= 0.9
        
        # Adjust based on response type
        if generated_response.response_type == ResponseType.CLARIFICATION:
            base_confidence *= 0.8  # Asking for clarification indicates uncertainty
        elif generated_response.response_type == ResponseType.ERROR:
            base_confidence = 0.0
        
        # Check for uncertainty markers in response
        uncertainty_markers = [
            "i'm not sure", "i don't know", "i'm not certain",
            "maybe", "perhaps", "i think", "probably", "might"
        ]
        
        uncertainty_count = sum(
            1 for marker in uncertainty_markers 
            if marker in generated_response.content.lower()
        )
        base_confidence -= min(uncertainty_count * 0.15, 0.5)
        
        return max(0.0, min(1.0, base_confidence))
    
    async def _analyze_sentiment(self, text: str) -> SentimentScore:
        """Analyze sentiment of user message"""
        text_lower = text.lower()
        
        # Simple keyword-based sentiment analysis
        # In production, use a proper sentiment analysis model
        sentiment_scores = {
            SentimentScore.VERY_NEGATIVE: 0,
            SentimentScore.NEGATIVE: 0,
            SentimentScore.NEUTRAL: 0,
            SentimentScore.POSITIVE: 0,
            SentimentScore.VERY_POSITIVE: 0
        }
        
        for sentiment, keywords in self.sentiment_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    sentiment_scores[sentiment] += 1
        
        # Find dominant sentiment
        max_score = max(sentiment_scores.values())
        if max_score == 0:
            return SentimentScore.NEUTRAL
        
        for sentiment, score in sentiment_scores.items():
            if score == max_score:
                return sentiment
        
        return SentimentScore.NEUTRAL
    
    async def _identify_escalation_reasons(
        self,
        conversation_context: ConversationContext,
        generated_response: GeneratedResponse,
        user_message: str,
        sentiment_score: SentimentScore
    ) -> List[EscalationReason]:
        """Identify reasons for potential escalation"""
        reasons = []
        user_lower = user_message.lower()
        
        # Check confidence score
        if generated_response.confidence_score < 0.7:
            reasons.append(EscalationReason.LOW_CONFIDENCE)
        
        # Check sentiment
        if sentiment_score in [SentimentScore.NEGATIVE, SentimentScore.VERY_NEGATIVE]:
            reasons.append(EscalationReason.SENTIMENT_NEGATIVE)
        
        # Check for specific escalation patterns
        for reason, patterns in self.escalation_patterns.items():
            for pattern in patterns:
                if re.search(pattern, user_lower):
                    reasons.append(reason)
                    break
        
        # Check conversation length (complex issue indicator)
        if len(conversation_context.messages) > 10:
            reasons.append(EscalationReason.COMPLEX_ISSUE)
        
        # Check for repeat issues (same user asking similar questions)
        if await self._is_repeat_issue(conversation_context):
            reasons.append(EscalationReason.REPEAT_ISSUE)
        
        # Check for system errors
        if generated_response.response_type == ResponseType.ERROR:
            reasons.append(EscalationReason.SYSTEM_ERROR)
        
        return list(set(reasons))  # Remove duplicates
    
    async def _is_repeat_issue(self, conversation_context: ConversationContext) -> bool:
        """Check if this is a repeat issue"""
        # Simple implementation - check if similar keywords appear multiple times
        user_messages = [
            msg.content for msg in conversation_context.messages 
            if msg.role.value == "user"
        ]
        
        if len(user_messages) < 3:
            return False
        
        # Extract keywords from messages and check for repetition
        all_keywords = []
        for message in user_messages:
            keywords = set(word.lower() for word in message.split() if len(word) > 3)
            all_keywords.extend(keywords)
        
        # If same keywords appear multiple times, it might be a repeat issue
        keyword_counts = {}
        for keyword in all_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        repeated_keywords = [k for k, v in keyword_counts.items() if v >= 3]
        return len(repeated_keywords) > 0


class EscalationManager:
    """Manages escalation events and human handoff process"""
    
    def __init__(self, conversation_manager: Optional[ConversationManager] = None):
        self.conversation_manager = conversation_manager or get_conversation_manager()
        self.quality_scorer = QualityScorer()
        self.active_escalations: Dict[str, EscalationEvent] = {}
        
        # Default escalation configurations per client
        self.default_escalation_config = {
            EscalationReason.LOW_CONFIDENCE: EscalationTrigger(
                confidence_threshold=0.6,
                priority=EscalationPriority.MEDIUM
            ),
            EscalationReason.USER_REQUEST: EscalationTrigger(
                priority=EscalationPriority.HIGH
            ),
            EscalationReason.SAFETY_CONCERN: EscalationTrigger(
                priority=EscalationPriority.CRITICAL
            ),
            EscalationReason.BILLING_ISSUE: EscalationTrigger(
                priority=EscalationPriority.HIGH
            ),
            EscalationReason.SENTIMENT_NEGATIVE: EscalationTrigger(
                sentiment_threshold=SentimentScore.VERY_NEGATIVE,
                priority=EscalationPriority.MEDIUM
            )
        }
    
    async def evaluate_escalation(
        self,
        conversation_id: str,
        generated_response: GeneratedResponse,
        user_message: str,
        client_escalation_config: Optional[Dict[EscalationReason, EscalationTrigger]] = None
    ) -> Optional[EscalationEvent]:
        """Evaluate if escalation is needed and create escalation event"""
        
        # Get conversation context
        conversation_context = await self.conversation_manager.get_conversation(conversation_id)
        if not conversation_context:
            logger.error(f"Conversation {conversation_id} not found for escalation evaluation")
            return None
        
        # Analyze conversation quality
        confidence_score, sentiment_score, escalation_reasons = await self.quality_scorer.analyze_conversation_quality(
            conversation_context, generated_response, user_message
        )
        
        # Use client config or default
        escalation_config = client_escalation_config or self.default_escalation_config
        
        # Check if any escalation reasons meet criteria
        triggered_reasons = []
        highest_priority = EscalationPriority.LOW
        
        for reason in escalation_reasons:
            if reason not in escalation_config:
                continue
            
            trigger_config = escalation_config[reason]
            if not trigger_config.enabled:
                continue
            
            should_escalate = False
            
            # Check specific criteria for each reason
            if reason == EscalationReason.LOW_CONFIDENCE:
                should_escalate = confidence_score < trigger_config.confidence_threshold
            elif reason == EscalationReason.SENTIMENT_NEGATIVE:
                should_escalate = (
                    sentiment_score == SentimentScore.VERY_NEGATIVE or
                    (sentiment_score == SentimentScore.NEGATIVE and 
                     trigger_config.sentiment_threshold == SentimentScore.NEGATIVE)
                )
            else:
                should_escalate = True  # Other reasons always trigger if detected
            
            if should_escalate:
                triggered_reasons.append(reason)
                if self._priority_level(trigger_config.priority) > self._priority_level(highest_priority):
                    highest_priority = trigger_config.priority
        
        # Create escalation event if any reasons were triggered
        if triggered_reasons:
            escalation_event = await self._create_escalation_event(
                conversation_context,
                triggered_reasons[0],  # Primary reason
                highest_priority,
                confidence_score,
                sentiment_score,
                user_message,
                generated_response.content,
                {
                    "all_reasons": [r.value for r in triggered_reasons],
                    "conversation_length": len(conversation_context.messages),
                    "response_type": generated_response.response_type.value,
                    "tokens_used": generated_response.tokens_used
                }
            )
            
            # Store active escalation
            self.active_escalations[escalation_event.escalation_id] = escalation_event
            
            # Mark conversation as escalated
            await self.conversation_manager.escalate_conversation(
                conversation_id,
                f"Escalated due to: {', '.join([r.value for r in triggered_reasons])}",
                {"escalation_id": escalation_event.escalation_id}
            )
            
            logger.info(f"Escalation created: {escalation_event.escalation_id} for {conversation_id}")
            return escalation_event
        
        return None
    
    async def _create_escalation_event(
        self,
        conversation_context: ConversationContext,
        primary_reason: EscalationReason,
        priority: EscalationPriority,
        confidence_score: float,
        sentiment_score: SentimentScore,
        user_message: str,
        bot_response: str,
        escalation_context: Dict[str, Any]
    ) -> EscalationEvent:
        """Create escalation event record"""
        escalation_id = self._generate_escalation_id(conversation_context.conversation_id)
        
        return EscalationEvent(
            escalation_id=escalation_id,
            conversation_id=conversation_context.conversation_id,
            client_id=conversation_context.client_id,
            reason=primary_reason,
            priority=priority,
            confidence_score=confidence_score,
            sentiment_score=sentiment_score,
            user_message=user_message,
            bot_response=bot_response,
            escalation_context=escalation_context
        )
    
    def _priority_level(self, priority: EscalationPriority) -> int:
        """Convert priority to numeric level for comparison"""
        levels = {
            EscalationPriority.LOW: 1,
            EscalationPriority.MEDIUM: 2,
            EscalationPriority.HIGH: 3,
            EscalationPriority.URGENT: 4,
            EscalationPriority.CRITICAL: 5
        }
        return levels.get(priority, 1)
    
    def _generate_escalation_id(self, conversation_id: str) -> str:
        """Generate unique escalation ID"""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"ESC_{conversation_id[:8]}_{timestamp}"
    
    async def resolve_escalation(
        self,
        escalation_id: str,
        resolution_notes: str,
        human_agent_id: Optional[str] = None
    ):
        """Mark escalation as resolved"""
        if escalation_id in self.active_escalations:
            escalation = self.active_escalations[escalation_id]
            escalation.resolved_at = datetime.now(timezone.utc)
            escalation.resolution_notes = resolution_notes
            escalation.human_agent_id = human_agent_id
            
            # Remove from active escalations
            del self.active_escalations[escalation_id]
            
            logger.info(f"Escalation {escalation_id} resolved by {human_agent_id or 'system'}")
    
    def get_active_escalations(
        self, 
        client_id: Optional[str] = None,
        priority: Optional[EscalationPriority] = None
    ) -> List[EscalationEvent]:
        """Get active escalations with optional filters"""
        escalations = list(self.active_escalations.values())
        
        if client_id:
            escalations = [e for e in escalations if e.client_id == client_id]
        
        if priority:
            escalations = [e for e in escalations if e.priority == priority]
        
        # Sort by priority and creation time
        escalations.sort(
            key=lambda e: (self._priority_level(e.priority), e.created_at),
            reverse=True
        )
        
        return escalations
    
    def get_escalation_stats(self) -> Dict[str, Any]:
        """Get escalation statistics"""
        active_escalations = list(self.active_escalations.values())
        
        stats = {
            "total_active": len(active_escalations),
            "by_priority": {},
            "by_reason": {},
            "by_client": {},
            "avg_age_hours": 0
        }
        
        if not active_escalations:
            return stats
        
        now = datetime.now(timezone.utc)
        total_age = timedelta()
        
        for escalation in active_escalations:
            # Priority stats
            priority = escalation.priority.value
            stats["by_priority"][priority] = stats["by_priority"].get(priority, 0) + 1
            
            # Reason stats
            reason = escalation.reason.value
            stats["by_reason"][reason] = stats["by_reason"].get(reason, 0) + 1
            
            # Client stats
            client = escalation.client_id
            stats["by_client"][client] = stats["by_client"].get(client, 0) + 1
            
            # Age calculation
            total_age += now - escalation.created_at
        
        stats["avg_age_hours"] = total_age.total_seconds() / 3600 / len(active_escalations)
        
        return stats


# Global escalation manager instance
_escalation_manager = None

def get_escalation_manager() -> EscalationManager:
    """Get or create global escalation manager instance"""
    global _escalation_manager
    if _escalation_manager is None:
        _escalation_manager = EscalationManager()
    return _escalation_manager