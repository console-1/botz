"""
Response Generation Service

Handles response generation with tone injection, client-specific customization,
and quality control for customer service bot interactions.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json
import re
from datetime import datetime, timezone

from .llm_service import get_llm_service, LLMService
from .conversation_service import get_conversation_manager, ConversationManager, ConversationContext
from ..core.config import settings

logger = logging.getLogger(__name__)


class ToneStyle(str, Enum):
    """Available tone styles for responses"""
    PROFESSIONAL = "professional"
    WARM = "warm"
    CASUAL = "casual"
    TECHNICAL = "technical"
    PLAYFUL = "playful"
    EMPATHETIC = "empathetic"
    CONCISE = "concise"
    DETAILED = "detailed"


class ResponseType(str, Enum):
    """Types of responses"""
    ANSWER = "answer"
    CLARIFICATION = "clarification"
    ESCALATION = "escalation"
    GREETING = "greeting"
    GOODBYE = "goodbye"
    ERROR = "error"


@dataclass
class ClientToneConfig:
    """Client-specific tone and style configuration"""
    primary_tone: ToneStyle = ToneStyle.PROFESSIONAL
    secondary_tone: Optional[ToneStyle] = None
    persona_description: str = "A helpful customer service representative"
    brand_voice: Optional[str] = None
    custom_instructions: List[str] = None
    prohibited_words: List[str] = None
    preferred_phrases: Dict[str, str] = None
    escalation_triggers: List[str] = None
    max_response_length: int = 500
    
    def __post_init__(self):
        if self.custom_instructions is None:
            self.custom_instructions = []
        if self.prohibited_words is None:
            self.prohibited_words = []
        if self.preferred_phrases is None:
            self.preferred_phrases = {}
        if self.escalation_triggers is None:
            self.escalation_triggers = []


@dataclass
class ResponseContext:
    """Context for response generation"""
    conversation_id: str
    client_id: str
    user_message: str
    conversation_history: List[Dict[str, str]]
    client_config: ClientToneConfig
    knowledge_context: Optional[str] = None
    confidence_threshold: float = 0.7
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class GeneratedResponse:
    """Generated response with metadata"""
    content: str
    response_type: ResponseType
    confidence_score: float
    tone_applied: ToneStyle
    tokens_used: int
    cost_usd: float
    processing_time_ms: int
    should_escalate: bool = False
    escalation_reason: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ResponseGenerator:
    """Generates contextual responses with tone injection"""
    
    def __init__(
        self, 
        llm_service: Optional[LLMService] = None,
        conversation_manager: Optional[ConversationManager] = None
    ):
        self.llm_service = llm_service or get_llm_service()
        self.conversation_manager = conversation_manager or get_conversation_manager()
        
        # Tone style prompts
        self.tone_prompts = {
            ToneStyle.PROFESSIONAL: (
                "Respond in a professional, courteous manner. Use formal language "
                "and maintain a respectful tone throughout."
            ),
            ToneStyle.WARM: (
                "Respond in a warm, friendly manner. Be welcoming and personable "
                "while maintaining professionalism."
            ),
            ToneStyle.CASUAL: (
                "Respond in a casual, conversational manner. Use friendly language "
                "that feels natural and approachable."
            ),
            ToneStyle.TECHNICAL: (
                "Respond with technical accuracy and precision. Use appropriate "
                "technical terminology and provide detailed explanations."
            ),
            ToneStyle.PLAYFUL: (
                "Respond in a playful, lighthearted manner while staying helpful. "
                "Use humor appropriately and keep the mood positive."
            ),
            ToneStyle.EMPATHETIC: (
                "Respond with empathy and understanding. Acknowledge the customer's "
                "feelings and show genuine concern for their situation."
            ),
            ToneStyle.CONCISE: (
                "Respond concisely and directly. Get to the point quickly while "
                "providing all necessary information."
            ),
            ToneStyle.DETAILED: (
                "Respond with comprehensive detail. Provide thorough explanations "
                "and cover all relevant aspects of the topic."
            )
        }
        
        # Response templates
        self.response_templates = {
            ResponseType.GREETING: [
                "Hello! How can I help you today?",
                "Hi there! What can I assist you with?",
                "Welcome! How may I be of service?"
            ],
            ResponseType.GOODBYE: [
                "Thank you for contacting us. Have a great day!",
                "Is there anything else I can help you with today?",
                "Thank you for your time. Feel free to reach out if you need further assistance."
            ],
            ResponseType.CLARIFICATION: [
                "Could you please provide more details about",
                "I'd be happy to help! Can you clarify",
                "To better assist you, could you tell me more about"
            ]
        }
    
    async def generate_response(self, context: ResponseContext) -> GeneratedResponse:
        """Generate a response with tone injection and quality control"""
        start_time = datetime.now(timezone.utc)
        
        try:
            # Build the system prompt with tone and client configuration
            system_prompt = await self._build_system_prompt(context.client_config)
            
            # Prepare messages for LLM
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add knowledge context if available
            if context.knowledge_context:
                messages.append({
                    "role": "system",
                    "content": f"Relevant information: {context.knowledge_context}"
                })
            
            # Add conversation history
            messages.extend(context.conversation_history)
            
            # Add current user message
            messages.append({"role": "user", "content": context.user_message})
            
            # Generate response using LLM service
            llm_response = await self.llm_service.generate_response(
                messages=messages,
                max_tokens=context.client_config.max_response_length,
                temperature=0.7
            )
            
            end_time = datetime.now(timezone.utc)
            processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Post-process the response
            processed_content = await self._post_process_response(
                llm_response.content, 
                context.client_config
            )
            
            # Calculate confidence score
            confidence_score = await self._calculate_confidence_score(
                context.user_message,
                processed_content,
                context.knowledge_context
            )
            
            # Determine response type
            response_type = await self._classify_response_type(
                context.user_message,
                processed_content
            )
            
            # Check if escalation is needed
            should_escalate, escalation_reason = await self._check_escalation_needed(
                context,
                confidence_score,
                processed_content
            )
            
            return GeneratedResponse(
                content=processed_content,
                response_type=response_type,
                confidence_score=confidence_score,
                tone_applied=context.client_config.primary_tone,
                tokens_used=llm_response.tokens_used,
                cost_usd=llm_response.cost_usd,
                processing_time_ms=processing_time_ms,
                should_escalate=should_escalate,
                escalation_reason=escalation_reason,
                metadata={
                    "model_used": llm_response.model,
                    "provider_used": llm_response.provider.value,
                    "original_response_length": len(llm_response.content),
                    "processed_response_length": len(processed_content)
                }
            )
            
        except Exception as e:
            logger.error(f"Response generation failed for {context.conversation_id}: {e}")
            
            # Return error response
            return GeneratedResponse(
                content="I apologize, but I'm having trouble processing your request right now. Please try again or contact our support team for assistance.",
                response_type=ResponseType.ERROR,
                confidence_score=0.0,
                tone_applied=context.client_config.primary_tone,
                tokens_used=0,
                cost_usd=0.0,
                processing_time_ms=int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000),
                should_escalate=True,
                escalation_reason="System error during response generation",
                metadata={"error": str(e)}
            )
    
    async def _build_system_prompt(self, client_config: ClientToneConfig) -> str:
        """Build system prompt with tone and client configuration"""
        prompt_parts = [
            f"You are {client_config.persona_description}.",
            self.tone_prompts[client_config.primary_tone]
        ]
        
        # Add secondary tone if specified
        if client_config.secondary_tone:
            prompt_parts.append(f"Also incorporate elements of: {self.tone_prompts[client_config.secondary_tone]}")
        
        # Add brand voice if specified
        if client_config.brand_voice:
            prompt_parts.append(f"Brand voice: {client_config.brand_voice}")
        
        # Add custom instructions
        if client_config.custom_instructions:
            prompt_parts.append("Additional instructions:")
            prompt_parts.extend(client_config.custom_instructions)
        
        # Add preferred phrases
        if client_config.preferred_phrases:
            phrases_text = ", ".join([f"use '{phrase}' instead of '{original}'" 
                                    for original, phrase in client_config.preferred_phrases.items()])
            prompt_parts.append(f"Preferred language: {phrases_text}")
        
        # Add prohibited words warning
        if client_config.prohibited_words:
            prohibited_text = ", ".join(client_config.prohibited_words)
            prompt_parts.append(f"Avoid using these words: {prohibited_text}")
        
        # Add response length guidance
        prompt_parts.append(f"Keep responses under {client_config.max_response_length} words unless more detail is specifically requested.")
        
        return " ".join(prompt_parts)
    
    async def _post_process_response(self, content: str, client_config: ClientToneConfig) -> str:
        """Post-process response to apply client-specific rules"""
        processed = content.strip()
        
        # Apply preferred phrase substitutions
        for original, preferred in client_config.preferred_phrases.items():
            processed = re.sub(r'\b' + re.escape(original) + r'\b', preferred, processed, flags=re.IGNORECASE)
        
        # Remove prohibited words (replace with alternatives)
        for prohibited in client_config.prohibited_words:
            if prohibited.lower() in processed.lower():
                # Simple replacement strategy - in production, use more sophisticated alternatives
                processed = re.sub(r'\b' + re.escape(prohibited) + r'\b', '[alternative]', processed, flags=re.IGNORECASE)
        
        # Ensure appropriate length
        if len(processed.split()) > client_config.max_response_length:
            words = processed.split()
            processed = " ".join(words[:client_config.max_response_length]) + "..."
        
        return processed
    
    async def _calculate_confidence_score(
        self, 
        user_message: str, 
        response: str, 
        knowledge_context: Optional[str]
    ) -> float:
        """Calculate confidence score for the response"""
        # Simple heuristic-based confidence scoring
        # In production, this could use a trained model
        
        confidence = 0.5  # Base confidence
        
        # Higher confidence if we have knowledge context
        if knowledge_context:
            confidence += 0.3
        
        # Check for uncertainty markers in response
        uncertainty_markers = [
            "i'm not sure", "i don't know", "maybe", "perhaps", 
            "i think", "probably", "might", "could be"
        ]
        
        uncertainty_count = sum(1 for marker in uncertainty_markers if marker in response.lower())
        confidence -= min(uncertainty_count * 0.1, 0.4)
        
        # Check for definitive language
        definitive_markers = [
            "certainly", "definitely", "absolutely", "exactly", 
            "specifically", "precisely", "clearly"
        ]
        
        definitive_count = sum(1 for marker in definitive_markers if marker in response.lower())
        confidence += min(definitive_count * 0.05, 0.2)
        
        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, confidence))
    
    async def _classify_response_type(self, user_message: str, response: str) -> ResponseType:
        """Classify the type of response"""
        user_lower = user_message.lower()
        response_lower = response.lower()
        
        # Greeting detection
        greeting_words = ["hello", "hi", "hey", "good morning", "good afternoon"]
        if any(word in user_lower for word in greeting_words):
            return ResponseType.GREETING
        
        # Goodbye detection
        goodbye_words = ["bye", "goodbye", "thanks", "thank you", "that's all"]
        if any(word in user_lower for word in goodbye_words):
            return ResponseType.GOODBYE
        
        # Clarification detection
        clarification_phrases = ["could you", "can you provide", "more details", "clarify"]
        if any(phrase in response_lower for phrase in clarification_phrases):
            return ResponseType.CLARIFICATION
        
        # Default to answer
        return ResponseType.ANSWER
    
    async def _check_escalation_needed(
        self, 
        context: ResponseContext, 
        confidence_score: float, 
        response: str
    ) -> Tuple[bool, Optional[str]]:
        """Check if response should be escalated to human"""
        
        # Check confidence threshold
        if confidence_score < context.confidence_threshold:
            return True, f"Low confidence score: {confidence_score:.2f}"
        
        # Check for escalation triggers in user message
        user_lower = context.user_message.lower()
        for trigger in context.client_config.escalation_triggers:
            if trigger.lower() in user_lower:
                return True, f"Escalation trigger detected: {trigger}"
        
        # Check for common escalation requests
        escalation_phrases = [
            "speak to manager", "human agent", "supervisor", "escalate",
            "this is urgent", "legal action", "complaint", "refund"
        ]
        
        for phrase in escalation_phrases:
            if phrase in user_lower:
                return True, f"Escalation phrase detected: {phrase}"
        
        # Check response for uncertainty
        uncertainty_phrases = [
            "i can't help", "beyond my capabilities", "contact support",
            "speak to a human", "transfer you"
        ]
        
        response_lower = response.lower()
        for phrase in uncertainty_phrases:
            if phrase in response_lower:
                return True, f"Response indicates escalation needed: {phrase}"
        
        return False, None
    
    async def generate_greeting(self, client_config: ClientToneConfig) -> str:
        """Generate a personalized greeting"""
        templates = self.response_templates[ResponseType.GREETING]
        base_greeting = templates[0]  # Could randomize or personalize
        
        # Apply tone
        if client_config.primary_tone == ToneStyle.WARM:
            return "Hello! I'm so glad you're here. How can I help make your day better?"
        elif client_config.primary_tone == ToneStyle.PROFESSIONAL:
            return "Good day. I'm here to assist you with any questions or concerns you may have."
        elif client_config.primary_tone == ToneStyle.CASUAL:
            return "Hey there! What's up? How can I help you out today?"
        elif client_config.primary_tone == ToneStyle.PLAYFUL:
            return "Hi! Ready to solve some problems together? What can I help you with? 😊"
        
        return base_greeting
    
    async def generate_goodbye(self, client_config: ClientToneConfig) -> str:
        """Generate a personalized goodbye"""
        templates = self.response_templates[ResponseType.GOODBYE]
        
        # Apply tone
        if client_config.primary_tone == ToneStyle.WARM:
            return "It was wonderful helping you today! Please don't hesitate to reach out anytime you need assistance."
        elif client_config.primary_tone == ToneStyle.PROFESSIONAL:
            return "Thank you for contacting us. We appreciate your business and hope we've resolved your inquiry satisfactorily."
        elif client_config.primary_tone == ToneStyle.CASUAL:
            return "Awesome, glad I could help! Hit me up anytime if you need anything else."
        elif client_config.primary_tone == ToneStyle.PLAYFUL:
            return "That was fun! Hope I made your day a little brighter. Come back anytime! 🌟"
        
        return templates[0]


# Global response generator instance
_response_generator = None

def get_response_generator() -> ResponseGenerator:
    """Get or create global response generator instance"""
    global _response_generator
    if _response_generator is None:
        _response_generator = ResponseGenerator()
    return _response_generator