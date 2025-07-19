"""
Usage tracking and analytics service for multi-tenant system
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
from uuid import UUID
import asyncio
from collections import defaultdict

from sqlalchemy.orm import Session
from sqlalchemy import and_, func, text
from pydantic import BaseModel

from app.core.database import get_db
from app.models.client import Client, UsageRecord
from app.models.conversation import Conversation, Message, UsageMetric
from app.core.config import settings

class UsageEvent(BaseModel):
    """Usage event for tracking"""
    client_id: UUID
    event_type: str  # conversation_start, message_sent, message_received, api_call, escalation
    timestamp: datetime
    metadata: Dict[str, Any] = {}

class UsageAggregator:
    """
    Service for tracking and aggregating usage metrics
    """
    
    def __init__(self):
        self.event_buffer: Dict[UUID, List[UsageEvent]] = defaultdict(list)
        self.buffer_size_limit = 1000
        self.flush_interval_seconds = 300  # 5 minutes
        
    def track_conversation_start(
        self,
        client_id: UUID,
        conversation_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Track the start of a new conversation"""
        event = UsageEvent(
            client_id=client_id,
            event_type="conversation_start",
            timestamp=datetime.now(timezone.utc),
            metadata=metadata or {}
        )
        self._buffer_event(event)
    
    def track_message_sent(
        self,
        client_id: UUID,
        conversation_id: str,
        message_id: str,
        role: str,
        token_count: Optional[int] = None,
        model_used: Optional[str] = None,
        processing_time_ms: Optional[int] = None,
        cost_usd: Optional[float] = None
    ):
        """Track a message being sent"""
        event = UsageEvent(
            client_id=client_id,
            event_type="message_sent",
            timestamp=datetime.now(timezone.utc),
            metadata={
                "conversation_id": conversation_id,
                "message_id": message_id,
                "role": role,
                "token_count": token_count,
                "model_used": model_used,
                "processing_time_ms": processing_time_ms,
                "cost_usd": cost_usd
            }
        )
        self._buffer_event(event)
    
    def track_api_call(
        self,
        client_id: UUID,
        endpoint: str,
        method: str,
        status_code: int,
        response_time_ms: int,
        api_key_id: Optional[UUID] = None
    ):
        """Track an API call"""
        event = UsageEvent(
            client_id=client_id,
            event_type="api_call",
            timestamp=datetime.now(timezone.utc),
            metadata={
                "endpoint": endpoint,
                "method": method,
                "status_code": status_code,
                "response_time_ms": response_time_ms,
                "api_key_id": str(api_key_id) if api_key_id else None,
                "success": 200 <= status_code < 300
            }
        )
        self._buffer_event(event)
    
    def track_escalation(
        self,
        client_id: UUID,
        conversation_id: str,
        escalation_reason: str,
        confidence_score: Optional[float] = None
    ):
        """Track a conversation escalation"""
        event = UsageEvent(
            client_id=client_id,
            event_type="escalation",
            timestamp=datetime.now(timezone.utc),
            metadata={
                "conversation_id": conversation_id,
                "escalation_reason": escalation_reason,
                "confidence_score": confidence_score
            }
        )
        self._buffer_event(event)
    
    def _buffer_event(self, event: UsageEvent):
        """Add event to buffer and flush if needed"""
        self.event_buffer[event.client_id].append(event)
        
        # Check if we need to flush
        if len(self.event_buffer[event.client_id]) >= self.buffer_size_limit:
            asyncio.create_task(self._flush_client_events(event.client_id))
    
    async def _flush_client_events(self, client_id: UUID):
        """Flush events for a specific client to database"""
        if client_id not in self.event_buffer or not self.event_buffer[client_id]:
            return
        
        events = self.event_buffer[client_id].copy()
        self.event_buffer[client_id].clear()
        
        try:
            db = next(get_db())
            await self._process_events(db, client_id, events)
            db.close()
        except Exception as e:
            # Log error and put events back in buffer
            print(f"Error flushing events for client {client_id}: {e}")
            self.event_buffer[client_id].extend(events)
    
    async def _process_events(self, db: Session, client_id: UUID, events: List[UsageEvent]):
        """Process and aggregate events into database records"""
        
        # Group events by hour for aggregation
        hourly_stats = defaultdict(lambda: {
            "conversations": set(),
            "messages": 0,
            "tokens": 0,
            "api_calls": 0,
            "successful_api_calls": 0,
            "failed_api_calls": 0,
            "escalations": 0,
            "response_times": [],
            "costs": 0.0
        })
        
        for event in events:
            hour_key = event.timestamp.replace(minute=0, second=0, microsecond=0)
            stats = hourly_stats[hour_key]
            
            if event.event_type == "conversation_start":
                stats["conversations"].add(event.metadata.get("conversation_id"))
            
            elif event.event_type == "message_sent":
                stats["messages"] += 1
                if event.metadata.get("token_count"):
                    stats["tokens"] += event.metadata["token_count"]
                if event.metadata.get("processing_time_ms"):
                    stats["response_times"].append(event.metadata["processing_time_ms"])
                if event.metadata.get("cost_usd"):
                    stats["costs"] += event.metadata["cost_usd"]
            
            elif event.event_type == "api_call":
                stats["api_calls"] += 1
                if event.metadata.get("success"):
                    stats["successful_api_calls"] += 1
                else:
                    stats["failed_api_calls"] += 1
                if event.metadata.get("response_time_ms"):
                    stats["response_times"].append(event.metadata["response_time_ms"])
            
            elif event.event_type == "escalation":
                stats["escalations"] += 1
        
        # Create or update usage records
        for hour, stats in hourly_stats.items():
            period_start = hour
            period_end = hour + timedelta(hours=1)
            
            # Check if record exists
            existing_record = db.query(UsageRecord).filter(
                and_(
                    UsageRecord.client_id == client_id,
                    UsageRecord.period_start == period_start,
                    UsageRecord.period_end == period_end
                )
            ).first()
            
            if existing_record:
                # Update existing record
                existing_record.total_conversations += len(stats["conversations"])
                existing_record.total_messages += stats["messages"]
                existing_record.total_tokens_used += stats["tokens"]
                existing_record.api_calls_made += stats["api_calls"]
                existing_record.successful_responses += stats["successful_api_calls"]
                existing_record.failed_responses += stats["failed_api_calls"]
                existing_record.escalations_triggered += stats["escalations"]
                existing_record.estimated_cost_usd += int(stats["costs"] * 100)  # Convert to cents
                
                # Update average response time
                if stats["response_times"]:
                    new_avg = sum(stats["response_times"]) / len(stats["response_times"])
                    existing_record.avg_response_time_ms = int(
                        (existing_record.avg_response_time_ms + new_avg) / 2
                    )
            else:
                # Create new record
                avg_response_time = 0
                if stats["response_times"]:
                    avg_response_time = int(sum(stats["response_times"]) / len(stats["response_times"]))
                
                new_record = UsageRecord(
                    client_id=client_id,
                    period_start=period_start,
                    period_end=period_end,
                    total_conversations=len(stats["conversations"]),
                    total_messages=stats["messages"],
                    total_tokens_used=stats["tokens"],
                    api_calls_made=stats["api_calls"],
                    successful_responses=stats["successful_api_calls"],
                    failed_responses=stats["failed_api_calls"],
                    avg_response_time_ms=avg_response_time,
                    escalations_triggered=stats["escalations"],
                    estimated_cost_usd=int(stats["costs"] * 100)  # Convert to cents
                )
                db.add(new_record)
        
        db.commit()
    
    async def flush_all_events(self):
        """Flush all buffered events to database"""
        for client_id in list(self.event_buffer.keys()):
            await self._flush_client_events(client_id)
    
    def get_client_usage_summary(
        self,
        db: Session,
        client_id: UUID,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get usage summary for a client in a date range"""
        
        usage_records = db.query(UsageRecord).filter(
            and_(
                UsageRecord.client_id == client_id,
                UsageRecord.period_start >= start_date,
                UsageRecord.period_end <= end_date
            )
        ).all()
        
        if not usage_records:
            return {
                "total_conversations": 0,
                "total_messages": 0,
                "total_tokens": 0,
                "total_api_calls": 0,
                "total_cost_usd": 0.0,
                "avg_response_time_ms": 0,
                "escalation_rate": 0.0,
                "success_rate": 0.0
            }
        
        total_conversations = sum(r.total_conversations for r in usage_records)
        total_messages = sum(r.total_messages for r in usage_records)
        total_tokens = sum(r.total_tokens_used for r in usage_records)
        total_api_calls = sum(r.api_calls_made for r in usage_records)
        total_cost_cents = sum(r.estimated_cost_usd for r in usage_records)
        total_escalations = sum(r.escalations_triggered for r in usage_records)
        total_successful = sum(r.successful_responses for r in usage_records)
        total_failed = sum(r.failed_responses for r in usage_records)
        
        # Calculate averages
        avg_response_time = 0
        if usage_records:
            response_times = [r.avg_response_time_ms for r in usage_records if r.avg_response_time_ms > 0]
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
        
        escalation_rate = 0.0
        if total_conversations > 0:
            escalation_rate = total_escalations / total_conversations
        
        success_rate = 0.0
        if total_api_calls > 0:
            success_rate = total_successful / total_api_calls
        
        return {
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "total_tokens": total_tokens,
            "total_api_calls": total_api_calls,
            "total_cost_usd": total_cost_cents / 100.0,
            "avg_response_time_ms": avg_response_time,
            "escalation_rate": escalation_rate,
            "success_rate": success_rate
        }
    
    def get_usage_trends(
        self,
        db: Session,
        client_id: UUID,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get daily usage trends for a client"""
        
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        # Query usage records grouped by day
        daily_query = text("""
            SELECT 
                DATE(period_start) as usage_date,
                SUM(total_conversations) as conversations,
                SUM(total_messages) as messages,
                SUM(total_tokens_used) as tokens,
                SUM(api_calls_made) as api_calls,
                AVG(avg_response_time_ms) as avg_response_time,
                SUM(escalations_triggered) as escalations,
                SUM(estimated_cost_usd) as cost_cents
            FROM usage_records 
            WHERE client_id = :client_id 
                AND period_start >= :start_date 
                AND period_start <= :end_date
            GROUP BY DATE(period_start)
            ORDER BY usage_date
        """)
        
        result = db.execute(daily_query, {
            "client_id": str(client_id),
            "start_date": start_date,
            "end_date": end_date
        })
        
        trends = []
        for row in result:
            trends.append({
                "date": row.usage_date,
                "conversations": row.conversations or 0,
                "messages": row.messages or 0,
                "tokens": row.tokens or 0,
                "api_calls": row.api_calls or 0,
                "avg_response_time_ms": row.avg_response_time or 0,
                "escalations": row.escalations or 0,
                "cost_usd": (row.cost_cents or 0) / 100.0
            })
        
        return trends

# Global usage aggregator instance
usage_aggregator = UsageAggregator()

def get_usage_aggregator() -> UsageAggregator:
    """Get the global usage aggregator instance"""
    return usage_aggregator

# Background task to periodically flush events
async def periodic_flush_task():
    """Background task to flush events periodically"""
    while True:
        try:
            await asyncio.sleep(usage_aggregator.flush_interval_seconds)
            await usage_aggregator.flush_all_events()
        except Exception as e:
            print(f"Error in periodic flush task: {e}")
            await asyncio.sleep(60)  # Wait 1 minute before retrying