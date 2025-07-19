"""
Monitoring and alerting service for system health and performance
"""

import time
import psutil
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json

from sqlalchemy.orm import Session
from sqlalchemy import func, text
from pydantic import BaseModel

from app.core.database import get_db
from app.models.client import Client, UsageRecord
from app.models.conversation import Conversation, Message
from app.services.audit_logging import audit_logger, AuditEventType, AuditSeverity
from app.core.config import settings

class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"

class MetricType(str, Enum):
    """Types of metrics to monitor"""
    SYSTEM_HEALTH = "system_health"
    APPLICATION_PERFORMANCE = "application_performance"
    DATABASE_PERFORMANCE = "database_performance"
    API_METRICS = "api_metrics"
    BUSINESS_METRICS = "business_metrics"
    SECURITY_METRICS = "security_metrics"
    COMPLIANCE_METRICS = "compliance_metrics"

@dataclass
class Metric:
    """System metric with metadata"""
    name: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.SYSTEM_HEALTH

@dataclass
class Alert:
    """System alert"""
    id: str
    severity: AlertSeverity
    title: str
    description: str
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    labels: Dict[str, str] = field(default_factory=dict)

@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    metric_name: str
    condition: str  # "greater_than", "less_than", "equals", "not_equals"
    threshold: float
    severity: AlertSeverity
    description: str
    enabled: bool = True
    cooldown_minutes: int = 5  # Minimum time between alerts
    last_triggered: Optional[datetime] = None

class HealthStatus(str, Enum):
    """System health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"

class MonitoringService:
    """
    Comprehensive monitoring and alerting service for system health
    """
    
    def __init__(self):
        self.metrics_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alerts: List[Alert] = []
        self.alert_rules = self._get_default_alert_rules()
        self.health_checks: Dict[str, Callable] = {}
        self.monitoring_enabled = True
        
        # Register default health checks
        self._register_default_health_checks()
    
    def _get_default_alert_rules(self) -> List[AlertRule]:
        """Get default alert rules for system monitoring"""
        return [
            # System resource alerts
            AlertRule(
                name="high_cpu_usage",
                metric_name="cpu_usage_percent",
                condition="greater_than",
                threshold=80.0,
                severity=AlertSeverity.WARNING,
                description="CPU usage is above 80%"
            ),
            AlertRule(
                name="critical_cpu_usage",
                metric_name="cpu_usage_percent", 
                condition="greater_than",
                threshold=95.0,
                severity=AlertSeverity.CRITICAL,
                description="CPU usage is critically high (>95%)"
            ),
            AlertRule(
                name="high_memory_usage",
                metric_name="memory_usage_percent",
                condition="greater_than",
                threshold=85.0,
                severity=AlertSeverity.WARNING,
                description="Memory usage is above 85%"
            ),
            AlertRule(
                name="low_disk_space",
                metric_name="disk_usage_percent",
                condition="greater_than",
                threshold=90.0,
                severity=AlertSeverity.ERROR,
                description="Disk space is running low (<10% free)"
            ),
            
            # Database alerts
            AlertRule(
                name="high_db_connections",
                metric_name="database_connections",
                condition="greater_than",
                threshold=80.0,
                severity=AlertSeverity.WARNING,
                description="Database connection count is high"
            ),
            AlertRule(
                name="slow_db_queries",
                metric_name="avg_query_time_ms",
                condition="greater_than",
                threshold=1000.0,
                severity=AlertSeverity.WARNING,
                description="Database queries are running slowly"
            ),
            
            # API performance alerts
            AlertRule(
                name="high_api_response_time",
                metric_name="avg_response_time_ms",
                condition="greater_than",
                threshold=2000.0,
                severity=AlertSeverity.WARNING,
                description="API response times are high (>2s)"
            ),
            AlertRule(
                name="high_error_rate",
                metric_name="error_rate_percent",
                condition="greater_than",
                threshold=5.0,
                severity=AlertSeverity.ERROR,
                description="API error rate is above 5%"
            ),
            
            # Business metrics alerts
            AlertRule(
                name="low_conversation_success_rate",
                metric_name="conversation_success_rate",
                condition="less_than",
                threshold=85.0,
                severity=AlertSeverity.WARNING,
                description="Conversation success rate is below 85%"
            ),
            AlertRule(
                name="high_escalation_rate",
                metric_name="escalation_rate_percent",
                condition="greater_than",
                threshold=20.0,
                severity=AlertSeverity.WARNING,
                description="Escalation rate is above 20%"
            ),
            
            # Security alerts
            AlertRule(
                name="multiple_failed_logins",
                metric_name="failed_login_attempts",
                condition="greater_than",
                threshold=10.0,
                severity=AlertSeverity.ERROR,
                description="Multiple failed login attempts detected"
            ),
            AlertRule(
                name="suspicious_api_activity",
                metric_name="api_requests_per_minute",
                condition="greater_than",
                threshold=1000.0,
                severity=AlertSeverity.WARNING,
                description="Unusually high API request rate"
            )
        ]
    
    def _register_default_health_checks(self):
        """Register default health check functions"""
        self.health_checks.update({
            "database": self._check_database_health,
            "api_server": self._check_api_server_health,
            "external_services": self._check_external_services_health,
            "storage": self._check_storage_health
        })
    
    async def collect_system_metrics(self) -> List[Metric]:
        """Collect comprehensive system metrics"""
        metrics = []
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics.append(Metric(
                name="cpu_usage_percent",
                value=cpu_percent,
                unit="percent",
                metric_type=MetricType.SYSTEM_HEALTH
            ))
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics.append(Metric(
                name="memory_usage_percent",
                value=memory.percent,
                unit="percent",
                metric_type=MetricType.SYSTEM_HEALTH
            ))
            metrics.append(Metric(
                name="memory_available_gb",
                value=memory.available / (1024**3),
                unit="GB",
                metric_type=MetricType.SYSTEM_HEALTH
            ))
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            metrics.append(Metric(
                name="disk_usage_percent",
                value=disk_percent,
                unit="percent",
                metric_type=MetricType.SYSTEM_HEALTH
            ))
            metrics.append(Metric(
                name="disk_free_gb",
                value=disk.free / (1024**3),
                unit="GB",
                metric_type=MetricType.SYSTEM_HEALTH
            ))
            
            # Network metrics
            network = psutil.net_io_counters()
            metrics.append(Metric(
                name="network_bytes_sent",
                value=network.bytes_sent,
                unit="bytes",
                metric_type=MetricType.SYSTEM_HEALTH
            ))
            metrics.append(Metric(
                name="network_bytes_recv",
                value=network.bytes_recv,
                unit="bytes",
                metric_type=MetricType.SYSTEM_HEALTH
            ))
            
        except Exception as e:
            await audit_logger.log_system_error(
                error_type="metrics_collection_failure",
                error_message=str(e),
                component="monitoring_service",
                severity=AuditSeverity.MEDIUM
            )
        
        return metrics
    
    async def collect_database_metrics(self, db: Session) -> List[Metric]:
        """Collect database performance metrics"""
        metrics = []
        
        try:
            # Database connection count
            connection_count_query = text("""
                SELECT count(*) as connections 
                FROM pg_stat_activity 
                WHERE state = 'active'
            """)
            result = db.execute(connection_count_query).fetchone()
            if result:
                metrics.append(Metric(
                    name="database_connections",
                    value=float(result.connections),
                    unit="count",
                    metric_type=MetricType.DATABASE_PERFORMANCE
                ))
            
            # Database size
            db_size_query = text("""
                SELECT pg_size_pretty(pg_database_size(current_database())) as size
            """)
            result = db.execute(db_size_query).fetchone()
            
            # Table counts
            conversation_count = db.query(func.count(Conversation.id)).scalar()
            message_count = db.query(func.count(Message.id)).scalar()
            
            metrics.extend([
                Metric(
                    name="total_conversations",
                    value=float(conversation_count or 0),
                    unit="count",
                    metric_type=MetricType.DATABASE_PERFORMANCE
                ),
                Metric(
                    name="total_messages",
                    value=float(message_count or 0),
                    unit="count",
                    metric_type=MetricType.DATABASE_PERFORMANCE
                )
            ])
            
        except Exception as e:
            await audit_logger.log_system_error(
                error_type="database_metrics_failure",
                error_message=str(e),
                component="monitoring_service",
                severity=AuditSeverity.MEDIUM
            )
        
        return metrics
    
    async def collect_business_metrics(self, db: Session) -> List[Metric]:
        """Collect business-specific metrics"""
        metrics = []
        
        try:
            # Metrics for the last hour
            one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
            
            # Conversation metrics
            recent_conversations = db.query(func.count(Conversation.id)).filter(
                Conversation.created_at >= one_hour_ago
            ).scalar()
            
            escalated_conversations = db.query(func.count(Conversation.id)).filter(
                and_(
                    Conversation.created_at >= one_hour_ago,
                    Conversation.is_escalated == True
                )
            ).scalar()
            
            # Calculate escalation rate
            escalation_rate = 0.0
            if recent_conversations and recent_conversations > 0:
                escalation_rate = (escalated_conversations / recent_conversations) * 100
            
            # Response time metrics
            avg_response_time = db.query(func.avg(Message.processing_time_ms)).join(Conversation).filter(
                Conversation.created_at >= one_hour_ago
            ).scalar()
            
            metrics.extend([
                Metric(
                    name="conversations_last_hour",
                    value=float(recent_conversations or 0),
                    unit="count",
                    metric_type=MetricType.BUSINESS_METRICS
                ),
                Metric(
                    name="escalation_rate_percent",
                    value=escalation_rate,
                    unit="percent",
                    metric_type=MetricType.BUSINESS_METRICS
                ),
                Metric(
                    name="avg_response_time_ms",
                    value=float(avg_response_time or 0),
                    unit="milliseconds",
                    metric_type=MetricType.BUSINESS_METRICS
                )
            ])
            
        except Exception as e:
            await audit_logger.log_system_error(
                error_type="business_metrics_failure",
                error_message=str(e),
                component="monitoring_service",
                severity=AuditSeverity.MEDIUM
            )
        
        return metrics
    
    async def check_alert_rules(self, metrics: List[Metric]):
        """Check metrics against alert rules and trigger alerts"""
        
        current_time = datetime.now(timezone.utc)
        
        for rule in self.alert_rules:
            if not rule.enabled:
                continue
            
            # Check cooldown period
            if (rule.last_triggered and 
                current_time - rule.last_triggered < timedelta(minutes=rule.cooldown_minutes)):
                continue
            
            # Find matching metric
            matching_metric = None
            for metric in metrics:
                if metric.name == rule.metric_name:
                    matching_metric = metric
                    break
            
            if not matching_metric:
                continue
            
            # Check if rule condition is met
            alert_triggered = False
            
            if rule.condition == "greater_than" and matching_metric.value > rule.threshold:
                alert_triggered = True
            elif rule.condition == "less_than" and matching_metric.value < rule.threshold:
                alert_triggered = True
            elif rule.condition == "equals" and matching_metric.value == rule.threshold:
                alert_triggered = True
            elif rule.condition == "not_equals" and matching_metric.value != rule.threshold:
                alert_triggered = True
            
            if alert_triggered:
                alert = Alert(
                    id=f"{rule.name}_{int(current_time.timestamp())}",
                    severity=rule.severity,
                    title=f"Alert: {rule.name}",
                    description=rule.description,
                    metric_name=rule.metric_name,
                    current_value=matching_metric.value,
                    threshold_value=rule.threshold,
                    labels=matching_metric.labels
                )
                
                await self._trigger_alert(alert)
                rule.last_triggered = current_time
    
    async def _trigger_alert(self, alert: Alert):
        """Trigger an alert and send notifications"""
        self.alerts.append(alert)
        
        # Log the alert
        await audit_logger.log_event({
            "event_type": AuditEventType.SECURITY_ALERT if alert.severity == AlertSeverity.CRITICAL 
                         else AuditEventType.SYSTEM_ERROR,
            "severity": AuditSeverity.CRITICAL if alert.severity == AlertSeverity.CRITICAL 
                       else AuditSeverity.HIGH,
            "actor_type": "system",
            "actor_id": "monitoring_service",
            "action": f"alert_triggered_{alert.severity.value}",
            "description": f"Alert triggered: {alert.title}",
            "event_data": {
                "alert_id": alert.id,
                "metric_name": alert.metric_name,
                "current_value": alert.current_value,
                "threshold_value": alert.threshold_value,
                "severity": alert.severity.value
            }
        })
        
        # Send notifications (placeholder - implement actual notification channels)
        await self._send_alert_notification(alert)
    
    async def _send_alert_notification(self, alert: Alert):
        """Send alert notification via configured channels"""
        # In production, implement:
        # - Email notifications
        # - Slack/Discord webhooks
        # - SMS alerts for critical issues
        # - PagerDuty integration
        # - Custom webhook endpoints
        
        print(f"ALERT [{alert.severity.value.upper()}]: {alert.title}")
        print(f"Description: {alert.description}")
        print(f"Metric: {alert.metric_name} = {alert.current_value} (threshold: {alert.threshold_value})")
        print(f"Timestamp: {alert.timestamp}")
    
    async def perform_health_checks(self) -> Dict[str, HealthStatus]:
        """Perform all registered health checks"""
        health_status = {}
        
        for check_name, check_func in self.health_checks.items():
            try:
                status = await check_func()
                health_status[check_name] = status
            except Exception as e:
                health_status[check_name] = HealthStatus.UNHEALTHY
                await audit_logger.log_system_error(
                    error_type="health_check_failure",
                    error_message=str(e),
                    component=f"health_check_{check_name}",
                    severity=AuditSeverity.MEDIUM
                )
        
        return health_status
    
    async def _check_database_health(self) -> HealthStatus:
        """Check database connectivity and performance"""
        try:
            db = next(get_db())
            start_time = time.time()
            
            # Simple query to test connectivity
            result = db.execute(text("SELECT 1")).fetchone()
            query_time = (time.time() - start_time) * 1000  # Convert to ms
            
            db.close()
            
            if result and query_time < 100:  # < 100ms is healthy
                return HealthStatus.HEALTHY
            elif result and query_time < 500:  # < 500ms is degraded
                return HealthStatus.DEGRADED
            else:
                return HealthStatus.UNHEALTHY
                
        except Exception:
            return HealthStatus.CRITICAL
    
    async def _check_api_server_health(self) -> HealthStatus:
        """Check API server health"""
        # This would typically make HTTP requests to health endpoints
        # For now, assume healthy if we're running
        return HealthStatus.HEALTHY
    
    async def _check_external_services_health(self) -> HealthStatus:
        """Check external service dependencies"""
        # Would check:
        # - LLM providers (OpenAI, Anthropic, etc.)
        # - Vector database (Qdrant)
        # - Redis cache
        # - Email service
        return HealthStatus.HEALTHY
    
    async def _check_storage_health(self) -> HealthStatus:
        """Check storage system health"""
        try:
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            if disk_percent < 80:
                return HealthStatus.HEALTHY
            elif disk_percent < 90:
                return HealthStatus.DEGRADED
            elif disk_percent < 95:
                return HealthStatus.UNHEALTHY
            else:
                return HealthStatus.CRITICAL
        except Exception:
            return HealthStatus.CRITICAL
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system overview"""
        recent_metrics = {}
        for metric_name, metric_buffer in self.metrics_buffer.items():
            if metric_buffer:
                recent_metrics[metric_name] = {
                    "current_value": metric_buffer[-1].value,
                    "unit": metric_buffer[-1].unit,
                    "last_updated": metric_buffer[-1].timestamp.isoformat()
                }
        
        active_alerts = [alert for alert in self.alerts if not alert.resolved]
        
        return {
            "system_status": "monitoring_active" if self.monitoring_enabled else "monitoring_disabled",
            "metrics_count": len(recent_metrics),
            "active_alerts_count": len(active_alerts),
            "total_alerts_count": len(self.alerts),
            "recent_metrics": recent_metrics,
            "active_alerts": [
                {
                    "id": alert.id,
                    "severity": alert.severity.value,
                    "title": alert.title,
                    "timestamp": alert.timestamp.isoformat()
                }
                for alert in active_alerts[-10:]  # Last 10 alerts
            ],
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
    
    async def run_monitoring_cycle(self):
        """Run one complete monitoring cycle"""
        if not self.monitoring_enabled:
            return
        
        try:
            # Collect all metrics
            system_metrics = await self.collect_system_metrics()
            
            db = next(get_db())
            database_metrics = await self.collect_database_metrics(db)
            business_metrics = await self.collect_business_metrics(db)
            db.close()
            
            all_metrics = system_metrics + database_metrics + business_metrics
            
            # Store metrics in buffer
            for metric in all_metrics:
                self.metrics_buffer[metric.name].append(metric)
            
            # Check alert rules
            await self.check_alert_rules(all_metrics)
            
            # Perform health checks
            health_status = await self.perform_health_checks()
            
            # Log monitoring cycle completion
            if len(all_metrics) > 0:
                await audit_logger.log_event({
                    "event_type": AuditEventType.SYSTEM_START,  # Using available event type
                    "severity": AuditSeverity.LOW,
                    "actor_type": "system",
                    "actor_id": "monitoring_service",
                    "action": "monitoring_cycle_completed",
                    "description": f"Monitoring cycle completed: {len(all_metrics)} metrics collected",
                    "event_data": {
                        "metrics_collected": len(all_metrics),
                        "health_checks": health_status
                    }
                })
            
        except Exception as e:
            await audit_logger.log_system_error(
                error_type="monitoring_cycle_failure",
                error_message=str(e),
                component="monitoring_service",
                severity=AuditSeverity.HIGH
            )

# Global monitoring service instance
monitoring_service = MonitoringService()

def get_monitoring_service() -> MonitoringService:
    """Get monitoring service instance"""
    return monitoring_service

# Background monitoring task
async def run_monitoring_loop():
    """Run continuous monitoring loop"""
    while True:
        try:
            await monitoring_service.run_monitoring_cycle()
            await asyncio.sleep(60)  # Run every minute
        except Exception as e:
            print(f"Monitoring loop error: {e}")
            await asyncio.sleep(60)  # Continue after error