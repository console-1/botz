/**
 * Chat Widget Types
 * 
 * Core TypeScript interfaces for the customer service bot widget
 */

// Message Types
export interface ChatMessage {
  id: string;
  content: string;
  role: 'user' | 'assistant' | 'system';
  timestamp: Date;
  metadata?: Record<string, any>;
}

export interface MessageRequest {
  message: string;
  clientId: string;
  sessionId: string;
  conversationId?: string;
  userId?: string;
  metadata?: Record<string, any>;
}

export interface MessageResponse {
  message: string;
  conversationId: string;
  responseType: string;
  confidenceScore: number;
  shouldEscalate: boolean;
  escalationInfo?: EscalationInfo;
  processingTimeMs: number;
  tokensUsed: number;
  costUsd: number;
  metadata: Record<string, any>;
}

// Escalation Types
export interface EscalationInfo {
  escalationId: string;
  reason: string;
  priority: 'low' | 'medium' | 'high' | 'urgent' | 'critical';
  confidence_score: number;
  sentiment_score: string;
  created_at: string;
  status: string;
}

// Widget Configuration
export interface WidgetConfig {
  clientId: string;
  apiUrl: string;
  theme: WidgetTheme;
  branding: BrandingConfig;
  behavior: BehaviorConfig;
  position: WidgetPosition;
  customCss?: string;
  debug?: boolean;
}

export interface WidgetTheme {
  primaryColor: string;
  secondaryColor: string;
  backgroundColor: string;
  textColor: string;
  borderRadius: number;
  fontFamily: string;
  fontSize: number;
  darkMode?: boolean;
}

export interface BrandingConfig {
  companyName: string;
  logoUrl?: string;
  welcomeMessage?: string;
  placeholderText?: string;
  botName?: string;
  avatarUrl?: string;
  poweredByText?: string;
}

export interface BehaviorConfig {
  autoOpen?: boolean;
  openDelay?: number;
  showTypingIndicator?: boolean;
  enableSound?: boolean;
  enableEmojis?: boolean;
  maxMessageLength?: number;
  enableFileUpload?: boolean;
  enableVoiceInput?: boolean;
  persistConversation?: boolean;
}

export interface WidgetPosition {
  side: 'left' | 'right';
  bottom: number;
  horizontal: number;
}

// Widget State
export interface WidgetState {
  isOpen: boolean;
  isMinimized: boolean;
  conversationId?: string;
  sessionId: string;
  messages: ChatMessage[];
  isTyping: boolean;
  isConnected: boolean;
  hasUnreadMessages: boolean;
  unreadCount: number;
  currentUser?: UserInfo;
  escalationStatus?: EscalationStatus;
}

export interface UserInfo {
  id?: string;
  name?: string;
  email?: string;
  avatar?: string;
  metadata?: Record<string, any>;
}

export interface EscalationStatus {
  isEscalated: boolean;
  escalationId?: string;
  reason?: string;
  priority?: string;
  agentInfo?: AgentInfo;
}

export interface AgentInfo {
  id: string;
  name: string;
  avatar?: string;
  status: 'online' | 'busy' | 'away';
}

// API Types
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  statusCode: number;
}

export interface ConversationHistory {
  conversationId: string;
  messages: ChatMessage[];
  totalMessages: number;
  conversationState: string;
  createdAt: string;
  updatedAt: string;
}

// Event Types
export type WidgetEvent = 
  | 'widget:open'
  | 'widget:close'
  | 'widget:minimize'
  | 'widget:maximize'
  | 'message:send'
  | 'message:receive'
  | 'conversation:start'
  | 'conversation:end'
  | 'escalation:triggered'
  | 'escalation:resolved'
  | 'typing:start'
  | 'typing:stop'
  | 'connection:connect'
  | 'connection:disconnect'
  | 'error:occurred';

export interface WidgetEventData {
  type: WidgetEvent;
  payload?: any;
  timestamp: Date;
}

// Streaming Types
export interface StreamingChunk {
  content: string;
  done: boolean;
  conversationId?: string;
  metadata?: Record<string, any>;
}

// Animation Types
export interface AnimationConfig {
  duration: number;
  easing: string;
  delay?: number;
}

// Sound Types
export interface SoundConfig {
  enabled: boolean;
  volume: number;
  notificationSound?: string;
  messageSound?: string;
  errorSound?: string;
}

// Accessibility Types
export interface AccessibilityConfig {
  enableKeyboardNavigation: boolean;
  enableScreenReader: boolean;
  highContrast?: boolean;
  reducedMotion?: boolean;
  focusManagement?: boolean;
}

// Integration Types
export interface WebhookConfig {
  url: string;
  events: WidgetEvent[];
  headers?: Record<string, string>;
  retryAttempts?: number;
}

export interface AnalyticsConfig {
  enabled: boolean;
  trackingId?: string;
  customEvents?: string[];
  userIdentification?: boolean;
}

// Error Types
export interface WidgetError {
  code: string;
  message: string;
  details?: any;
  timestamp: Date;
  context?: Record<string, any>;
}

// Performance Types
export interface PerformanceMetrics {
  responseTime: number;
  renderTime: number;
  memoryUsage?: number;
  networkLatency?: number;
  errorRate?: number;
}

// Export utility types
export type MessageRole = ChatMessage['role'];
export type WidgetEventType = WidgetEvent;
export type EscalationPriority = EscalationInfo['priority'];
export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';
export type WidgetSize = 'compact' | 'normal' | 'expanded';
export type MessageStatus = 'sending' | 'sent' | 'delivered' | 'failed';