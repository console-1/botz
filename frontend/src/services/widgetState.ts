/**
 * Widget State Management Service
 * 
 * Manages widget state, persistence, and session handling
 */

import {
  WidgetState,
  WidgetConfig,
  ChatMessage,
  WidgetEvent,
  WidgetEventData,
  UserInfo,
  EscalationStatus,
  ConnectionStatus
} from '../types/chat';

export class WidgetStateService {
  private state: WidgetState;
  private config: WidgetConfig;
  private eventListeners: Map<WidgetEvent, Set<(data: WidgetEventData) => void>>;
  private storageKey: string;
  private sessionStorageKey: string;

  constructor(config: WidgetConfig) {
    this.config = config;
    this.storageKey = `chat-widget-${config.clientId}`;
    this.sessionStorageKey = `chat-session-${config.clientId}`;
    this.eventListeners = new Map();

    // Initialize state
    this.state = this.loadState() || this.createInitialState();
    
    // Set up session management
    this.initializeSession();
    
    // Set up persistence
    if (config.behavior.persistConversation) {
      this.setupPersistence();
    }
  }

  /**
   * Create initial widget state
   */
  private createInitialState(): WidgetState {
    return {
      isOpen: this.config.behavior.autoOpen || false,
      isMinimized: false,
      sessionId: this.generateSessionId(),
      messages: [],
      isTyping: false,
      isConnected: false,
      hasUnreadMessages: false,
      unreadCount: 0,
    };
  }

  /**
   * Initialize session handling
   */
  private initializeSession(): void {
    // Check for existing session
    const existingSession = this.getSessionData();
    if (existingSession && existingSession.sessionId) {
      this.state.sessionId = existingSession.sessionId;
      this.state.conversationId = existingSession.conversationId;
    } else {
      // Create new session
      this.saveSessionData({
        sessionId: this.state.sessionId,
        timestamp: Date.now(),
      });
    }

    // Handle page visibility for session management
    document.addEventListener('visibilitychange', this.handleVisibilityChange.bind(this));
    
    // Handle page unload for cleanup
    window.addEventListener('beforeunload', this.handlePageUnload.bind(this));
  }

  /**
   * Set up state persistence
   */
  private setupPersistence(): void {
    // Auto-save state changes
    const originalSetState = this.setState.bind(this);
    this.setState = (updates: Partial<WidgetState>) => {
      originalSetState(updates);
      this.saveState();
    };

    // Periodic cleanup of old conversations
    setInterval(() => {
      this.cleanupOldConversations();
    }, 5 * 60 * 1000); // Every 5 minutes
  }

  /**
   * Get current state
   */
  getState(): WidgetState {
    return { ...this.state };
  }

  /**
   * Update state
   */
  setState(updates: Partial<WidgetState>): void {
    const oldState = { ...this.state };
    this.state = { ...this.state, ...updates };

    // Emit events for specific state changes
    if (updates.isOpen !== undefined && updates.isOpen !== oldState.isOpen) {
      this.emitEvent(updates.isOpen ? 'widget:open' : 'widget:close');
    }

    if (updates.isMinimized !== undefined && updates.isMinimized !== oldState.isMinimized) {
      this.emitEvent(updates.isMinimized ? 'widget:minimize' : 'widget:maximize');
    }

    if (updates.conversationId && updates.conversationId !== oldState.conversationId) {
      this.emitEvent('conversation:start', { conversationId: updates.conversationId });
    }
  }

  /**
   * Add message to conversation
   */
  addMessage(message: ChatMessage): void {
    const messages = [...this.state.messages, message];
    
    // Update unread count if widget is closed and message is from assistant
    let unreadCount = this.state.unreadCount;
    let hasUnreadMessages = this.state.hasUnreadMessages;
    
    if (!this.state.isOpen && message.role === 'assistant') {
      unreadCount += 1;
      hasUnreadMessages = true;
    }

    this.setState({
      messages,
      unreadCount,
      hasUnreadMessages,
    });

    this.emitEvent(message.role === 'user' ? 'message:send' : 'message:receive', message);
  }

  /**
   * Clear unread messages
   */
  clearUnreadMessages(): void {
    this.setState({
      hasUnreadMessages: false,
      unreadCount: 0,
    });
  }

  /**
   * Set typing indicator
   */
  setTyping(isTyping: boolean): void {
    this.setState({ isTyping });
    this.emitEvent(isTyping ? 'typing:start' : 'typing:stop');
  }

  /**
   * Set connection status
   */
  setConnectionStatus(status: ConnectionStatus): void {
    const isConnected = status === 'connected';
    this.setState({ isConnected });
    this.emitEvent(isConnected ? 'connection:connect' : 'connection:disconnect', { status });
  }

  /**
   * Set user information
   */
  setUser(user: UserInfo): void {
    this.setState({ currentUser: user });
  }

  /**
   * Set escalation status
   */
  setEscalationStatus(escalationStatus: EscalationStatus): void {
    this.setState({ escalationStatus });
    
    if (escalationStatus.isEscalated) {
      this.emitEvent('escalation:triggered', escalationStatus);
    } else {
      this.emitEvent('escalation:resolved', escalationStatus);
    }
  }

  /**
   * Start new conversation
   */
  startConversation(conversationId: string): void {
    this.setState({
      conversationId,
      messages: [],
      escalationStatus: undefined,
    });

    // Update session data
    this.saveSessionData({
      sessionId: this.state.sessionId,
      conversationId,
      timestamp: Date.now(),
    });
  }

  /**
   * End conversation
   */
  endConversation(): void {
    this.emitEvent('conversation:end', { conversationId: this.state.conversationId });
    
    this.setState({
      conversationId: undefined,
      messages: [],
      escalationStatus: undefined,
      isTyping: false,
    });

    // Clear session conversation
    this.saveSessionData({
      sessionId: this.state.sessionId,
      timestamp: Date.now(),
    });
  }

  /**
   * Reset widget state
   */
  resetState(): void {
    this.state = this.createInitialState();
    this.clearStorage();
    this.saveState();
  }

  /**
   * Generate unique session ID
   */
  private generateSessionId(): string {
    return `sess_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Load state from storage
   */
  private loadState(): WidgetState | null {
    if (!this.config.behavior.persistConversation) {
      return null;
    }

    try {
      const stored = localStorage.getItem(this.storageKey);
      if (stored) {
        const parsed = JSON.parse(stored);
        
        // Validate and clean up old data
        if (this.isValidStoredState(parsed)) {
          return {
            ...parsed,
            isOpen: false, // Always start closed
            isTyping: false,
            isConnected: false,
          };
        }
      }
    } catch (error) {
      console.warn('[WidgetState] Failed to load state from storage:', error);
    }

    return null;
  }

  /**
   * Save state to storage
   */
  private saveState(): void {
    if (!this.config.behavior.persistConversation) {
      return;
    }

    try {
      const stateToSave = {
        ...this.state,
        isOpen: false, // Don't persist open state
        isTyping: false,
        isConnected: false,
      };
      
      localStorage.setItem(this.storageKey, JSON.stringify(stateToSave));
    } catch (error) {
      console.warn('[WidgetState] Failed to save state to storage:', error);
    }
  }

  /**
   * Validate stored state structure
   */
  private isValidStoredState(state: any): boolean {
    return (
      state &&
      typeof state.sessionId === 'string' &&
      Array.isArray(state.messages) &&
      typeof state.hasUnreadMessages === 'boolean' &&
      typeof state.unreadCount === 'number'
    );
  }

  /**
   * Get session data
   */
  private getSessionData(): any {
    try {
      const stored = sessionStorage.getItem(this.sessionStorageKey);
      return stored ? JSON.parse(stored) : null;
    } catch (error) {
      console.warn('[WidgetState] Failed to load session data:', error);
      return null;
    }
  }

  /**
   * Save session data
   */
  private saveSessionData(data: any): void {
    try {
      sessionStorage.setItem(this.sessionStorageKey, JSON.stringify(data));
    } catch (error) {
      console.warn('[WidgetState] Failed to save session data:', error);
    }
  }

  /**
   * Clear all storage
   */
  private clearStorage(): void {
    try {
      localStorage.removeItem(this.storageKey);
      sessionStorage.removeItem(this.sessionStorageKey);
    } catch (error) {
      console.warn('[WidgetState] Failed to clear storage:', error);
    }
  }

  /**
   * Clean up old conversations
   */
  private cleanupOldConversations(): void {
    // Remove conversations older than 7 days
    const cutoff = Date.now() - (7 * 24 * 60 * 60 * 1000);
    
    try {
      const keys = Object.keys(localStorage);
      for (const key of keys) {
        if (key.startsWith(`chat-widget-${this.config.clientId}-conv-`)) {
          const stored = localStorage.getItem(key);
          if (stored) {
            const data = JSON.parse(stored);
            if (data.timestamp && data.timestamp < cutoff) {
              localStorage.removeItem(key);
            }
          }
        }
      }
    } catch (error) {
      console.warn('[WidgetState] Failed to cleanup old conversations:', error);
    }
  }

  /**
   * Handle page visibility change
   */
  private handleVisibilityChange(): void {
    if (document.hidden) {
      // Page is hidden - save state
      this.saveState();
    } else {
      // Page is visible - check for updates
      if (this.state.hasUnreadMessages && this.state.isOpen) {
        this.clearUnreadMessages();
      }
    }
  }

  /**
   * Handle page unload
   */
  private handlePageUnload(): void {
    this.saveState();
    this.emitEvent('widget:close');
  }

  /**
   * Add event listener
   */
  addEventListener(event: WidgetEvent, callback: (data: WidgetEventData) => void): void {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, new Set());
    }
    this.eventListeners.get(event)!.add(callback);
  }

  /**
   * Remove event listener
   */
  removeEventListener(event: WidgetEvent, callback: (data: WidgetEventData) => void): void {
    const listeners = this.eventListeners.get(event);
    if (listeners) {
      listeners.delete(callback);
    }
  }

  /**
   * Emit event
   */
  private emitEvent(type: WidgetEvent, payload?: any): void {
    const eventData: WidgetEventData = {
      type,
      payload,
      timestamp: new Date(),
    };

    const listeners = this.eventListeners.get(type);
    if (listeners) {
      listeners.forEach(callback => {
        try {
          callback(eventData);
        } catch (error) {
          console.error(`[WidgetState] Event listener error for ${type}:`, error);
        }
      });
    }

    // Log event in debug mode
    if (this.config.debug) {
      console.log(`[WidgetState] Event: ${type}`, eventData);
    }
  }

  /**
   * Get session info
   */
  getSessionInfo(): { sessionId: string; conversationId?: string; duration: number } {
    const sessionData = this.getSessionData();
    const startTime = sessionData?.timestamp || Date.now();
    
    return {
      sessionId: this.state.sessionId,
      conversationId: this.state.conversationId,
      duration: Date.now() - startTime,
    };
  }

  /**
   * Get conversation statistics
   */
  getConversationStats(): {
    messageCount: number;
    userMessages: number;
    assistantMessages: number;
    conversationDuration: number;
    hasEscalation: boolean;
  } {
    const messages = this.state.messages;
    const userMessages = messages.filter(m => m.role === 'user').length;
    const assistantMessages = messages.filter(m => m.role === 'assistant').length;
    
    const startTime = messages.length > 0 ? messages[0].timestamp.getTime() : Date.now();
    const endTime = messages.length > 0 ? messages[messages.length - 1].timestamp.getTime() : Date.now();
    
    return {
      messageCount: messages.length,
      userMessages,
      assistantMessages,
      conversationDuration: endTime - startTime,
      hasEscalation: Boolean(this.state.escalationStatus?.isEscalated),
    };
  }
}