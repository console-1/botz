/**
 * Main Chat Widget Component
 * 
 * The root component that orchestrates the entire chat widget experience
 */

import React, { useEffect, useState, useCallback, useRef } from 'react';
import styled, { ThemeProvider, createGlobalStyle } from 'styled-components';
import { AnimatePresence, motion } from 'framer-motion';

import { 
  WidgetConfig, 
  WidgetState, 
  ChatMessage, 
  MessageRequest,
  WidgetEventData,
  ConnectionStatus 
} from '../types/chat';
import { ChatApiService } from '../services/chatApi';
import { WidgetStateService } from '../services/widgetState';

import ChatButton from './ChatButton';
import ChatWindow from './ChatWindow';
import ChatHeader from './ChatHeader';
import MessageList from './MessageList';
import MessageInput from './MessageInput';
import TypingIndicator from './TypingIndicator';
import EscalationBanner from './EscalationBanner';
import LoadingSpinner from './LoadingSpinner';
import ErrorBoundary from './ErrorBoundary';

// Global styles for the widget
const GlobalWidgetStyles = createGlobalStyle<{ theme: any }>`
  .chat-widget-overlay {
    position: fixed;
    z-index: 999999;
    pointer-events: none;
    font-family: ${props => props.theme.fontFamily};
    
    * {
      box-sizing: border-box;
    }
    
    /* Reset any inherited styles */
    button, input, textarea {
      font-family: inherit;
      font-size: inherit;
    }
  }
  
  .chat-widget-overlay * {
    pointer-events: auto;
  }
`;

// Styled components
const WidgetContainer = styled(motion.div)<{ position: any }>`
  position: fixed;
  ${props => props.position.side}: ${props => props.position.horizontal}px;
  bottom: ${props => props.position.bottom}px;
  z-index: 999999;
  font-family: ${props => props.theme.fontFamily};
  font-size: ${props => props.theme.fontSize}px;
`;

const ChatWindowContainer = styled(motion.div)`
  position: absolute;
  bottom: 80px;
  ${props => props.theme.position?.side === 'right' ? 'right: 0' : 'left: 0'};
  width: 380px;
  height: 600px;
  background: ${props => props.theme.backgroundColor};
  border-radius: ${props => props.theme.borderRadius}px;
  box-shadow: 0 5px 40px rgba(0, 0, 0, 0.16);
  overflow: hidden;
  display: flex;
  flex-direction: column;
`;

const ReconnectingBanner = styled.div`
  background: #f39c12;
  color: white;
  padding: 8px 16px;
  font-size: 12px;
  text-align: center;
  font-weight: 500;
`;

// Widget animation variants
const widgetVariants = {
  hidden: { 
    opacity: 0, 
    scale: 0.8,
    y: 20 
  },
  visible: { 
    opacity: 1, 
    scale: 1,
    y: 0,
    transition: {
      type: 'spring',
      stiffness: 300,
      damping: 30
    }
  },
  exit: { 
    opacity: 0, 
    scale: 0.8,
    y: 20,
    transition: {
      duration: 0.2
    }
  }
};

interface ChatWidgetProps {
  config: WidgetConfig;
  onStateChange?: (state: WidgetState) => void;
  onError?: (error: Error) => void;
}

export const ChatWidget: React.FC<ChatWidgetProps> = ({
  config,
  onStateChange,
  onError
}) => {
  // Services
  const [apiService] = useState(() => new ChatApiService({
    apiUrl: config.apiUrl,
    clientId: config.clientId,
    timeout: 30000
  }));
  
  const [stateService] = useState(() => new WidgetStateService(config));
  
  // State
  const [widgetState, setWidgetState] = useState<WidgetState>(stateService.getState());
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('disconnected');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Refs for managing focus and scrolling
  const messageListRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  
  // Theme based on config
  const theme = {
    ...config.theme,
    position: config.position
  };

  /**
   * Initialize widget
   */
  useEffect(() => {
    initializeWidget();
    setupEventListeners();
    
    return () => {
      cleanupEventListeners();
    };
  }, []);

  /**
   * Handle state changes
   */
  useEffect(() => {
    onStateChange?.(widgetState);
  }, [widgetState, onStateChange]);

  /**
   * Initialize widget and services
   */
  const initializeWidget = useCallback(async () => {
    try {
      setIsLoading(true);
      
      // Test API connection
      const isConnected = await apiService.testConnection();
      setConnectionStatus(isConnected ? 'connected' : 'disconnected');
      stateService.setConnectionStatus(isConnected ? 'connected' : 'disconnected');
      
      // Auto-open if configured
      if (config.behavior.autoOpen && config.behavior.openDelay) {
        setTimeout(() => {
          handleToggleWidget();
        }, config.behavior.openDelay);
      }
      
      setError(null);
    } catch (error) {
      console.error('[ChatWidget] Initialization failed:', error);
      setError('Failed to initialize chat widget');
      setConnectionStatus('error');
      onError?.(error as Error);
    } finally {
      setIsLoading(false);
    }
  }, [apiService, config, onError]);

  /**
   * Set up event listeners
   */
  const setupEventListeners = useCallback(() => {
    // Listen to state service events
    stateService.addEventListener('widget:open', handleWidgetOpened);
    stateService.addEventListener('widget:close', handleWidgetClosed);
    stateService.addEventListener('message:receive', handleMessageReceived);
    stateService.addEventListener('escalation:triggered', handleEscalationTriggered);
    
    // Listen to global events
    document.addEventListener('keydown', handleKeyDown);
    window.addEventListener('resize', handleWindowResize);
    
    // Set up periodic connection check
    const connectionCheck = setInterval(checkConnection, 30000); // Every 30 seconds
    
    return () => {
      clearInterval(connectionCheck);
    };
  }, [stateService]);

  /**
   * Clean up event listeners
   */
  const cleanupEventListeners = useCallback(() => {
    document.removeEventListener('keydown', handleKeyDown);
    window.removeEventListener('resize', handleWindowResize);
  }, []);

  /**
   * Check API connection
   */
  const checkConnection = useCallback(async () => {
    if (widgetState.isOpen) {
      const isConnected = await apiService.testConnection();
      const newStatus: ConnectionStatus = isConnected ? 'connected' : 'disconnected';
      
      if (newStatus !== connectionStatus) {
        setConnectionStatus(newStatus);
        stateService.setConnectionStatus(newStatus);
      }
    }
  }, [apiService, connectionStatus, widgetState.isOpen, stateService]);

  /**
   * Handle widget toggle
   */
  const handleToggleWidget = useCallback(() => {
    const newIsOpen = !widgetState.isOpen;
    
    stateService.setState({ isOpen: newIsOpen });
    setWidgetState(stateService.getState());
    
    if (newIsOpen) {
      // Clear unread messages when opening
      stateService.clearUnreadMessages();
      setWidgetState(stateService.getState());
      
      // Focus input after animation
      setTimeout(() => {
        inputRef.current?.focus();
      }, 300);
      
      // Start conversation if needed
      if (!widgetState.conversationId) {
        startConversation();
      }
    }
  }, [widgetState.isOpen, widgetState.conversationId, stateService]);

  /**
   * Start new conversation
   */
  const startConversation = useCallback(async () => {
    try {
      setIsLoading(true);
      
      const response = await apiService.startConversation(
        widgetState.sessionId,
        widgetState.currentUser?.id,
        { source: 'widget', userAgent: navigator.userAgent }
      );
      
      stateService.startConversation(response.conversationId);
      
      // Add greeting message
      const greetingMessage: ChatMessage = {
        id: `msg_${Date.now()}`,
        content: response.message,
        role: 'assistant',
        timestamp: new Date(),
        metadata: { type: 'greeting' }
      };
      
      stateService.addMessage(greetingMessage);
      setWidgetState(stateService.getState());
      
    } catch (error) {
      console.error('[ChatWidget] Failed to start conversation:', error);
      setError('Failed to start conversation');
      onError?.(error as Error);
    } finally {
      setIsLoading(false);
    }
  }, [apiService, widgetState.sessionId, widgetState.currentUser, stateService, onError]);

  /**
   * Send message
   */
  const handleSendMessage = useCallback(async (content: string) => {
    if (!content.trim() || !widgetState.conversationId) return;
    
    try {
      // Add user message immediately
      const userMessage: ChatMessage = {
        id: `msg_${Date.now()}_user`,
        content: content.trim(),
        role: 'user',
        timestamp: new Date(),
      };
      
      stateService.addMessage(userMessage);
      setWidgetState(stateService.getState());
      
      // Show typing indicator
      stateService.setTyping(true);
      setWidgetState(stateService.getState());
      
      // Prepare request
      const request: MessageRequest = {
        message: content.trim(),
        clientId: config.clientId,
        sessionId: widgetState.sessionId,
        conversationId: widgetState.conversationId,
        userId: widgetState.currentUser?.id,
        metadata: { source: 'widget' }
      };
      
      // Send message and get response
      const response = await apiService.sendMessage(request);
      
      // Hide typing indicator
      stateService.setTyping(false);
      
      // Add assistant response
      const assistantMessage: ChatMessage = {
        id: `msg_${Date.now()}_assistant`,
        content: response.message,
        role: 'assistant',
        timestamp: new Date(),
        metadata: {
          responseType: response.responseType,
          confidenceScore: response.confidenceScore,
          tokensUsed: response.tokensUsed,
          costUsd: response.costUsd
        }
      };
      
      stateService.addMessage(assistantMessage);
      
      // Handle escalation if needed
      if (response.shouldEscalate && response.escalationInfo) {
        stateService.setEscalationStatus({
          isEscalated: true,
          escalationId: response.escalationInfo.escalationId,
          reason: response.escalationInfo.reason,
          priority: response.escalationInfo.priority
        });
      }
      
      setWidgetState(stateService.getState());
      
    } catch (error) {
      console.error('[ChatWidget] Failed to send message:', error);
      
      // Hide typing indicator
      stateService.setTyping(false);
      
      // Add error message
      const errorMessage: ChatMessage = {
        id: `msg_${Date.now()}_error`,
        content: 'Sorry, I\'m having trouble responding right now. Please try again.',
        role: 'assistant',
        timestamp: new Date(),
        metadata: { type: 'error' }
      };
      
      stateService.addMessage(errorMessage);
      setWidgetState(stateService.getState());
      
      onError?.(error as Error);
    }
  }, [widgetState, config.clientId, apiService, stateService, onError]);

  /**
   * Event handlers
   */
  const handleWidgetOpened = useCallback((data: WidgetEventData) => {
    setWidgetState(stateService.getState());
  }, [stateService]);

  const handleWidgetClosed = useCallback((data: WidgetEventData) => {
    setWidgetState(stateService.getState());
  }, [stateService]);

  const handleMessageReceived = useCallback((data: WidgetEventData) => {
    setWidgetState(stateService.getState());
    
    // Scroll to bottom
    setTimeout(() => {
      messageListRef.current?.scrollTo({
        top: messageListRef.current.scrollHeight,
        behavior: 'smooth'
      });
    }, 100);
  }, [stateService]);

  const handleEscalationTriggered = useCallback((data: WidgetEventData) => {
    setWidgetState(stateService.getState());
    
    // Could trigger additional UI changes or notifications here
    console.log('[ChatWidget] Escalation triggered:', data.payload);
  }, [stateService]);

  const handleKeyDown = useCallback((event: KeyboardEvent) => {
    // Close widget on Escape key
    if (event.key === 'Escape' && widgetState.isOpen) {
      handleToggleWidget();
    }
  }, [widgetState.isOpen, handleToggleWidget]);

  const handleWindowResize = useCallback(() => {
    // Handle responsive behavior if needed
  }, []);

  /**
   * Render widget
   */
  if (error && !widgetState.isOpen) {
    return null; // Hide widget if there's an initialization error
  }

  return (
    <ErrorBoundary onError={onError}>
      <ThemeProvider theme={theme}>
        <GlobalWidgetStyles />
        <div className="chat-widget-overlay">
          <WidgetContainer
            position={config.position}
            initial="hidden"
            animate="visible"
            variants={widgetVariants}
          >
            {/* Chat Button */}
            <ChatButton
              isOpen={widgetState.isOpen}
              hasUnreadMessages={widgetState.hasUnreadMessages}
              unreadCount={widgetState.unreadCount}
              onClick={handleToggleWidget}
              theme={theme}
              branding={config.branding}
            />
            
            {/* Chat Window */}
            <AnimatePresence>
              {widgetState.isOpen && (
                <ChatWindowContainer
                  initial="hidden"
                  animate="visible"
                  exit="exit"
                  variants={widgetVariants}
                >
                  {/* Connection Status Banner */}
                  {connectionStatus !== 'connected' && (
                    <ReconnectingBanner>
                      {connectionStatus === 'connecting' && 'Connecting...'}
                      {connectionStatus === 'disconnected' && 'Reconnecting...'}
                      {connectionStatus === 'error' && 'Connection failed'}
                    </ReconnectingBanner>
                  )}
                  
                  {/* Escalation Banner */}
                  {widgetState.escalationStatus?.isEscalated && (
                    <EscalationBanner
                      escalationInfo={widgetState.escalationStatus}
                      onResolve={() => {
                        stateService.setEscalationStatus({ isEscalated: false });
                        setWidgetState(stateService.getState());
                      }}
                    />
                  )}
                  
                  {/* Chat Header */}
                  <ChatHeader
                    branding={config.branding}
                    isConnected={connectionStatus === 'connected'}
                    onClose={handleToggleWidget}
                    onMinimize={() => {
                      stateService.setState({ isMinimized: !widgetState.isMinimized });
                      setWidgetState(stateService.getState());
                    }}
                  />
                  
                  {/* Message List */}
                  <MessageList
                    ref={messageListRef}
                    messages={widgetState.messages}
                    isLoading={isLoading}
                    branding={config.branding}
                  />
                  
                  {/* Typing Indicator */}
                  {widgetState.isTyping && (
                    <TypingIndicator botName={config.branding.botName} />
                  )}
                  
                  {/* Loading Spinner */}
                  {isLoading && <LoadingSpinner />}
                  
                  {/* Message Input */}
                  <MessageInput
                    ref={inputRef}
                    onSendMessage={handleSendMessage}
                    disabled={isLoading || connectionStatus !== 'connected'}
                    placeholder={config.branding.placeholderText}
                    maxLength={config.behavior.maxMessageLength}
                  />
                </ChatWindowContainer>
              )}
            </AnimatePresence>
          </WidgetContainer>
        </div>
      </ThemeProvider>
    </ErrorBoundary>
  );
};

export default ChatWidget;