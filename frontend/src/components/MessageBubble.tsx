/**
 * Message Bubble Component
 * 
 * Individual message bubble with content, timestamp, and status indicators
 */

import React, { useState } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { ChatMessage, BrandingConfig, MessageRole } from '../types/chat';

// Styled components
const BubbleContainer = styled(motion.div)<{ role: MessageRole }>`
  display: flex;
  flex-direction: column;
  align-items: ${props => props.role === 'user' ? 'flex-end' : 'flex-start'};
  max-width: 100%;
  margin-bottom: 4px;
`;

const Bubble = styled.div<{ role: MessageRole; isError?: boolean }>`
  background: ${props => {
    if (props.isError) return '#e74c3c';
    return props.role === 'user' 
      ? props.theme.primaryColor 
      : '#f8f9fa';
  }};
  color: ${props => {
    if (props.isError) return 'white';
    return props.role === 'user' 
      ? 'white' 
      : props.theme.textColor;
  }};
  padding: 12px 16px;
  border-radius: ${props => {
    const radius = props.theme.borderRadius || 12;
    return props.role === 'user'
      ? `${radius}px ${radius}px 4px ${radius}px`
      : `${radius}px ${radius}px ${radius}px 4px`;
  }};
  max-width: 100%;
  word-wrap: break-word;
  position: relative;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
  
  /* Markdown support */
  p {
    margin: 0 0 8px 0;
    
    &:last-child {
      margin-bottom: 0;
    }
  }
  
  strong {
    font-weight: 600;
  }
  
  em {
    font-style: italic;
  }
  
  code {
    background: rgba(0, 0, 0, 0.1);
    padding: 2px 4px;
    border-radius: 3px;
    font-family: 'Monaco', 'Consolas', monospace;
    font-size: 0.9em;
  }
  
  pre {
    background: rgba(0, 0, 0, 0.05);
    padding: 8px;
    border-radius: 4px;
    overflow-x: auto;
    margin: 8px 0;
    
    code {
      background: none;
      padding: 0;
    }
  }
  
  a {
    color: ${props => props.role === 'user' ? 'rgba(255, 255, 255, 0.9)' : props.theme.primaryColor};
    text-decoration: underline;
    
    &:hover {
      opacity: 0.8;
    }
  }
  
  ul, ol {
    margin: 8px 0;
    padding-left: 20px;
  }
  
  li {
    margin-bottom: 4px;
  }
`;

const MessageMeta = styled.div<{ role: MessageRole }>`
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 4px;
  font-size: 11px;
  color: ${props => props.theme.textColor};
  opacity: 0.6;
  ${props => props.role === 'user' ? 'justify-content: flex-end;' : ''}
`;

const Timestamp = styled.span`
  white-space: nowrap;
`;

const StatusIndicator = styled.div<{ status: 'sending' | 'sent' | 'delivered' | 'failed' }>`
  width: 12px;
  height: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  
  svg {
    width: 100%;
    height: 100%;
    fill: currentColor;
  }
`;

const ConfidenceScore = styled.span<{ score: number }>`
  color: ${props => {
    if (props.score >= 0.8) return '#27ae60';
    if (props.score >= 0.6) return '#f39c12';
    return '#e74c3c';
  }};
  font-weight: 500;
`;

const ActionButtons = styled.div`
  display: flex;
  gap: 4px;
  margin-top: 8px;
`;

const ActionButton = styled.button`
  background: rgba(0, 0, 0, 0.1);
  border: 1px solid rgba(0, 0, 0, 0.2);
  border-radius: 12px;
  padding: 4px 8px;
  font-size: 11px;
  cursor: pointer;
  transition: all 0.2s ease;
  
  &:hover {
    background: rgba(0, 0, 0, 0.2);
  }
`;

const EscalationIndicator = styled.div`
  background: #e74c3c;
  color: white;
  font-size: 10px;
  padding: 2px 6px;
  border-radius: 8px;
  margin-top: 4px;
`;

const TypingDots = styled.div`
  display: flex;
  gap: 2px;
  padding: 4px 0;
  
  .dot {
    width: 4px;
    height: 4px;
    background: currentColor;
    border-radius: 50%;
    opacity: 0.4;
    animation: typing 1.4s infinite;
    
    &:nth-child(2) {
      animation-delay: 0.2s;
    }
    
    &:nth-child(3) {
      animation-delay: 0.4s;
    }
  }
  
  @keyframes typing {
    0%, 60%, 100% {
      opacity: 0.4;
      transform: scale(1);
    }
    30% {
      opacity: 1;
      transform: scale(1.2);
    }
  }
`;

// Status icons
const SendingIcon = () => (
  <svg viewBox="0 0 24 24">
    <path d="M12 2C13.1 2 14 2.9 14 4C14 5.1 13.1 6 12 6C10.9 6 10 5.1 10 4C10 2.9 10.9 2 12 2ZM12 20C10.9 20 10 19.1 10 18C10 16.9 10.9 16 12 16C13.1 16 14 16.9 14 18C14 19.1 13.1 20 12 20ZM6 12C6 10.9 6.9 10 8 10C9.1 10 10 10.9 10 12C10 13.1 9.1 14 8 14C6.9 14 6 13.1 6 12ZM16 12C16 13.1 16.9 14 18 14C19.1 14 20 13.1 20 12C20 10.9 19.1 10 18 10C16.9 10 16 10.9 16 12Z" />
  </svg>
);

const SentIcon = () => (
  <svg viewBox="0 0 24 24">
    <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z" />
  </svg>
);

const DeliveredIcon = () => (
  <svg viewBox="0 0 24 24">
    <path d="M18 7l-1.41-1.41-6.34 6.34 1.41 1.41L18 7zm4.24-1.41L11.66 16.17 7.48 12l-1.41 1.41L11.66 19l12-12-1.42-1.41zM.41 13.41L6 19l1.41-1.41L1.83 12 .41 13.41z" />
  </svg>
);

const FailedIcon = () => (
  <svg viewBox="0 0 24 24">
    <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12 19 6.41z" />
  </svg>
);

// Component props
interface MessageBubbleProps {
  message: ChatMessage;
  branding: BrandingConfig;
  isLatest?: boolean;
  onRetry?: () => void;
  onCopy?: () => void;
  onReact?: (reaction: string) => void;
}

// Helper functions
const formatTimestamp = (date: Date): string => {
  return date.toLocaleTimeString([], { 
    hour: '2-digit', 
    minute: '2-digit',
    hour12: false 
  });
};

const getMessageStatus = (message: ChatMessage): 'sending' | 'sent' | 'delivered' | 'failed' => {
  if (message.metadata?.status) {
    return message.metadata.status;
  }
  
  if (message.metadata?.type === 'error') {
    return 'failed';
  }
  
  return message.role === 'user' ? 'delivered' : 'sent';
};

const renderStatusIcon = (status: 'sending' | 'sent' | 'delivered' | 'failed') => {
  switch (status) {
    case 'sending':
      return <SendingIcon />;
    case 'sent':
      return <SentIcon />;
    case 'delivered':
      return <DeliveredIcon />;
    case 'failed':
      return <FailedIcon />;
    default:
      return null;
  }
};

// Animation variants
const bubbleVariants = {
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
      stiffness: 400,
      damping: 25
    }
  }
};

export const MessageBubble: React.FC<MessageBubbleProps> = ({
  message,
  branding,
  isLatest = false,
  onRetry,
  onCopy,
  onReact
}) => {
  const [showActions, setShowActions] = useState(false);
  
  const status = getMessageStatus(message);
  const isError = message.metadata?.type === 'error' || status === 'failed';
  const isEscalated = message.metadata?.escalated;
  const confidenceScore = message.metadata?.confidenceScore;

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      onCopy?.();
    } catch (error) {
      console.error('Failed to copy message:', error);
    }
  };

  const handleRetry = () => {
    onRetry?.();
  };

  return (
    <BubbleContainer
      role={message.role}
      variants={bubbleVariants}
      initial="hidden"
      animate="visible"
      onHoverStart={() => setShowActions(true)}
      onHoverEnd={() => setShowActions(false)}
    >
      <Bubble 
        role={message.role} 
        isError={isError}
      >
        {/* Special handling for typing indicator */}
        {message.metadata?.type === 'typing' ? (
          <TypingDots>
            <div className="dot" />
            <div className="dot" />
            <div className="dot" />
          </TypingDots>
        ) : (
          <div>
            {message.content}
          </div>
        )}
        
        {/* Action buttons for assistant messages */}
        {message.role === 'assistant' && showActions && (
          <ActionButtons>
            <ActionButton onClick={handleCopy} title="Copy message">
              📋
            </ActionButton>
            {onReact && (
              <>
                <ActionButton onClick={() => onReact('👍')} title="Helpful">
                  👍
                </ActionButton>
                <ActionButton onClick={() => onReact('👎')} title="Not helpful">
                  👎
                </ActionButton>
              </>
            )}
            {isError && onRetry && (
              <ActionButton onClick={handleRetry} title="Retry">
                🔄
              </ActionButton>
            )}
          </ActionButtons>
        )}
      </Bubble>
      
      {/* Message metadata */}
      <MessageMeta role={message.role}>
        <Timestamp>{formatTimestamp(message.timestamp)}</Timestamp>
        
        {/* Status indicator for user messages */}
        {message.role === 'user' && (
          <StatusIndicator status={status}>
            {renderStatusIcon(status)}
          </StatusIndicator>
        )}
        
        {/* Confidence score for assistant messages */}
        {message.role === 'assistant' && confidenceScore && (
          <ConfidenceScore score={confidenceScore}>
            {Math.round(confidenceScore * 100)}% confidence
          </ConfidenceScore>
        )}
      </MessageMeta>
      
      {/* Escalation indicator */}
      {isEscalated && (
        <EscalationIndicator>
          Escalated to human agent
        </EscalationIndicator>
      )}
    </BubbleContainer>
  );
};

export default MessageBubble;