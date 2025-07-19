/**
 * Message List Component
 * 
 * Displays the conversation history with messages from user and assistant
 */

import React, { forwardRef, useEffect, useRef } from 'react';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { ChatMessage, BrandingConfig, MessageRole } from '../types/chat';
import MessageBubble from './MessageBubble';
import LoadingSpinner from './LoadingSpinner';

// Styled components
const Container = styled.div`
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
  padding: 16px 20px;
  background: ${props => props.theme.backgroundColor};
  position: relative;
  
  /* Custom scrollbar */
  &::-webkit-scrollbar {
    width: 6px;
  }
  
  &::-webkit-scrollbar-track {
    background: transparent;
  }
  
  &::-webkit-scrollbar-thumb {
    background: rgba(0, 0, 0, 0.2);
    border-radius: 3px;
  }
  
  &::-webkit-scrollbar-thumb:hover {
    background: rgba(0, 0, 0, 0.3);
  }
`;

const MessageContainer = styled(motion.div)<{ role: MessageRole }>`
  display: flex;
  align-items: flex-end;
  margin-bottom: 16px;
  flex-direction: ${props => props.role === 'user' ? 'row-reverse' : 'row'};
  
  &:last-child {
    margin-bottom: 8px;
  }
`;

const Avatar = styled.div<{ role: MessageRole; avatarUrl?: string }>`
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background: ${props => {
    if (props.avatarUrl) {
      return `url(${props.avatarUrl}) center/cover`;
    }
    return props.role === 'user' 
      ? props.theme.primaryColor 
      : props.theme.secondaryColor;
  }};
  display: flex;
  align-items: center;
  justify-content: center;
  margin: ${props => props.role === 'user' ? '0 0 0 8px' : '0 8px 0 0'};
  flex-shrink: 0;
  color: white;
  font-size: 12px;
  font-weight: 600;
  border: 2px solid rgba(255, 255, 255, 0.1);
`;

const MessagesGroup = styled.div<{ role: MessageRole }>`
  max-width: 280px;
  display: flex;
  flex-direction: column;
  align-items: ${props => props.role === 'user' ? 'flex-end' : 'flex-start'};
`;

const WelcomeMessage = styled(motion.div)`
  text-align: center;
  padding: 32px 20px;
  color: ${props => props.theme.textColor};
  opacity: 0.7;
`;

const WelcomeTitle = styled.h3`
  margin: 0 0 8px 0;
  font-size: 18px;
  font-weight: 600;
  color: ${props => props.theme.primaryColor};
`;

const WelcomeText = styled.p`
  margin: 0;
  font-size: 14px;
  line-height: 1.4;
`;

const EmptyState = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  padding: 40px 20px;
  text-align: center;
  color: ${props => props.theme.textColor};
  opacity: 0.6;
`;

const EmptyStateIcon = styled.div`
  width: 64px;
  height: 64px;
  background: ${props => props.theme.primaryColor};
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 16px;
  opacity: 0.3;
`;

const ScrollToBottomButton = styled(motion.button)`
  position: absolute;
  bottom: 16px;
  left: 50%;
  transform: translateX(-50%);
  background: ${props => props.theme.primaryColor};
  color: white;
  border: none;
  border-radius: 20px;
  padding: 8px 16px;
  font-size: 12px;
  cursor: pointer;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
  display: flex;
  align-items: center;
  gap: 4px;
  z-index: 10;
  
  &:hover {
    transform: translateX(-50%) scale(1.05);
  }
`;

const TimestampSeparator = styled.div`
  text-align: center;
  margin: 16px 0;
  position: relative;
  
  &::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 0;
    right: 0;
    height: 1px;
    background: rgba(0, 0, 0, 0.1);
  }
`;

const TimestampText = styled.span`
  background: ${props => props.theme.backgroundColor};
  padding: 0 12px;
  font-size: 12px;
  color: ${props => props.theme.textColor};
  opacity: 0.5;
  position: relative;
  z-index: 1;
`;

// Icons
const ChatIcon: React.FC<{ size?: number }> = ({ size = 24 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
    <path d="M20 2H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h4l4 4 4-4h4c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z"/>
  </svg>
);

const ArrowDownIcon: React.FC<{ size?: number }> = ({ size = 16 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
    <path d="M7.41 8.59L12 13.17l4.59-4.58L18 10l-6 6-6-6 1.41-1.41z"/>
  </svg>
);

const UserIcon: React.FC<{ size?: number }> = ({ size = 16 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
  </svg>
);

// Component props
interface MessageListProps {
  messages: ChatMessage[];
  isLoading?: boolean;
  branding: BrandingConfig;
  onScrollToBottom?: () => void;
  className?: string;
}

// Helper functions
const formatTimestamp = (date: Date): string => {
  const now = new Date();
  const diff = now.getTime() - date.getTime();
  const hours = diff / (1000 * 60 * 60);
  
  if (hours < 1) {
    return 'Just now';
  } else if (hours < 24) {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } else {
    return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
  }
};

const shouldShowTimestampSeparator = (
  current: ChatMessage, 
  previous?: ChatMessage
): boolean => {
  if (!previous) return false;
  
  const timeDiff = current.timestamp.getTime() - previous.timestamp.getTime();
  const hoursDiff = timeDiff / (1000 * 60 * 60);
  
  return hoursDiff > 1; // Show separator if more than 1 hour between messages
};

const getAvatarContent = (role: MessageRole, branding: BrandingConfig): string => {
  if (role === 'user') {
    return 'You';
  }
  
  if (branding.botName) {
    return branding.botName.charAt(0).toUpperCase();
  }
  
  return 'AI';
};

// Animation variants
const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1
    }
  }
};

const messageVariants = {
  hidden: { opacity: 0, y: 20, scale: 0.8 },
  visible: { 
    opacity: 1, 
    y: 0, 
    scale: 1,
    transition: {
      type: 'spring',
      stiffness: 400,
      damping: 25
    }
  }
};

const scrollButtonVariants = {
  hidden: { opacity: 0, scale: 0.8, y: 20 },
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

export const MessageList = forwardRef<HTMLDivElement, MessageListProps>(({
  messages,
  isLoading = false,
  branding,
  onScrollToBottom,
  className
}, ref) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [showScrollButton, setShowScrollButton] = React.useState(false);
  const [isUserScrolling, setIsUserScrolling] = React.useState(false);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (!isUserScrolling && containerRef.current) {
      const element = containerRef.current;
      element.scrollTop = element.scrollHeight;
    }
  }, [messages, isUserScrolling]);

  // Handle scroll events
  const handleScroll = (event: React.UIEvent<HTMLDivElement>) => {
    const element = event.currentTarget;
    const isAtBottom = element.scrollHeight - element.scrollTop - element.clientHeight < 50;
    
    setShowScrollButton(!isAtBottom && messages.length > 0);
    
    // Detect if user is manually scrolling
    if (!isAtBottom) {
      setIsUserScrolling(true);
      // Clear user scrolling flag after 3 seconds of no new messages
      const timer = setTimeout(() => setIsUserScrolling(false), 3000);
      return () => clearTimeout(timer);
    } else {
      setIsUserScrolling(false);
    }
  };

  // Scroll to bottom handler
  const scrollToBottom = () => {
    if (containerRef.current) {
      containerRef.current.scrollTo({
        top: containerRef.current.scrollHeight,
        behavior: 'smooth'
      });
    }
    setIsUserScrolling(false);
    onScrollToBottom?.();
  };

  // Combine refs
  const combinedRef = (node: HTMLDivElement) => {
    containerRef.current = node;
    if (typeof ref === 'function') {
      ref(node);
    } else if (ref) {
      ref.current = node;
    }
  };

  // Show welcome message if no messages
  if (messages.length === 0 && !isLoading) {
    return (
      <Container ref={combinedRef} className={className}>
        <WelcomeMessage
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <WelcomeTitle>
            Welcome to {branding.companyName || 'Customer Support'}!
          </WelcomeTitle>
          <WelcomeText>
            {branding.welcomeMessage || 'How can we help you today?'}
          </WelcomeText>
        </WelcomeMessage>
      </Container>
    );
  }

  return (
    <Container 
      ref={combinedRef} 
      onScroll={handleScroll}
      className={className}
    >
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {messages.map((message, index) => {
          const previousMessage = index > 0 ? messages[index - 1] : undefined;
          const showSeparator = shouldShowTimestampSeparator(message, previousMessage);
          
          return (
            <React.Fragment key={message.id}>
              {/* Timestamp separator */}
              {showSeparator && (
                <TimestampSeparator>
                  <TimestampText>
                    {formatTimestamp(message.timestamp)}
                  </TimestampText>
                </TimestampSeparator>
              )}
              
              {/* Message */}
              <MessageContainer
                role={message.role}
                variants={messageVariants}
                layout
              >
                {/* Avatar */}
                <Avatar 
                  role={message.role}
                  avatarUrl={message.role === 'assistant' ? branding.avatarUrl : undefined}
                >
                  {!branding.avatarUrl && message.role === 'assistant' && getAvatarContent(message.role, branding)}
                  {message.role === 'user' && <UserIcon size={16} />}
                </Avatar>
                
                {/* Message bubble */}
                <MessagesGroup role={message.role}>
                  <MessageBubble
                    message={message}
                    branding={branding}
                    isLatest={index === messages.length - 1}
                  />
                </MessagesGroup>
              </MessageContainer>
            </React.Fragment>
          );
        })}
      </motion.div>
      
      {/* Loading indicator */}
      {isLoading && (
        <MessageContainer role="assistant">
          <Avatar role="assistant" avatarUrl={branding.avatarUrl}>
            {!branding.avatarUrl && getAvatarContent('assistant', branding)}
          </Avatar>
          <MessagesGroup role="assistant">
            <LoadingSpinner size="small" />
          </MessagesGroup>
        </MessageContainer>
      )}
      
      {/* Scroll to bottom button */}
      <AnimatePresence>
        {showScrollButton && (
          <ScrollToBottomButton
            onClick={scrollToBottom}
            variants={scrollButtonVariants}
            initial="hidden"
            animate="visible"
            exit="hidden"
            aria-label="Scroll to bottom"
          >
            <ArrowDownIcon />
            New messages
          </ScrollToBottomButton>
        )}
      </AnimatePresence>
    </Container>
  );
});

MessageList.displayName = 'MessageList';

export default MessageList;