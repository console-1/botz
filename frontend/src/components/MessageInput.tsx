/**
 * Message Input Component
 * 
 * Text input with send button and typing indicators
 */

import React, { useState, useRef, forwardRef, useImperativeHandle } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';

// Styled components
const InputContainer = styled.div`
  padding: 16px 20px;
  background: ${props => props.theme.backgroundColor};
  border-top: 1px solid rgba(0, 0, 0, 0.1);
  display: flex;
  align-items: flex-end;
  gap: 12px;
`;

const InputWrapper = styled.div`
  flex: 1;
  position: relative;
  display: flex;
  flex-direction: column;
`;

const TextArea = styled.textarea<{ hasContent: boolean }>`
  border: 2px solid ${props => props.hasContent ? props.theme.primaryColor : 'rgba(0, 0, 0, 0.1)'};
  border-radius: ${props => props.theme.borderRadius || 20}px;
  padding: 12px 16px;
  font-family: inherit;
  font-size: 14px;
  line-height: 1.4;
  resize: none;
  outline: none;
  min-height: 44px;
  max-height: 120px;
  background: white;
  color: ${props => props.theme.textColor};
  transition: all 0.2s ease;
  
  &::placeholder {
    color: rgba(0, 0, 0, 0.4);
  }
  
  &:focus {
    border-color: ${props => props.theme.primaryColor};
    box-shadow: 0 0 0 3px ${props => props.theme.primaryColor}20;
  }
  
  &:disabled {
    background: #f5f5f5;
    color: #999;
    cursor: not-allowed;
  }
  
  /* Hide scrollbar but keep functionality */
  scrollbar-width: none;
  -ms-overflow-style: none;
  &::-webkit-scrollbar {
    display: none;
  }
`;

const SendButton = styled(motion.button)<{ hasContent: boolean; disabled: boolean }>`
  width: 44px;
  height: 44px;
  border-radius: 50%;
  border: none;
  background: ${props => {
    if (props.disabled) return '#ccc';
    return props.hasContent ? props.theme.primaryColor : 'rgba(0, 0, 0, 0.1)';
  }};
  color: ${props => props.hasContent ? 'white' : 'rgba(0, 0, 0, 0.4)'};
  cursor: ${props => props.disabled ? 'not-allowed' : 'pointer'};
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s ease;
  flex-shrink: 0;
  
  &:hover:not(:disabled) {
    transform: scale(1.05);
    background: ${props => props.hasContent ? props.theme.primaryColor : 'rgba(0, 0, 0, 0.15)'};
  }
  
  &:active:not(:disabled) {
    transform: scale(0.95);
  }
  
  &:focus {
    outline: 2px solid ${props => props.theme.primaryColor};
    outline-offset: 2px;
  }
  
  svg {
    transition: transform 0.2s ease;
  }
`;

const CharacterCount = styled.div<{ isNearLimit: boolean; isOverLimit: boolean }>`
  position: absolute;
  bottom: -20px;
  right: 0;
  font-size: 11px;
  color: ${props => {
    if (props.isOverLimit) return '#e74c3c';
    if (props.isNearLimit) return '#f39c12';
    return 'rgba(0, 0, 0, 0.4)';
  }};
  pointer-events: none;
`;

const QuickActions = styled.div`
  display: flex;
  gap: 8px;
  margin-bottom: 8px;
  overflow-x: auto;
  padding: 0 2px;
  
  /* Hide scrollbar */
  scrollbar-width: none;
  -ms-overflow-style: none;
  &::-webkit-scrollbar {
    display: none;
  }
`;

const QuickActionButton = styled.button`
  background: rgba(0, 0, 0, 0.05);
  border: 1px solid rgba(0, 0, 0, 0.1);
  border-radius: 16px;
  padding: 6px 12px;
  font-size: 12px;
  color: ${props => props.theme.textColor};
  cursor: pointer;
  white-space: nowrap;
  transition: all 0.2s ease;
  
  &:hover {
    background: rgba(0, 0, 0, 0.1);
    transform: translateY(-1px);
  }
  
  &:active {
    transform: translateY(0);
  }
`;

const TypingIndicator = styled.div`
  font-size: 11px;
  color: rgba(0, 0, 0, 0.6);
  padding: 4px 0;
  height: 20px;
  display: flex;
  align-items: center;
`;

const AttachButton = styled.button`
  width: 32px;
  height: 32px;
  border: none;
  background: transparent;
  color: rgba(0, 0, 0, 0.4);
  cursor: pointer;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s ease;
  
  &:hover {
    background: rgba(0, 0, 0, 0.05);
    color: rgba(0, 0, 0, 0.6);
  }
  
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

// Icons
const SendIcon: React.FC<{ size?: number }> = ({ size = 20 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
    <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
  </svg>
);

const AttachIcon: React.FC<{ size?: number }> = ({ size = 16 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
    <path d="M16.5 6v11.5c0 2.21-1.79 4-4 4s-4-1.79-4-4V5c0-1.38 1.12-2.5 2.5-2.5s2.5 1.12 2.5 2.5v10.5c0 .55-.45 1-1 1s-1-.45-1-1V6H10v9.5c0 1.38 1.12 2.5 2.5 2.5s2.5-1.12 2.5-2.5V5c0-2.21-1.79-4-4-4S7 2.79 7 5v12.5c0 3.04 2.46 5.5 5.5 5.5s5.5-2.46 5.5-5.5V6h-1.5z"/>
  </svg>
);

const EmojiIcon: React.FC<{ size?: number }> = ({ size = 16 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
    <path d="M11.99 2C6.47 2 2 6.48 2 12s4.47 10 9.99 10C17.52 22 22 17.52 22 12S17.52 2 11.99 2zM12 20c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8zm3.5-9c.83 0 1.5-.67 1.5-1.5S16.33 8 15.5 8 14 8.67 14 9.5s.67 1.5 1.5 1.5zm-7 0c.83 0 1.5-.67 1.5-1.5S9.33 8 8.5 8 7 8.67 7 9.5 7.67 11 8.5 11zm3.5 6.5c2.33 0 4.31-1.46 5.11-3.5H6.89c.8 2.04 2.78 3.5 5.11 3.5z"/>
  </svg>
);

// Component props
interface MessageInputProps {
  onSendMessage: (message: string) => void;
  disabled?: boolean;
  placeholder?: string;
  maxLength?: number;
  showQuickActions?: boolean;
  quickActions?: string[];
  showAttachments?: boolean;
  showEmojis?: boolean;
  onTyping?: (isTyping: boolean) => void;
  className?: string;
}

// Quick action suggestions
const defaultQuickActions = [
  "I need help with my order",
  "How can I contact support?",
  "What are your hours?",
  "I have a technical issue",
  "I need a refund"
];

// Animation variants
const buttonVariants = {
  hover: { scale: 1.05 },
  tap: { scale: 0.95 }
};

export const MessageInput = forwardRef<HTMLTextAreaElement, MessageInputProps>(({
  onSendMessage,
  disabled = false,
  placeholder = "Type your message...",
  maxLength = 1000,
  showQuickActions = true,
  quickActions = defaultQuickActions,
  showAttachments = false,
  showEmojis = false,
  onTyping,
  className
}, ref) => {
  const [message, setMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const textAreaRef = useRef<HTMLTextAreaElement>(null);
  const typingTimeoutRef = useRef<NodeJS.Timeout>();

  // Expose focus method via ref
  useImperativeHandle(ref, () => textAreaRef.current!);

  // Auto-resize textarea
  const adjustTextAreaHeight = () => {
    const textArea = textAreaRef.current;
    if (textArea) {
      textArea.style.height = 'auto';
      textArea.style.height = Math.min(textArea.scrollHeight, 120) + 'px';
    }
  };

  // Handle input change
  const handleInputChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newMessage = event.target.value;
    
    // Enforce max length
    if (maxLength && newMessage.length > maxLength) {
      return;
    }
    
    setMessage(newMessage);
    adjustTextAreaHeight();
    
    // Handle typing indicator
    if (onTyping) {
      if (!isTyping && newMessage.trim()) {
        setIsTyping(true);
        onTyping(true);
      }
      
      // Clear existing timeout
      if (typingTimeoutRef.current) {
        clearTimeout(typingTimeoutRef.current);
      }
      
      // Set new timeout to stop typing indicator
      typingTimeoutRef.current = setTimeout(() => {
        setIsTyping(false);
        onTyping(false);
      }, 1000);
    }
  };

  // Handle key down
  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  };

  // Handle send
  const handleSend = () => {
    const trimmedMessage = message.trim();
    if (trimmedMessage && !disabled) {
      onSendMessage(trimmedMessage);
      setMessage('');
      adjustTextAreaHeight();
      
      // Clear typing indicator
      if (isTyping) {
        setIsTyping(false);
        onTyping?.(false);
      }
      
      // Focus back to input
      setTimeout(() => {
        textAreaRef.current?.focus();
      }, 100);
    }
  };

  // Handle quick action click
  const handleQuickAction = (action: string) => {
    setMessage(action);
    textAreaRef.current?.focus();
    adjustTextAreaHeight();
  };

  // Character count helpers
  const characterCount = message.length;
  const isNearLimit = maxLength ? characterCount > maxLength * 0.8 : false;
  const isOverLimit = maxLength ? characterCount > maxLength : false;

  const hasContent = message.trim().length > 0;

  return (
    <InputContainer className={className}>
      <InputWrapper>
        {/* Quick actions */}
        {showQuickActions && !hasContent && quickActions.length > 0 && (
          <QuickActions>
            {quickActions.map((action, index) => (
              <QuickActionButton
                key={index}
                onClick={() => handleQuickAction(action)}
                disabled={disabled}
              >
                {action}
              </QuickActionButton>
            ))}
          </QuickActions>
        )}
        
        {/* Typing indicator */}
        {isTyping && (
          <TypingIndicator>
            You are typing...
          </TypingIndicator>
        )}
        
        {/* Main input area */}
        <div style={{ position: 'relative' }}>
          <TextArea
            ref={textAreaRef}
            value={message}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            disabled={disabled}
            hasContent={hasContent}
            rows={1}
            aria-label="Type your message"
          />
          
          {/* Character count */}
          {maxLength && characterCount > 0 && (
            <CharacterCount 
              isNearLimit={isNearLimit}
              isOverLimit={isOverLimit}
            >
              {characterCount}/{maxLength}
            </CharacterCount>
          )}
        </div>
      </InputWrapper>
      
      {/* Attachment button */}
      {showAttachments && (
        <AttachButton
          disabled={disabled}
          title="Attach file"
          aria-label="Attach file"
        >
          <AttachIcon />
        </AttachButton>
      )}
      
      {/* Emoji button */}
      {showEmojis && (
        <AttachButton
          disabled={disabled}
          title="Add emoji"
          aria-label="Add emoji"
        >
          <EmojiIcon />
        </AttachButton>
      )}
      
      {/* Send button */}
      <SendButton
        onClick={handleSend}
        disabled={disabled || !hasContent || isOverLimit}
        hasContent={hasContent}
        variants={buttonVariants}
        whileHover="hover"
        whileTap="tap"
        aria-label="Send message"
        title={hasContent ? "Send message" : "Type a message to send"}
      >
        <SendIcon />
      </SendButton>
    </InputContainer>
  );
});

MessageInput.displayName = 'MessageInput';

export default MessageInput;