/**
 * Typing Indicator Component
 * 
 * Shows when the bot is typing a response
 */

import React from 'react';
import styled, { keyframes } from 'styled-components';
import { motion } from 'framer-motion';

// Animations
const bounce = keyframes`
  0%, 60%, 100% {
    animation-timing-function: cubic-bezier(0.215, 0.61, 0.355, 1);
    transform: translate3d(0, 0, 0);
  }
  40% {
    animation-timing-function: cubic-bezier(0.755, 0.05, 0.855, 0.06);
    transform: translate3d(0, -4px, 0);
  }
  80% {
    animation-timing-function: cubic-bezier(0.215, 0.61, 0.355, 1);
    transform: translate3d(0, -2px, 0);
  }
`;

// Styled components
const Container = styled(motion.div)`
  padding: 12px 20px;
  display: flex;
  align-items: center;
  gap: 8px;
  color: ${props => props.theme.textColor};
  opacity: 0.7;
`;

const Avatar = styled.div<{ avatarUrl?: string }>`
  width: 24px;
  height: 24px;
  border-radius: 50%;
  background: ${props => props.avatarUrl 
    ? `url(${props.avatarUrl}) center/cover` 
    : props.theme.secondaryColor};
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  color: white;
  font-size: 10px;
  font-weight: 600;
`;

const TypingBubble = styled.div`
  background: #f8f9fa;
  border-radius: 18px;
  padding: 8px 12px;
  display: flex;
  align-items: center;
  gap: 3px;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
`;

const Dot = styled.div`
  width: 6px;
  height: 6px;
  background: #999;
  border-radius: 50%;
  animation: ${bounce} 1.4s infinite;
  
  &:nth-child(1) {
    animation-delay: 0s;
  }
  
  &:nth-child(2) {
    animation-delay: 0.2s;
  }
  
  &:nth-child(3) {
    animation-delay: 0.4s;
  }
`;

const TypingText = styled.span`
  font-size: 12px;
  font-style: italic;
  margin-left: 4px;
`;

// Component props
interface TypingIndicatorProps {
  botName?: string;
  avatarUrl?: string;
  showText?: boolean;
  className?: string;
}

// Animation variants
const containerVariants = {
  hidden: { opacity: 0, y: 10 },
  visible: { 
    opacity: 1, 
    y: 0,
    transition: {
      duration: 0.3,
      ease: 'easeOut'
    }
  },
  exit: {
    opacity: 0,
    y: -10,
    transition: {
      duration: 0.2
    }
  }
};

export const TypingIndicator: React.FC<TypingIndicatorProps> = ({
  botName = 'Assistant',
  avatarUrl,
  showText = true,
  className
}) => {
  const getAvatarContent = () => {
    if (avatarUrl) return null;
    return botName.charAt(0).toUpperCase();
  };

  return (
    <Container
      className={className}
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      exit="exit"
    >
      <Avatar avatarUrl={avatarUrl}>
        {getAvatarContent()}
      </Avatar>
      
      <TypingBubble>
        <Dot />
        <Dot />
        <Dot />
      </TypingBubble>
      
      {showText && (
        <TypingText>
          {botName} is typing...
        </TypingText>
      )}
    </Container>
  );
};

export default TypingIndicator;