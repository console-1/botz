/**
 * Chat Button Component
 * 
 * The floating action button that opens/closes the chat widget
 */

import React from 'react';
import styled, { keyframes } from 'styled-components';
import { motion } from 'framer-motion';
import { BrandingConfig, WidgetTheme } from '../types/chat';

// Animations
const pulse = keyframes`
  0% { transform: scale(1); }
  50% { transform: scale(1.05); }
  100% { transform: scale(1); }
`;

const bounce = keyframes`
  0%, 20%, 53%, 80%, 100% { transform: translate3d(0,0,0); }
  40%, 43% { transform: translate3d(0,-8px,0); }
  70% { transform: translate3d(0,-4px,0); }
  90% { transform: translate3d(0,-2px,0); }
`;

// Styled components
const ButtonContainer = styled(motion.div)`
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
`;

const Button = styled(motion.button)<{ theme: WidgetTheme; isOpen: boolean }>`
  width: 60px;
  height: 60px;
  border-radius: 50%;
  border: none;
  background: ${props => props.isOpen ? props.theme.secondaryColor : props.theme.primaryColor};
  color: white;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
  
  &:hover {
    transform: scale(1.05);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
  }
  
  &:active {
    transform: scale(0.95);
  }
  
  &:focus {
    outline: 2px solid ${props => props.theme.primaryColor};
    outline-offset: 2px;
  }
`;

const IconContainer = styled.div<{ isOpen: boolean }>`
  display: flex;
  align-items: center;
  justify-content: center;
  transform: ${props => props.isOpen ? 'rotate(45deg)' : 'rotate(0deg)'};
  transition: transform 0.3s ease;
`;

const UnreadBadge = styled(motion.div)<{ theme: WidgetTheme }>`
  position: absolute;
  top: -5px;
  right: -5px;
  background: #e74c3c;
  color: white;
  border-radius: 12px;
  min-width: 20px;
  height: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: 600;
  border: 2px solid white;
  animation: ${pulse} 2s infinite;
`;

const PulseRing = styled.div<{ theme: WidgetTheme }>`
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 80px;
  height: 80px;
  border: 2px solid ${props => props.theme.primaryColor};
  border-radius: 50%;
  opacity: 0.6;
  animation: ${pulse} 2s infinite;
`;

const TooltipContainer = styled(motion.div)<{ side: 'left' | 'right' }>`
  position: absolute;
  top: 50%;
  ${props => props.side === 'right' ? 'right: 75px' : 'left: 75px'};
  transform: translateY(-50%);
  background: rgba(0, 0, 0, 0.8);
  color: white;
  padding: 8px 12px;
  border-radius: 6px;
  font-size: 14px;
  white-space: nowrap;
  pointer-events: none;
  z-index: 1000;
  
  &::after {
    content: '';
    position: absolute;
    top: 50%;
    ${props => props.side === 'right' ? 'left: 100%' : 'right: 100%'};
    transform: translateY(-50%);
    width: 0;
    height: 0;
    border-style: solid;
    border-width: 5px 0 5px 8px;
    border-color: ${props => props.side === 'right' 
      ? 'transparent transparent transparent rgba(0, 0, 0, 0.8)'
      : 'transparent rgba(0, 0, 0, 0.8) transparent transparent'};
  }
`;

const BounceNotification = styled.div`
  animation: ${bounce} 1s ease-in-out;
`;

// SVG Icons
const ChatIcon: React.FC<{ size?: number }> = ({ size = 24 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
    <path d="M20 2H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h4l4 4 4-4h4c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm-2 12H6v-2h12v2zm0-3H6V9h12v2zm0-3H6V6h12v2z"/>
  </svg>
);

const CloseIcon: React.FC<{ size?: number }> = ({ size = 24 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
    <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12 19 6.41z"/>
  </svg>
);

// Component props
interface ChatButtonProps {
  isOpen: boolean;
  hasUnreadMessages: boolean;
  unreadCount: number;
  onClick: () => void;
  theme: WidgetTheme;
  branding: BrandingConfig;
  showTooltip?: boolean;
  disabled?: boolean;
}

// Animation variants
const buttonVariants = {
  hidden: { scale: 0, opacity: 0 },
  visible: { 
    scale: 1, 
    opacity: 1,
    transition: {
      type: 'spring',
      stiffness: 260,
      damping: 20
    }
  },
  tap: { scale: 0.9 }
};

const badgeVariants = {
  hidden: { scale: 0, opacity: 0 },
  visible: { 
    scale: 1, 
    opacity: 1,
    transition: {
      type: 'spring',
      stiffness: 400,
      damping: 25
    }
  }
};

const tooltipVariants = {
  hidden: { opacity: 0, scale: 0.8, x: 10 },
  visible: { 
    opacity: 1, 
    scale: 1, 
    x: 0,
    transition: {
      duration: 0.2
    }
  }
};

export const ChatButton: React.FC<ChatButtonProps> = ({
  isOpen,
  hasUnreadMessages,
  unreadCount,
  onClick,
  theme,
  branding,
  showTooltip = false,
  disabled = false
}) => {
  const [isHovered, setIsHovered] = React.useState(false);
  const [showBounce, setShowBounce] = React.useState(false);

  // Trigger bounce animation when new unread messages arrive
  React.useEffect(() => {
    if (hasUnreadMessages && !isOpen) {
      setShowBounce(true);
      const timer = setTimeout(() => setShowBounce(false), 1000);
      return () => clearTimeout(timer);
    }
  }, [hasUnreadMessages, isOpen, unreadCount]);

  const handleClick = () => {
    if (!disabled) {
      onClick();
    }
  };

  const shouldShowTooltip = showTooltip && isHovered && !isOpen;
  const shouldShowPulse = hasUnreadMessages && !isOpen;

  return (
    <ButtonContainer
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Pulse ring for notifications */}
      {shouldShowPulse && <PulseRing theme={theme} />}
      
      {/* Main button */}
      <Button
        theme={theme}
        isOpen={isOpen}
        onClick={handleClick}
        disabled={disabled}
        variants={buttonVariants}
        initial="hidden"
        animate="visible"
        whileTap="tap"
        className={showBounce ? 'bounce' : ''}
        aria-label={isOpen ? 'Close chat' : 'Open chat'}
        role="button"
        tabIndex={0}
      >
        {showBounce ? (
          <BounceNotification>
            <IconContainer isOpen={isOpen}>
              {isOpen ? <CloseIcon /> : <ChatIcon />}
            </IconContainer>
          </BounceNotification>
        ) : (
          <IconContainer isOpen={isOpen}>
            {isOpen ? <CloseIcon /> : <ChatIcon />}
          </IconContainer>
        )}
      </Button>
      
      {/* Unread message badge */}
      {hasUnreadMessages && !isOpen && (
        <UnreadBadge
          theme={theme}
          variants={badgeVariants}
          initial="hidden"
          animate="visible"
          exit="hidden"
        >
          {unreadCount > 99 ? '99+' : unreadCount}
        </UnreadBadge>
      )}
      
      {/* Tooltip */}
      {shouldShowTooltip && (
        <TooltipContainer
          side={theme.position?.side || 'right'}
          variants={tooltipVariants}
          initial="hidden"
          animate="visible"
          exit="hidden"
        >
          {isOpen ? 'Close chat' : branding.companyName ? `Chat with ${branding.companyName}` : 'Chat with us'}
        </TooltipContainer>
      )}
    </ButtonContainer>
  );
};

export default ChatButton;