/**
 * Escalation Banner Component
 * 
 * Shows when a conversation is escalated to human agent
 */

import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { EscalationStatus } from '../types/chat';

// Styled components
const BannerContainer = styled(motion.div)<{ priority: string }>`
  background: ${props => {
    switch (props.priority) {
      case 'critical': return '#e74c3c';
      case 'urgent': return '#e67e22';
      case 'high': return '#f39c12';
      case 'medium': return '#3498db';
      default: return '#95a5a6';
    }
  }};
  color: white;
  padding: 12px 20px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 13px;
  position: relative;
  overflow: hidden;
`;

const BannerContent = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  flex: 1;
`;

const Icon = styled.div`
  width: 20px;
  height: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
`;

const TextContent = styled.div`
  display: flex;
  flex-direction: column;
  gap: 2px;
`;

const MainText = styled.div`
  font-weight: 600;
  line-height: 1.2;
`;

const SubText = styled.div`
  opacity: 0.9;
  font-size: 11px;
  line-height: 1.2;
`;

const CloseButton = styled.button`
  background: rgba(255, 255, 255, 0.2);
  border: none;
  border-radius: 50%;
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  cursor: pointer;
  transition: background-color 0.2s ease;
  flex-shrink: 0;
  
  &:hover {
    background: rgba(255, 255, 255, 0.3);
  }
  
  &:focus {
    outline: 2px solid rgba(255, 255, 255, 0.5);
    outline-offset: 1px;
  }
`;

const PulseAnimation = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
  animation: pulse 2s infinite;
  
  @keyframes pulse {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
  }
`;

// Icons
const EscalationIcon: React.FC<{ size?: number }> = ({ size = 16 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
  </svg>
);

const AgentIcon: React.FC<{ size?: number }> = ({ size = 16 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
    <path d="M16 4c0-1.11.89-2 2-2s2 .89 2 2-.89 2-2 2-2-.89-2-2zm4 18v-6h2.5l-2.54-7.63A2.999 2.999 0 0 0 17.18 7H16c-.8 0-1.54.37-2.01.99L12 10l-1.99-2.01A2.99 2.99 0 0 0 8 7H6.82c-1.36 0-2.54.93-2.88 2.37L1.5 16H4v6h2v-6h2l2-2 2 2h2v6h4z"/>
  </svg>
);

const CloseIcon: React.FC<{ size?: number }> = ({ size = 14 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
    <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12 19 6.41z"/>
  </svg>
);

// Component props
interface EscalationBannerProps {
  escalationInfo: EscalationStatus;
  onResolve: () => void;
  onClose?: () => void;
  showAnimation?: boolean;
  className?: string;
}

// Helper functions
const getPriorityText = (priority?: string): string => {
  switch (priority) {
    case 'critical': return 'Critical Priority';
    case 'urgent': return 'Urgent';
    case 'high': return 'High Priority';
    case 'medium': return 'Medium Priority';
    case 'low': return 'Low Priority';
    default: return 'Standard';
  }
};

const getMainMessage = (escalationInfo: EscalationStatus): string => {
  if (escalationInfo.agentInfo) {
    return `Connected to ${escalationInfo.agentInfo.name}`;
  }
  
  return 'Escalated to Human Agent';
};

const getSubMessage = (escalationInfo: EscalationStatus): string => {
  if (escalationInfo.agentInfo?.status === 'online') {
    return 'Agent is online and will respond soon';
  }
  
  if (escalationInfo.reason) {
    return `Reason: ${escalationInfo.reason}`;
  }
  
  return 'You will be connected to a human agent shortly';
};

// Animation variants
const bannerVariants = {
  hidden: { 
    opacity: 0, 
    height: 0,
    y: -10 
  },
  visible: { 
    opacity: 1, 
    height: 'auto',
    y: 0,
    transition: {
      duration: 0.3,
      ease: 'easeOut'
    }
  },
  exit: {
    opacity: 0,
    height: 0,
    y: -10,
    transition: {
      duration: 0.2
    }
  }
};

export const EscalationBanner: React.FC<EscalationBannerProps> = ({
  escalationInfo,
  onResolve,
  onClose,
  showAnimation = true,
  className
}) => {
  if (!escalationInfo.isEscalated) {
    return null;
  }

  const priority = escalationInfo.priority || 'medium';
  const mainText = getMainMessage(escalationInfo);
  const subText = getSubMessage(escalationInfo);

  const handleClose = () => {
    onClose?.();
  };

  const handleKeyDown = (event: React.KeyboardEvent, action: () => void) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      action();
    }
  };

  return (
    <BannerContainer
      priority={priority}
      className={className}
      variants={bannerVariants}
      initial="hidden"
      animate="visible"
      exit="exit"
      role="alert"
      aria-live="polite"
    >
      {showAnimation && <PulseAnimation />}
      
      <BannerContent>
        <Icon>
          {escalationInfo.agentInfo ? (
            <AgentIcon />
          ) : (
            <EscalationIcon />
          )}
        </Icon>
        
        <TextContent>
          <MainText>{mainText}</MainText>
          <SubText>{subText}</SubText>
        </TextContent>
      </BannerContent>
      
      {onClose && (
        <CloseButton
          onClick={handleClose}
          onKeyDown={(e) => handleKeyDown(e, handleClose)}
          aria-label="Close escalation banner"
          title="Close"
        >
          <CloseIcon />
        </CloseButton>
      )}
    </BannerContainer>
  );
};

export default EscalationBanner;