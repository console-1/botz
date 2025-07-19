/**
 * Chat Header Component
 * 
 * The header section of the chat window with branding and controls
 */

import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { BrandingConfig } from '../types/chat';

// Styled components
const HeaderContainer = styled.div`
  background: ${props => props.theme.primaryColor};
  color: white;
  padding: 16px 20px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  min-height: 70px;
  position: relative;
  overflow: hidden;
`;

const BrandingSection = styled.div`
  display: flex;
  align-items: center;
  flex: 1;
  min-width: 0; /* Allows text truncation */
`;

const Avatar = styled.div<{ avatarUrl?: string }>`
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background: ${props => props.avatarUrl 
    ? `url(${props.avatarUrl}) center/cover` 
    : 'rgba(255, 255, 255, 0.2)'};
  display: flex;
  align-items: center;
  justify-content: center;
  margin-right: 12px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  flex-shrink: 0;
`;

const DefaultAvatarIcon: React.FC<{ size?: number }> = ({ size = 20 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 2C13.1 2 14 2.9 14 4C14 5.1 13.1 6 12 6C10.9 6 10 5.1 10 4C10 2.9 10.9 2 12 2ZM21 9V7L15 1H5C3.89 1 3 1.89 3 3V21C3 22.11 3.89 23 5 23H19C20.11 23 21 22.11 21 21V9M12 13C14.67 13 20 14.33 20 17V20H4V17C4 14.33 9.33 13 12 13Z"/>
  </svg>
);

const BrandingInfo = styled.div`
  flex: 1;
  min-width: 0;
`;

const CompanyName = styled.h3`
  margin: 0;
  font-size: 16px;
  font-weight: 600;
  color: white;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  line-height: 1.2;
`;

const StatusText = styled.p<{ isConnected: boolean }>`
  margin: 2px 0 0 0;
  font-size: 12px;
  opacity: 0.9;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  line-height: 1.2;
  display: flex;
  align-items: center;
`;

const StatusIndicator = styled.div<{ isConnected: boolean }>`
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: ${props => props.isConnected ? '#2ecc71' : '#e74c3c'};
  margin-right: 6px;
  animation: ${props => props.isConnected ? 'none' : 'pulse 2s infinite'};
  
  @keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
  }
`;

const ControlsSection = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
`;

const ControlButton = styled(motion.button)`
  background: rgba(255, 255, 255, 0.2);
  border: none;
  border-radius: 50%;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  cursor: pointer;
  transition: background-color 0.2s ease;
  
  &:hover {
    background: rgba(255, 255, 255, 0.3);
  }
  
  &:active {
    background: rgba(255, 255, 255, 0.4);
  }
  
  &:focus {
    outline: 2px solid rgba(255, 255, 255, 0.5);
    outline-offset: 1px;
  }
`;

const Logo = styled.img`
  height: 24px;
  width: auto;
  max-width: 120px;
  object-fit: contain;
`;

const PoweredBy = styled.div`
  position: absolute;
  bottom: 4px;
  right: 20px;
  font-size: 10px;
  opacity: 0.7;
  pointer-events: none;
`;

// Icons
const MinimizeIcon: React.FC<{ size?: number }> = ({ size = 16 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
    <path d="M6 19h12v2H6v-2z"/>
  </svg>
);

const CloseIcon: React.FC<{ size?: number }> = ({ size = 16 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
    <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12 19 6.41z"/>
  </svg>
);

const MoreIcon: React.FC<{ size?: number }> = ({ size = 16 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 8c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm0 2c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm0 6c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z"/>
  </svg>
);

// Component props
interface ChatHeaderProps {
  branding: BrandingConfig;
  isConnected: boolean;
  onClose: () => void;
  onMinimize?: () => void;
  onMenuClick?: () => void;
  showControls?: boolean;
}

// Animation variants
const buttonVariants = {
  hover: { scale: 1.1 },
  tap: { scale: 0.9 }
};

export const ChatHeader: React.FC<ChatHeaderProps> = ({
  branding,
  isConnected,
  onClose,
  onMinimize,
  onMenuClick,
  showControls = true
}) => {
  const getStatusText = () => {
    if (!isConnected) {
      return 'Connecting...';
    }
    
    return branding.botName 
      ? `Chat with ${branding.botName}` 
      : 'Online now';
  };

  const handleKeyDown = (event: React.KeyboardEvent, action: () => void) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      action();
    }
  };

  return (
    <HeaderContainer>
      <BrandingSection>
        {/* Avatar/Logo */}
        <Avatar avatarUrl={branding.avatarUrl}>
          {!branding.avatarUrl && <DefaultAvatarIcon />}
        </Avatar>
        
        {/* Company/Bot Info */}
        <BrandingInfo>
          {branding.logoUrl ? (
            <Logo 
              src={branding.logoUrl} 
              alt={branding.companyName || 'Company Logo'}
              onError={(e) => {
                // Hide logo if it fails to load
                e.currentTarget.style.display = 'none';
              }}
            />
          ) : (
            <CompanyName>
              {branding.companyName || 'Customer Support'}
            </CompanyName>
          )}
          
          <StatusText isConnected={isConnected}>
            <StatusIndicator isConnected={isConnected} />
            {getStatusText()}
          </StatusText>
        </BrandingInfo>
      </BrandingSection>
      
      {/* Control Buttons */}
      {showControls && (
        <ControlsSection>
          {onMenuClick && (
            <ControlButton
              onClick={onMenuClick}
              onKeyDown={(e) => handleKeyDown(e, onMenuClick)}
              variants={buttonVariants}
              whileHover="hover"
              whileTap="tap"
              aria-label="More options"
              title="More options"
            >
              <MoreIcon />
            </ControlButton>
          )}
          
          {onMinimize && (
            <ControlButton
              onClick={onMinimize}
              onKeyDown={(e) => handleKeyDown(e, onMinimize)}
              variants={buttonVariants}
              whileHover="hover"
              whileTap="tap"
              aria-label="Minimize chat"
              title="Minimize"
            >
              <MinimizeIcon />
            </ControlButton>
          )}
          
          <ControlButton
            onClick={onClose}
            onKeyDown={(e) => handleKeyDown(e, onClose)}
            variants={buttonVariants}
            whileHover="hover"
            whileTap="tap"
            aria-label="Close chat"
            title="Close"
          >
            <CloseIcon />
          </ControlButton>
        </ControlsSection>
      )}
      
      {/* Powered By */}
      {branding.poweredByText && (
        <PoweredBy>
          {branding.poweredByText}
        </PoweredBy>
      )}
    </HeaderContainer>
  );
};

export default ChatHeader;