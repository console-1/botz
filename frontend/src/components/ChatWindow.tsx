/**
 * Chat Window Component
 * 
 * Container component for the chat interface
 */

import React from 'react';
import styled from 'styled-components';

const WindowContainer = styled.div`
  display: flex;
  flex-direction: column;
  height: 100%;
  background: ${props => props.theme.backgroundColor};
  border-radius: ${props => props.theme.borderRadius}px;
  overflow: hidden;
`;

interface ChatWindowProps {
  children: React.ReactNode;
  className?: string;
}

export const ChatWindow: React.FC<ChatWindowProps> = ({ children, className }) => {
  return (
    <WindowContainer className={className}>
      {children}
    </WindowContainer>
  );
};

export default ChatWindow;