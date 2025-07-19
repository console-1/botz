/**
 * Loading Spinner Component
 * 
 * Reusable loading indicator for various states
 */

import React from 'react';
import styled, { keyframes } from 'styled-components';

// Animations
const spin = keyframes`
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
`;

const pulse = keyframes`
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
`;

const bounce = keyframes`
  0%, 80%, 100% { transform: scaleY(0.6); }
  40% { transform: scaleY(1); }
`;

// Styled components
const Container = styled.div<{ size: 'small' | 'medium' | 'large' }>`
  display: flex;
  align-items: center;
  justify-content: center;
  padding: ${props => {
    switch (props.size) {
      case 'small': return '8px';
      case 'medium': return '16px';
      case 'large': return '32px';
      default: return '16px';
    }
  }};
`;

const Spinner = styled.div<{ size: 'small' | 'medium' | 'large'; color?: string }>`
  width: ${props => {
    switch (props.size) {
      case 'small': return '16px';
      case 'medium': return '24px';
      case 'large': return '40px';
      default: return '24px';
    }
  }};
  height: ${props => {
    switch (props.size) {
      case 'small': return '16px';
      case 'medium': return '24px';
      case 'large': return '40px';
      default: return '24px';
    }
  }};
  border: 2px solid transparent;
  border-top: 2px solid ${props => props.color || props.theme.primaryColor || '#007bff'};
  border-radius: 50%;
  animation: ${spin} 1s linear infinite;
`;

const DotsContainer = styled.div`
  display: flex;
  gap: 4px;
  align-items: center;
`;

const Dot = styled.div<{ size: 'small' | 'medium' | 'large'; color?: string }>`
  width: ${props => {
    switch (props.size) {
      case 'small': return '4px';
      case 'medium': return '6px';
      case 'large': return '8px';
      default: return '6px';
    }
  }};
  height: ${props => {
    switch (props.size) {
      case 'small': return '4px';
      case 'medium': return '6px';
      case 'large': return '8px';
      default: return '6px';
    }
  }};
  background: ${props => props.color || props.theme.primaryColor || '#007bff'};
  border-radius: 50%;
  animation: ${pulse} 1.4s ease-in-out infinite;
  
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

const BarsContainer = styled.div`
  display: flex;
  gap: 2px;
  align-items: center;
`;

const Bar = styled.div<{ size: 'small' | 'medium' | 'large'; color?: string }>`
  width: ${props => {
    switch (props.size) {
      case 'small': return '2px';
      case 'medium': return '3px';
      case 'large': return '4px';
      default: return '3px';
    }
  }};
  height: ${props => {
    switch (props.size) {
      case 'small': return '12px';
      case 'medium': return '18px';
      case 'large': return '24px';
      default: return '18px';
    }
  }};
  background: ${props => props.color || props.theme.primaryColor || '#007bff'};
  border-radius: 1px;
  animation: ${bounce} 1.4s ease-in-out infinite;
  
  &:nth-child(1) {
    animation-delay: 0s;
  }
  
  &:nth-child(2) {
    animation-delay: 0.1s;
  }
  
  &:nth-child(3) {
    animation-delay: 0.2s;
  }
  
  &:nth-child(4) {
    animation-delay: 0.3s;
  }
`;

const LoadingText = styled.div<{ size: 'small' | 'medium' | 'large' }>`
  margin-left: 8px;
  color: ${props => props.theme.textColor || '#666'};
  font-size: ${props => {
    switch (props.size) {
      case 'small': return '12px';
      case 'medium': return '14px';
      case 'large': return '16px';
      default: return '14px';
    }
  }};
`;

// Component props
interface LoadingSpinnerProps {
  size?: 'small' | 'medium' | 'large';
  type?: 'spinner' | 'dots' | 'bars';
  color?: string;
  text?: string;
  className?: string;
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  size = 'medium',
  type = 'spinner',
  color,
  text,
  className
}) => {
  const renderLoader = () => {
    switch (type) {
      case 'dots':
        return (
          <DotsContainer>
            <Dot size={size} color={color} />
            <Dot size={size} color={color} />
            <Dot size={size} color={color} />
          </DotsContainer>
        );
      
      case 'bars':
        return (
          <BarsContainer>
            <Bar size={size} color={color} />
            <Bar size={size} color={color} />
            <Bar size={size} color={color} />
            <Bar size={size} color={color} />
          </BarsContainer>
        );
      
      case 'spinner':
      default:
        return <Spinner size={size} color={color} />;
    }
  };

  return (
    <Container size={size} className={className}>
      {renderLoader()}
      {text && <LoadingText size={size}>{text}</LoadingText>}
    </Container>
  );
};

export default LoadingSpinner;