/**
 * Error Boundary Component
 * 
 * Catches and handles React errors gracefully
 */

import React, { Component, ReactNode } from 'react';
import styled from 'styled-components';

// Styled components
const ErrorContainer = styled.div`
  padding: 20px;
  text-align: center;
  background: #fff5f5;
  border: 1px solid #fed7d7;
  border-radius: 8px;
  margin: 16px;
`;

const ErrorTitle = styled.h3`
  color: #e53e3e;
  margin: 0 0 8px 0;
  font-size: 16px;
`;

const ErrorMessage = styled.p`
  color: #744210;
  margin: 0 0 16px 0;
  font-size: 14px;
`;

const ErrorButton = styled.button`
  background: #e53e3e;
  color: white;
  border: none;
  border-radius: 6px;
  padding: 8px 16px;
  font-size: 14px;
  cursor: pointer;
  transition: background-color 0.2s;
  
  &:hover {
    background: #c53030;
  }
`;

const ErrorDetails = styled.details`
  margin-top: 16px;
  text-align: left;
  
  summary {
    cursor: pointer;
    font-weight: 600;
    color: #744210;
    margin-bottom: 8px;
  }
  
  pre {
    background: #f7fafc;
    padding: 12px;
    border-radius: 4px;
    font-size: 12px;
    overflow-x: auto;
    white-space: pre-wrap;
    color: #2d3748;
  }
`;

// Component props
interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: React.ErrorInfo | null;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null
    };
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    // Update state so the next render will show the fallback UI
    return {
      hasError: true,
      error
    };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    // Log error details
    console.error('Error Boundary caught an error:', error, errorInfo);
    
    this.setState({
      error,
      errorInfo
    });

    // Call optional error handler
    this.props.onError?.(error, errorInfo);
  }

  handleRetry = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null
    });
  };

  render() {
    if (this.state.hasError) {
      // Custom fallback UI
      if (this.props.fallback) {
        return this.props.fallback;
      }

      // Default error UI
      return (
        <ErrorContainer>
          <ErrorTitle>Something went wrong</ErrorTitle>
          <ErrorMessage>
            The chat widget encountered an unexpected error. Please try refreshing the page.
          </ErrorMessage>
          <ErrorButton onClick={this.handleRetry}>
            Try Again
          </ErrorButton>
          
          {process.env.NODE_ENV === 'development' && this.state.error && (
            <ErrorDetails>
              <summary>Error Details (Development Only)</summary>
              <pre>
                {this.state.error.toString()}
                {this.state.errorInfo?.componentStack}
              </pre>
            </ErrorDetails>
          )}
        </ErrorContainer>
      );
    }

    // No error, render children normally
    return this.props.children;
  }
}

export default ErrorBoundary;