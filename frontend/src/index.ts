/**
 * Customer Service Bot Widget - Main Entry Point
 * 
 * This file provides the public API for initializing and managing the chat widget
 */

import React from 'react';
import { createRoot } from 'react-dom/client';
import { WidgetConfig, WidgetState, WidgetEvent, WidgetEventData } from './types/chat';
import ChatWidget from './components/ChatWidget';
import { setDefaultApiService, createChatApiService } from './services/chatApi';

// Widget instance interface
export interface ChatWidgetInstance {
  open: () => void;
  close: () => void;
  toggle: () => void;
  destroy: () => void;
  updateConfig: (config: Partial<WidgetConfig>) => void;
  getState: () => WidgetState;
  addEventListener: (event: WidgetEvent, callback: (data: WidgetEventData) => void) => void;
  removeEventListener: (event: WidgetEvent, callback: (data: WidgetEventData) => void) => void;
  sendMessage: (message: string) => void;
  setUser: (user: { id?: string; name?: string; email?: string }) => void;
}

// Default configuration
const defaultConfig: Partial<WidgetConfig> = {
  theme: {
    primaryColor: '#007bff',
    secondaryColor: '#6c757d',
    backgroundColor: '#ffffff',
    textColor: '#333333',
    borderRadius: 12,
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
    fontSize: 14
  },
  branding: {
    companyName: 'Customer Support',
    welcomeMessage: 'Hi! How can we help you today?',
    placeholderText: 'Type your message...',
    botName: 'Assistant',
    poweredByText: 'Powered by AI'
  },
  behavior: {
    autoOpen: false,
    openDelay: 0,
    showTypingIndicator: true,
    enableSound: false,
    enableEmojis: true,
    maxMessageLength: 1000,
    enableFileUpload: false,
    enableVoiceInput: false,
    persistConversation: true
  },
  position: {
    side: 'right',
    bottom: 20,
    horizontal: 20
  }
};

// Widget manager class
class ChatWidgetManager {
  private root: any = null;
  private container: HTMLElement | null = null;
  private config: WidgetConfig;
  private eventListeners: Map<WidgetEvent, Set<(data: WidgetEventData) => void>> = new Map();

  constructor(config: WidgetConfig) {
    this.config = { ...defaultConfig, ...config } as WidgetConfig;
    this.init();
  }

  private init() {
    // Create container element
    this.container = document.createElement('div');
    this.container.id = `chat-widget-${this.config.clientId}`;
    this.container.className = 'chat-widget-container';
    
    // Apply container styles
    Object.assign(this.container.style, {
      position: 'fixed',
      zIndex: '999999',
      pointerEvents: 'none',
      fontFamily: this.config.theme.fontFamily
    });

    // Append to body
    document.body.appendChild(this.container);

    // Initialize API service
    const apiService = createChatApiService({
      apiUrl: this.config.apiUrl,
      clientId: this.config.clientId
    });
    setDefaultApiService(apiService);

    // Create React root and render widget
    this.root = createRoot(this.container);
    this.renderWidget();
  }

  private renderWidget() {
    if (!this.root) return;

    this.root.render(
      React.createElement(ChatWidget, {
        config: this.config,
        onStateChange: (state: WidgetState) => {
          this.emitEvent('widget:stateChange', state);
        },
        onError: (error: Error) => {
          this.emitEvent('error:occurred', { error: error.message });
        }
      })
    );
  }

  private emitEvent(type: WidgetEvent, payload?: any) {
    const eventData: WidgetEventData = {
      type,
      payload,
      timestamp: new Date()
    };

    const listeners = this.eventListeners.get(type);
    if (listeners) {
      listeners.forEach(callback => {
        try {
          callback(eventData);
        } catch (error) {
          console.error(`[ChatWidget] Event listener error for ${type}:`, error);
        }
      });
    }

    // Log event in debug mode
    if (this.config.debug) {
      console.log(`[ChatWidget] Event: ${type}`, eventData);
    }
  }

  // Public API methods
  public open() {
    this.emitEvent('widget:open');
  }

  public close() {
    this.emitEvent('widget:close');
  }

  public toggle() {
    this.emitEvent('widget:toggle');
  }

  public destroy() {
    if (this.root) {
      this.root.unmount();
      this.root = null;
    }

    if (this.container && this.container.parentNode) {
      this.container.parentNode.removeChild(this.container);
      this.container = null;
    }

    this.eventListeners.clear();
  }

  public updateConfig(newConfig: Partial<WidgetConfig>) {
    this.config = { ...this.config, ...newConfig };
    this.renderWidget();
  }

  public addEventListener(event: WidgetEvent, callback: (data: WidgetEventData) => void) {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, new Set());
    }
    this.eventListeners.get(event)!.add(callback);
  }

  public removeEventListener(event: WidgetEvent, callback: (data: WidgetEventData) => void) {
    const listeners = this.eventListeners.get(event);
    if (listeners) {
      listeners.delete(callback);
    }
  }

  public getConfig(): WidgetConfig {
    return { ...this.config };
  }
}

// Public API function
export function createChatWidget(config: {
  clientId: string;
  apiUrl: string;
  theme?: Partial<WidgetConfig['theme']>;
  branding?: Partial<WidgetConfig['branding']>;
  behavior?: Partial<WidgetConfig['behavior']>;
  position?: Partial<WidgetConfig['position']>;
  debug?: boolean;
}): ChatWidgetInstance {
  
  // Validate required config
  if (!config.clientId) {
    throw new Error('clientId is required');
  }
  
  if (!config.apiUrl) {
    throw new Error('apiUrl is required');
  }

  // Create full config
  const fullConfig: WidgetConfig = {
    clientId: config.clientId,
    apiUrl: config.apiUrl,
    theme: { ...defaultConfig.theme, ...config.theme },
    branding: { ...defaultConfig.branding, ...config.branding },
    behavior: { ...defaultConfig.behavior, ...config.behavior },
    position: { ...defaultConfig.position, ...config.position },
    debug: config.debug || false
  } as WidgetConfig;

  // Create widget manager
  const manager = new ChatWidgetManager(fullConfig);

  // Return public API
  return {
    open: () => manager.open(),
    close: () => manager.close(),
    toggle: () => manager.toggle(),
    destroy: () => manager.destroy(),
    updateConfig: (newConfig: Partial<WidgetConfig>) => manager.updateConfig(newConfig),
    getState: () => ({ isOpen: false } as WidgetState), // This would be connected to actual state
    addEventListener: (event: WidgetEvent, callback: (data: WidgetEventData) => void) => 
      manager.addEventListener(event, callback),
    removeEventListener: (event: WidgetEvent, callback: (data: WidgetEventData) => void) => 
      manager.removeEventListener(event, callback),
    sendMessage: (message: string) => {
      // This would trigger sending a message programmatically
      console.log('Sending message:', message);
    },
    setUser: (user: { id?: string; name?: string; email?: string }) => {
      // This would set user information
      console.log('Setting user:', user);
    }
  };
}

// Global API for script tag usage
declare global {
  interface Window {
    ChatWidget: {
      create: typeof createChatWidget;
    };
  }
}

// Expose global API
if (typeof window !== 'undefined') {
  window.ChatWidget = {
    create: createChatWidget
  };
}

// Export types and functions
export * from './types/chat';
export { default as ChatWidget } from './components/ChatWidget';
export type { ChatWidgetInstance };

// Default export
export default {
  create: createChatWidget
};