/**
 * Chat API Service
 * 
 * Handles all communication with the customer service bot backend API
 */

import axios, { AxiosInstance, AxiosResponse } from 'axios';
import {
  MessageRequest,
  MessageResponse,
  ConversationHistory,
  ApiResponse,
  WidgetConfig,
  StreamingChunk,
  EscalationInfo
} from '../types/chat';

export class ChatApiService {
  private api: AxiosInstance;
  private baseUrl: string;
  private clientId: string;

  constructor(config: { apiUrl: string; clientId: string; timeout?: number }) {
    this.baseUrl = config.apiUrl;
    this.clientId = config.clientId;

    this.api = axios.create({
      baseURL: this.baseUrl,
      timeout: config.timeout || 30000,
      headers: {
        'Content-Type': 'application/json',
        'X-Client-ID': this.clientId,
      },
    });

    // Request interceptor for logging and auth
    this.api.interceptors.request.use(
      (config) => {
        console.log(`[ChatAPI] ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => {
        console.error('[ChatAPI] Request error:', error);
        return Promise.reject(error);
      }
    );

    // Response interceptor for error handling
    this.api.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('[ChatAPI] Response error:', error);
        return Promise.reject(this.handleApiError(error));
      }
    );
  }

  /**
   * Start a new conversation
   */
  async startConversation(sessionId: string, userId?: string, metadata?: Record<string, any>): Promise<MessageResponse> {
    try {
      const response: AxiosResponse<MessageResponse> = await this.api.post('/api/v1/chat/v2/conversations/start', {
        client_id: this.clientId,
        session_id: sessionId,
        user_id: userId,
        metadata,
      });

      return response.data;
    } catch (error) {
      throw this.handleApiError(error);
    }
  }

  /**
   * Send a message and get response
   */
  async sendMessage(request: MessageRequest): Promise<MessageResponse> {
    try {
      const response: AxiosResponse<MessageResponse> = await this.api.post('/api/v1/chat/v2/message', {
        message: request.message,
        client_id: request.clientId,
        session_id: request.sessionId,
        conversation_id: request.conversationId,
        user_id: request.userId,
        metadata: request.metadata,
      });

      return response.data;
    } catch (error) {
      throw this.handleApiError(error);
    }
  }

  /**
   * Send message with streaming response
   */
  async sendMessageStream(
    request: MessageRequest,
    onChunk: (chunk: StreamingChunk) => void,
    onComplete: (response: MessageResponse) => void,
    onError: (error: Error) => void
  ): Promise<void> {
    try {
      const response = await fetch(`${this.baseUrl}/api/v1/chat/v2/message/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Client-ID': this.clientId,
        },
        body: JSON.stringify({
          message: request.message,
          client_id: request.clientId,
          session_id: request.sessionId,
          conversation_id: request.conversationId,
          user_id: request.userId,
          metadata: request.metadata,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('Failed to get response reader');
      }

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.done) {
                // Final response with metadata
                if (data.metadata) {
                  onComplete({
                    message: '', // Content was streamed
                    conversationId: data.conversation_id,
                    responseType: data.metadata.response_type,
                    confidenceScore: data.metadata.confidence_score,
                    shouldEscalate: data.metadata.should_escalate,
                    processingTimeMs: data.metadata.processing_time_ms,
                    tokensUsed: data.metadata.tokens_used,
                    costUsd: data.metadata.cost_usd,
                    metadata: data.metadata,
                  });
                }
              } else {
                // Content chunk
                onChunk({
                  content: data.content,
                  done: data.done,
                  conversationId: data.conversation_id,
                  metadata: data.metadata,
                });
              }
            } catch (parseError) {
              console.warn('[ChatAPI] Failed to parse streaming data:', parseError);
            }
          }
        }
      }
    } catch (error) {
      onError(this.handleApiError(error));
    }
  }

  /**
   * End a conversation
   */
  async endConversation(conversationId: string): Promise<MessageResponse> {
    try {
      const response: AxiosResponse<MessageResponse> = await this.api.post(
        `/api/v1/chat/v2/conversations/${conversationId}/end`,
        { client_id: this.clientId }
      );

      return response.data;
    } catch (error) {
      throw this.handleApiError(error);
    }
  }

  /**
   * Get conversation history
   */
  async getConversationHistory(conversationId: string, limit?: number): Promise<ConversationHistory> {
    try {
      const params = limit ? { limit } : {};
      const response: AxiosResponse<ConversationHistory> = await this.api.get(
        `/api/v1/chat/v2/conversations/${conversationId}/history`,
        { params }
      );

      return response.data;
    } catch (error) {
      throw this.handleApiError(error);
    }
  }

  /**
   * Get active escalations for client
   */
  async getEscalations(): Promise<EscalationInfo[]> {
    try {
      const response: AxiosResponse<EscalationInfo[]> = await this.api.get(
        `/api/v1/chat/v2/clients/${this.clientId}/escalations`
      );

      return response.data;
    } catch (error) {
      throw this.handleApiError(error);
    }
  }

  /**
   * Resolve an escalation
   */
  async resolveEscalation(escalationId: string, resolutionNotes: string, agentId?: string): Promise<void> {
    try {
      await this.api.post(`/api/v1/chat/v2/escalations/${escalationId}/resolve`, {
        resolution_notes: resolutionNotes,
        agent_id: agentId,
      });
    } catch (error) {
      throw this.handleApiError(error);
    }
  }

  /**
   * Update client configuration
   */
  async updateClientConfig(config: Partial<WidgetConfig>): Promise<void> {
    try {
      await this.api.post(`/api/v1/chat/v2/clients/${this.clientId}/config`, config);
    } catch (error) {
      throw this.handleApiError(error);
    }
  }

  /**
   * Get service health status
   */
  async getHealthStatus(): Promise<{ status: string; timestamp: string; [key: string]: any }> {
    try {
      const response = await this.api.get('/api/v1/chat/v2/health');
      return response.data;
    } catch (error) {
      throw this.handleApiError(error);
    }
  }

  /**
   * Get service statistics
   */
  async getStats(): Promise<Record<string, any>> {
    try {
      const response = await this.api.get('/api/v1/chat/v2/stats');
      return response.data;
    } catch (error) {
      throw this.handleApiError(error);
    }
  }

  /**
   * Handle API errors consistently
   */
  private handleApiError(error: any): Error {
    if (error.response) {
      // Server responded with error status
      const status = error.response.status;
      const message = error.response.data?.detail || error.response.data?.message || error.response.statusText;
      
      return new Error(`API Error ${status}: ${message}`);
    } else if (error.request) {
      // Request was made but no response received
      return new Error('Network Error: No response from server');
    } else {
      // Something else happened
      return new Error(`Request Error: ${error.message}`);
    }
  }

  /**
   * Check API connectivity
   */
  async testConnection(): Promise<boolean> {
    try {
      await this.getHealthStatus();
      return true;
    } catch (error) {
      console.error('[ChatAPI] Connection test failed:', error);
      return false;
    }
  }

  /**
   * Update API configuration
   */
  updateConfig(config: { apiUrl?: string; clientId?: string; timeout?: number }): void {
    if (config.apiUrl) {
      this.baseUrl = config.apiUrl;
      this.api.defaults.baseURL = config.apiUrl;
    }
    
    if (config.clientId) {
      this.clientId = config.clientId;
      this.api.defaults.headers['X-Client-ID'] = config.clientId;
    }
    
    if (config.timeout) {
      this.api.defaults.timeout = config.timeout;
    }
  }

  /**
   * Get current configuration
   */
  getConfig(): { apiUrl: string; clientId: string; timeout: number } {
    return {
      apiUrl: this.baseUrl,
      clientId: this.clientId,
      timeout: this.api.defaults.timeout as number,
    };
  }
}

// Export a factory function for creating API instances
export const createChatApiService = (config: { apiUrl: string; clientId: string; timeout?: number }): ChatApiService => {
  return new ChatApiService(config);
};

// Export default instance (will be configured when widget initializes)
let defaultApiService: ChatApiService | null = null;

export const setDefaultApiService = (service: ChatApiService): void => {
  defaultApiService = service;
};

export const getDefaultApiService = (): ChatApiService => {
  if (!defaultApiService) {
    throw new Error('Default API service not initialized. Call setDefaultApiService first.');
  }
  return defaultApiService;
};