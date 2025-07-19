/**
 * Chat Widget Embed Script
 * 
 * This script can be embedded in client websites to initialize the chat widget
 * Usage: <script src="https://your-domain.com/chat-widget/embed.js" data-client-id="your-client-id"></script>
 */

(function() {
    'use strict';

    // Prevent multiple initializations
    if (window.ChatWidgetLoaded) {
        return;
    }
    window.ChatWidgetLoaded = true;

    // Configuration from script tag attributes
    const script = document.currentScript || document.querySelector('script[data-client-id]');
    const config = {
        clientId: script?.getAttribute('data-client-id'),
        apiUrl: script?.getAttribute('data-api-url') || 'https://api.your-domain.com',
        theme: {
            primaryColor: script?.getAttribute('data-primary-color') || '#007bff',
            secondaryColor: script?.getAttribute('data-secondary-color') || '#6c757d',
            borderRadius: parseInt(script?.getAttribute('data-border-radius')) || 12
        },
        branding: {
            companyName: script?.getAttribute('data-company-name') || 'Customer Support',
            welcomeMessage: script?.getAttribute('data-welcome-message') || 'Hi! How can we help you today?',
            botName: script?.getAttribute('data-bot-name') || 'Assistant',
            avatarUrl: script?.getAttribute('data-avatar-url'),
            logoUrl: script?.getAttribute('data-logo-url')
        },
        behavior: {
            autoOpen: script?.getAttribute('data-auto-open') === 'true',
            openDelay: parseInt(script?.getAttribute('data-open-delay')) || 0,
            persistConversation: script?.getAttribute('data-persist') !== 'false'
        },
        position: {
            side: script?.getAttribute('data-position-side') || 'right',
            bottom: parseInt(script?.getAttribute('data-position-bottom')) || 20,
            horizontal: parseInt(script?.getAttribute('data-position-horizontal')) || 20
        },
        debug: script?.getAttribute('data-debug') === 'true'
    };

    // Validate required configuration
    if (!config.clientId) {
        console.error('[ChatWidget] Error: data-client-id is required');
        return;
    }

    // Widget loader
    const WidgetLoader = {
        loaded: false,
        widget: null,
        
        // CDN URLs for dependencies and widget files
        dependencies: {
            react: 'https://unpkg.com/react@18/umd/react.production.min.js',
            reactDOM: 'https://unpkg.com/react-dom@18/umd/react-dom.production.min.js',
            widget: script?.getAttribute('data-widget-url') || 'https://cdn.your-domain.com/chat-widget.umd.js'
        },

        // Load external script
        loadScript(src, onLoad, onError) {
            const script = document.createElement('script');
            script.src = src;
            script.crossOrigin = 'anonymous';
            script.onload = onLoad;
            script.onerror = onError || (() => console.error(`Failed to load: ${src}`));
            document.head.appendChild(script);
        },

        // Load CSS
        loadCSS(href) {
            const link = document.createElement('link');
            link.rel = 'stylesheet';
            link.href = href;
            document.head.appendChild(link);
        },

        // Initialize widget
        async init() {
            if (this.loaded) return;

            try {
                // Load dependencies
                await this.loadDependencies();
                
                // Load widget
                await this.loadWidget();
                
                // Initialize widget
                this.createWidget();
                
                this.loaded = true;
                console.log('[ChatWidget] Loaded successfully');
            } catch (error) {
                console.error('[ChatWidget] Failed to load:', error);
            }
        },

        // Load React dependencies
        loadDependencies() {
            return new Promise((resolve, reject) => {
                // Check if React is already loaded
                if (window.React && window.ReactDOM) {
                    resolve();
                    return;
                }

                let loaded = 0;
                const total = 2;
                
                const checkComplete = () => {
                    loaded++;
                    if (loaded === total) {
                        if (window.React && window.ReactDOM) {
                            resolve();
                        } else {
                            reject(new Error('React dependencies not loaded'));
                        }
                    }
                };

                // Load React
                this.loadScript(this.dependencies.react, checkComplete, reject);
                
                // Load ReactDOM
                this.loadScript(this.dependencies.reactDOM, checkComplete, reject);
            });
        },

        // Load widget script
        loadWidget() {
            return new Promise((resolve, reject) => {
                this.loadScript(
                    this.dependencies.widget,
                    () => {
                        if (window.ChatWidget) {
                            resolve();
                        } else {
                            reject(new Error('ChatWidget not found'));
                        }
                    },
                    reject
                );
            });
        },

        // Create widget instance
        createWidget() {
            if (!window.ChatWidget || !window.ChatWidget.create) {
                throw new Error('ChatWidget.create not available');
            }

            this.widget = window.ChatWidget.create(config);
            
            // Expose widget globally for debugging
            if (config.debug) {
                window.debugChatWidget = this.widget;
            }

            // Add event listeners
            this.setupEventListeners();
        },

        // Setup event listeners
        setupEventListeners() {
            if (!this.widget) return;

            // Listen to widget events
            this.widget.addEventListener('widget:open', () => {
                if (config.debug) console.log('[ChatWidget] Opened');
            });

            this.widget.addEventListener('widget:close', () => {
                if (config.debug) console.log('[ChatWidget] Closed');
            });

            this.widget.addEventListener('message:receive', (data) => {
                if (config.debug) console.log('[ChatWidget] Message received:', data);
            });

            this.widget.addEventListener('escalation:triggered', (data) => {
                if (config.debug) console.log('[ChatWidget] Escalation triggered:', data);
                
                // Fire custom event for website integration
                window.dispatchEvent(new CustomEvent('chatWidgetEscalation', {
                    detail: data
                }));
            });

            this.widget.addEventListener('error:occurred', (data) => {
                console.error('[ChatWidget] Error:', data);
            });
        }
    };

    // Auto-initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => WidgetLoader.init());
    } else {
        WidgetLoader.init();
    }

    // Expose loader for manual control
    window.ChatWidgetLoader = WidgetLoader;

    // Expose configuration API
    window.ChatWidgetConfig = {
        update: (newConfig) => {
            if (WidgetLoader.widget) {
                WidgetLoader.widget.updateConfig(newConfig);
            }
        },
        
        getWidget: () => WidgetLoader.widget,
        
        reload: () => {
            if (WidgetLoader.widget) {
                WidgetLoader.widget.destroy();
            }
            WidgetLoader.loaded = false;
            WidgetLoader.init();
        }
    };

})();