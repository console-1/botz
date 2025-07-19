# Customer Service Bot Widget

A React TypeScript widget for embedding AI-powered customer service chat functionality into any website.

## Features

- 💬 **Real-time Chat Interface** - Smooth, responsive chat experience
- 🎨 **Fully Customizable** - Themes, branding, colors, and positioning
- 🤖 **AI-Powered Responses** - Intelligent responses with confidence scoring
- 📈 **Smart Escalation** - Automatic escalation to human agents when needed
- 📱 **Mobile Responsive** - Works perfectly on all devices
- 🔧 **Developer Friendly** - Easy integration with comprehensive API
- ⚡ **Performance Optimized** - Lightweight and fast loading
- 🎯 **Accessibility** - WCAG compliant with keyboard navigation

## Quick Start

### 1. Script Tag Integration (Easiest)

Add this script tag to your website:

```html
<script 
  src="https://cdn.your-domain.com/chat-widget/embed.js"
  data-client-id="your-client-id"
  data-api-url="https://api.your-domain.com"
  data-company-name="Your Company"
  data-primary-color="#007bff"
  data-welcome-message="Hi! How can we help you today?"
  async
></script>
```

### 2. NPM Package Integration

```bash
npm install @your-company/chat-widget
```

```javascript
import { createChatWidget } from '@your-company/chat-widget';

const widget = createChatWidget({
  clientId: 'your-client-id',
  apiUrl: 'https://api.your-domain.com',
  theme: {
    primaryColor: '#007bff',
    backgroundColor: '#ffffff'
  },
  branding: {
    companyName: 'Your Company',
    welcomeMessage: 'Hi! How can we help you today?'
  }
});
```

### 3. React Component Integration

```jsx
import React from 'react';
import { ChatWidget } from '@your-company/chat-widget';

function App() {
  const config = {
    clientId: 'your-client-id',
    apiUrl: 'https://api.your-domain.com',
    theme: {
      primaryColor: '#007bff'
    },
    branding: {
      companyName: 'Your Company'
    }
  };

  return (
    <div>
      <h1>Your Website</h1>
      <ChatWidget config={config} />
    </div>
  );
}
```

## Configuration Options

### Required Configuration

```javascript
{
  clientId: 'your-unique-client-id',
  apiUrl: 'https://your-api-endpoint.com'
}
```

### Theme Configuration

```javascript
{
  theme: {
    primaryColor: '#007bff',      // Main brand color
    secondaryColor: '#6c757d',    // Secondary color
    backgroundColor: '#ffffff',   // Background color
    textColor: '#333333',         // Text color
    borderRadius: 12,             // Border radius in pixels
    fontFamily: 'system-ui',      // Font family
    fontSize: 14                  // Font size in pixels
  }
}
```

### Branding Configuration

```javascript
{
  branding: {
    companyName: 'Your Company',
    logoUrl: 'https://your-domain.com/logo.png',
    welcomeMessage: 'Hi! How can we help you today?',
    placeholderText: 'Type your message...',
    botName: 'Assistant',
    avatarUrl: 'https://your-domain.com/bot-avatar.png',
    poweredByText: 'Powered by Your Company AI'
  }
}
```

### Behavior Configuration

```javascript
{
  behavior: {
    autoOpen: false,              // Auto-open widget on page load
    openDelay: 2000,              // Delay before auto-opening (ms)
    showTypingIndicator: true,    // Show typing indicator
    enableSound: false,           // Enable notification sounds
    enableEmojis: true,           // Enable emoji picker
    maxMessageLength: 1000,       // Max message length
    enableFileUpload: false,      // Enable file attachments
    enableVoiceInput: false,      // Enable voice input
    persistConversation: true     // Persist conversation across page loads
  }
}
```

### Position Configuration

```javascript
{
  position: {
    side: 'right',                // 'left' or 'right'
    bottom: 20,                   // Distance from bottom (px)
    horizontal: 20                // Distance from side (px)
  }
}
```

## API Methods

```javascript
const widget = createChatWidget(config);

// Control widget visibility
widget.open();
widget.close();
widget.toggle();

// Send messages programmatically
widget.sendMessage('Hello from the website!');

// Set user information
widget.setUser({
  id: 'user123',
  name: 'John Doe',
  email: 'john@example.com'
});

// Update configuration
widget.updateConfig({
  theme: {
    primaryColor: '#28a745'
  }
});

// Listen to events
widget.addEventListener('widget:open', () => {
  console.log('Widget opened');
});

widget.addEventListener('message:receive', (data) => {
  console.log('Message received:', data.payload);
});

widget.addEventListener('escalation:triggered', (data) => {
  console.log('Escalated to human agent:', data.payload);
});

// Cleanup
widget.destroy();
```

## Events

The widget emits various events that you can listen to:

- `widget:open` - Widget is opened
- `widget:close` - Widget is closed
- `widget:minimize` - Widget is minimized
- `widget:maximize` - Widget is maximized
- `message:send` - User sends a message
- `message:receive` - Bot sends a response
- `conversation:start` - New conversation started
- `conversation:end` - Conversation ended
- `escalation:triggered` - Conversation escalated to human
- `escalation:resolved` - Escalation resolved
- `typing:start` - Typing indicator shown
- `typing:stop` - Typing indicator hidden
- `connection:connect` - Connected to API
- `connection:disconnect` - Disconnected from API
- `error:occurred` - Error occurred

## Development

### Setup

```bash
cd frontend
pnpm install
```

### Development Server

```bash
pnpm dev
```

### Build for Production

```bash
pnpm build
```

### Run Tests

```bash
pnpm test
pnpm test:coverage
```

### Lint and Format

```bash
pnpm lint
pnpm lint:fix
```

## File Structure

```
frontend/
├── src/
│   ├── components/           # React components
│   │   ├── ChatWidget.tsx   # Main widget component
│   │   ├── ChatButton.tsx   # Floating chat button
│   │   ├── ChatHeader.tsx   # Chat window header
│   │   ├── MessageList.tsx  # Message history
│   │   ├── MessageBubble.tsx# Individual messages
│   │   ├── MessageInput.tsx # Message input field
│   │   ├── TypingIndicator.tsx
│   │   ├── LoadingSpinner.tsx
│   │   ├── ErrorBoundary.tsx
│   │   └── EscalationBanner.tsx
│   ├── services/            # API and state management
│   │   ├── chatApi.ts       # API communication
│   │   └── widgetState.ts   # State management
│   ├── types/               # TypeScript types
│   │   └── chat.ts          # Core type definitions
│   ├── widget/              # Embeddable scripts
│   │   ├── embed.html       # Demo page
│   │   └── embed.js         # Embed script
│   └── index.ts             # Main entry point
├── dist/                    # Built files
├── package.json
├── vite.config.ts          # Build configuration
└── README.md
```

## Browser Support

- Chrome 70+
- Firefox 63+
- Safari 12+
- Edge 79+
- Mobile browsers (iOS Safari, Chrome Mobile)

## Performance

- Initial bundle size: ~50KB gzipped
- React dependencies: External (CDN or existing)
- Lazy loading: Components loaded on demand
- Caching: Aggressive caching for static assets

## Security

- XSS protection via React's built-in sanitization
- CORS configuration for API calls
- No sensitive data stored in localStorage
- Optional CSP compatibility

## Accessibility

- WCAG 2.1 AA compliant
- Keyboard navigation support
- Screen reader compatible
- High contrast mode support
- Focus management
- ARIA labels and roles

## Customization Examples

### E-commerce Theme

```javascript
const ecommerceTheme = {
  theme: {
    primaryColor: '#ff6b35',
    secondaryColor: '#004e89',
    borderRadius: 8
  },
  branding: {
    companyName: 'ShopMart',
    welcomeMessage: 'Welcome to ShopMart! Need help finding something?',
    botName: 'ShopBot',
    poweredByText: 'Powered by ShopMart AI'
  }
};
```

### Professional Services Theme

```javascript
const professionalTheme = {
  theme: {
    primaryColor: '#2c3e50',
    secondaryColor: '#7f8c8d',
    backgroundColor: '#ecf0f1',
    borderRadius: 4
  },
  branding: {
    companyName: 'Legal Associates',
    welcomeMessage: 'How can we assist you with your legal needs?',
    botName: 'Legal Assistant'
  }
};
```

### Healthcare Theme

```javascript
const healthcareTheme = {
  theme: {
    primaryColor: '#27ae60',
    secondaryColor: '#2ecc71',
    borderRadius: 16
  },
  branding: {
    companyName: 'MediCare Plus',
    welcomeMessage: 'Hi! How can we help with your healthcare questions?',
    botName: 'HealthBot'
  }
};
```

## Support

For support and questions:

- 📖 [Documentation](https://docs.your-domain.com/chat-widget)
- 🐛 [Issues](https://github.com/your-company/chat-widget/issues)
- 💬 [Support Chat](https://your-domain.com/support)
- 📧 [Email Support](mailto:support@your-domain.com)

## License

MIT License - see [LICENSE](LICENSE) file for details.