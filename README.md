# Customer Service Bot

A generic customer service bot with hot-swappable knowledge bases and configurable tone/style. Built with FastAPI, React, and vector databases for intelligent document retrieval.

## Features

### 🏗️ **Multi-Tenant Architecture**
- Secure client isolation with UUID-based identification
- Tier-based resource quotas (Free, Starter, Professional, Enterprise)
- API key authentication with scoped permissions
- Client onboarding flow with invitation system

### 📚 **Knowledge Base Management**
- Hot-swappable knowledge bases with versioning
- Multi-format document ingestion (text, HTML, JSON, CSV)
- Intelligent semantic chunking with 5 strategies
- Hybrid search (semantic + keyword) with re-ranking

### 🤖 **Advanced LLM Integration** 
- 6 LLM providers with fallback chains (OpenAI, Anthropic, Mistral, Venice.ai, OpenRouter, Ollama)
- 8 customizable tone styles (Professional, Warm, Casual, Technical, etc.)
- Conversation memory optimization and context management
- Confidence scoring and intelligent escalation

### 💬 **Embeddable Chat Widget**
- React TypeScript widget with 12 components
- Real-time messaging with streaming support
- Mobile-responsive design with touch interactions
- Comprehensive branding customization

### 📊 **Analytics & Monitoring**
- Real-time usage tracking and event aggregation
- Cost attribution and token usage monitoring
- Admin dashboard with comprehensive client management
- Performance metrics and escalation analytics

### 🛡️ **Security & Compliance**
- GDPR-ready with data retention controls
- Secure API key generation with bcrypt hashing
- Tenant isolation and data access validation
- Audit trails and comprehensive logging

## Architecture

- **Backend**: Python FastAPI with LangChain for LLM integration
- **Frontend**: React TypeScript embeddable widget
- **Database**: PostgreSQL for metadata, Redis for sessions
- **Vector Database**: Qdrant for document embeddings and similarity search
- **Containerization**: Docker and Docker Compose for development

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Node.js 18+ (for frontend development)
- Python 3.11+ (for backend development)

### Development Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd customer-service-bot
```

2. Run the development setup script:
```bash
./scripts/dev-setup.sh
```

3. Update the `.env` file with your API keys:
```bash
# LLM Provider API Keys
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
MISTRAL_API_KEY=your-mistral-key
```

4. Start the services:
```bash
docker-compose up -d
```

### Available Services

- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Frontend**: http://localhost:3000
- **Qdrant Dashboard**: http://localhost:6333/dashboard

## Project Structure

```
/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── api/            # API endpoints
│   │   ├── core/           # Core configuration and utilities
│   │   ├── models/         # Database models
│   │   ├── schemas/        # Pydantic schemas
│   │   ├── services/       # Business logic
│   │   └── rag/            # RAG pipeline
│   ├── tests/              # Backend tests
│   └── requirements.txt    # Python dependencies
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── services/       # API services
│   │   └── types/          # TypeScript types
│   └── package.json        # Node.js dependencies
├── infrastructure/         # Deployment configurations
│   ├── docker/
│   ├── kubernetes/
│   └── terraform/
├── scripts/                # Development scripts
└── docs/                   # Documentation
    ├── CLAUDE.md           # Development guidance
    ├── PLANNING.md         # Project roadmap
    └── TASK.md             # Task tracking
```

## Development Commands

### Backend

```bash
# Install dependencies (use uv for faster dependency management)
cd backend
uv pip install -r requirements.txt

# Run development server
uvicorn app.main:app --reload --port 8000

# Run tests with coverage
pytest --cov=app --cov-report=html

# Format and lint code (use ruff for fast Python linting)
ruff check .
ruff format .

# Type checking
mypy app/
```

### Frontend

```bash
# Install dependencies (use pnpm for faster package management)
cd frontend
pnpm install

# Run development server
pnpm dev

# Build for production
pnpm build

# Run tests with coverage
pnpm test:coverage

# Run tests in watch mode
pnpm test:watch

# Type checking
pnpm type-check

# Lint code
pnpm lint
```

### Docker Commands

```bash
# Build and start all services
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild specific service
docker-compose up -d --build backend
```

## Configuration

### Client Configuration

Each client can be configured with:

- **Voice & Persona**: Professional, warm, casual, technical, playful
- **Response Settings**: Confidence thresholds, escalation rules
- **Branding**: Colors, fonts, logos, custom CSS
- **Features**: File uploads, voice messages, analytics, integrations

### Knowledge Base Configuration

- **Chunking Strategy**: Semantic, fixed-size, or paragraph-based
- **Embedding Models**: Configurable embedding providers
- **Search Settings**: Hybrid search, reranking, confidence thresholds
- **Versioning**: Automatic version control and rollback capabilities

## API Endpoints

### Core Endpoints

- `GET /api/v1/health` - Health check
- `POST /api/v1/chat/message` - Send chat message
- `GET /api/v1/clients/{client_id}` - Get client configuration
- `POST /api/v1/knowledge-base/upload` - Upload knowledge base documents
- `GET /api/v1/analytics/usage` - Get usage analytics

### Authentication

API endpoints require client authentication via API key or JWT token.

## Deployment

### Development
```bash
docker-compose up -d
```

### Production
```bash
# Build production images
docker-compose -f docker-compose.prod.yml build

# Deploy with production configuration
docker-compose -f docker-compose.prod.yml up -d
```

### Kubernetes
```bash
# Deploy to Kubernetes
kubectl apply -f infrastructure/kubernetes/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run linting and tests
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support and questions:
- Check the [documentation](docs/)
- Open an issue on GitHub
- Contact the development team

---

**Note**: This is a development version. For production deployment, ensure proper security configurations, API key management, and scaling considerations.