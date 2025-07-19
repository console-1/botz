# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**CRITICAL**: All documentation files must be kept current and consulted before any development work:

1. **PLANNING.md** - Always check for current implementation roadmap
2. **TASK.md** - Update with all tasks (past, present, future) as living document
3. **CLAUDE.md** - This file, updated with new patterns and learnings
4. **README.md** - User-facing documentation when created

The documentation represents the single source of truth for the project and must never be neglected.

**CRITICAL**: Testing and tooling requirements that must be followed:

1. **Thorough testing after each new feature** - Every feature must have comprehensive tests before being considered complete
2. **Python tooling** - Use uv for dependency management and ruff for linting/formatting
3. **TypeScript tooling** - Use pnpm for package management and jest for testing
4. **Test coverage** - Maintain minimum 80% code coverage for all new features
5. **Integration testing** - Test all API endpoints and database operations
6. **End-to-end testing** - Test complete user workflows before deployment

**CRITICAL**: Version control workflow that must be followed:

1. **Commit after every completed task or feature** - Create solid rollback points for development safety
2. **Remote repository** - https://github.com/console-1/botz.git
3. **Commit message format** - Use descriptive messages following conventional commits format
4. **Push immediately after commit** - Ensure remote backup of all progress
5. **Branch protection** - Use feature branches for major changes, commit directly to main for small tasks
6. **Never leave uncommitted changes** - Always commit working state before ending development session

## Project Overview

This is a generic customer service bot with hot-swappable knowledge bases and configurable tone/style. The bot uses RAG (Retrieval-Augmented Generation) architecture with vector databases, supports multi-tenant deployment, and provides an embeddable React widget for client websites.

## Core Architecture

### Backend Stack
- **Language**: Python (FastAPI + LangChain) for ML ecosystem integration
- **Vector Database**: Qdrant (primary) or Chroma (simpler alternative)
- **Traditional Database**: PostgreSQL for metadata, Redis for sessions
- **Storage**: S3/Azure Blob for source files, vector DB for embeddings

### Frontend Stack  
- **Framework**: React + TypeScript
- **Deployment**: NPM package for embeddable widget
- **UI Pattern**: Floating chat button that expands to modal

### LLM Integration
- **Primary Models**: Venice.ai (OpenAI/Anthropic models), OpenRouter.ai fallback
- **Provider Strategy**: Multi-provider abstraction for cost optimization and reliability
- **Fallback Chain**: Primary Venice.ai → OpenRouter.ai → Static FAQ → Human escalation
- **Budget Target**: ~$0.30 per 100K tokens per client
- **Response Time**: < 1s target (300-800ms typical)
- **Week 3 Focus**: Implement LLM abstraction layer and provider integration

## Key Development Principles

### Multi-Tenancy
- Each client has logical isolation via DB schemas and vector namespaces
- Client identification via API key or subdomain slug
- Per-client configuration for tone, branding, and knowledge base
- Resource quotas per tenant to prevent abuse

### Knowledge Base Management
- **Format**: Markdown/HTML chunked into embeddings
- **Chunking**: Semantic chunking over fixed-size for better retrieval
- **Search**: Hybrid search (semantic + keyword) for factual queries
- **Versioning**: Content fingerprinting and prompt versioning for A/B testing
- **Updates**: Near-real-time via streaming or nightly batch refresh

### Security & Compliance
- **Privacy**: GDPR by default, optional CCPA/POPIA
- **Encryption**: TLS 1.3 in transit, AES-256 at rest
- **Data Handling**: 30-day chat log retention, PII masking via regex
- **Monitoring**: Audit logging for all model interactions

## Development Workflow

### Essential Commands
```bash
# Backend (Python/FastAPI) - Use uv for dependency management
uv pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Python linting and formatting with ruff
ruff check .
ruff format .

# Backend testing
pytest --cov=app --cov-report=html

# Frontend (React/TypeScript) - Use pnpm for package management
pnpm install
pnpm dev
pnpm build
pnpm test

# Frontend testing with jest
pnpm test --coverage
pnpm test:watch

# Vector Database (Qdrant)
docker run -p 6333:6333 qdrant/qdrant

# Database migrations
alembic upgrade head

# Git workflow (CRITICAL: Use after every completed task/feature)
git add .
git commit -m "feat: descriptive message using conventional commits"
git push origin main
```

### Testing Strategy
**CRITICAL**: Every feature must be thoroughly tested before being considered complete.

#### Backend Testing (Python + pytest)
- **Unit tests**: RAG pipeline components, business logic, utilities
- **Integration tests**: API endpoints, database operations, multi-tenant isolation
- **Load testing**: 500 concurrent sessions target with realistic data
- **Coverage requirement**: Minimum 80% code coverage for all new features

#### Frontend Testing (TypeScript + Jest)
- **Unit tests**: React components, utility functions, API services
- **Integration tests**: User workflows, widget embedding, API integration
- **E2E tests**: Complete user journeys from widget to response
- **Coverage requirement**: Minimum 80% code coverage for all new features

#### System Testing
- **A/B testing**: Infrastructure for prompt variants and configuration changes
- **Performance testing**: Response times, memory usage, database queries
- **Security testing**: Input validation, authentication, data privacy
- **Compliance testing**: GDPR, data retention, audit logging

## Critical Implementation Details

### RAG Pipeline Architecture
1. **Document Ingestion**: Chunk → Embed → Store with metadata
2. **Query Processing**: Semantic search → Context ranking → LLM generation
3. **Response Generation**: Template injection with client tone/style
4. **Fallback Handling**: Confidence thresholds trigger escalation

### Client Configuration Schema
```json
{
  "client_id": "acme-corp",
  "voice": "professional",
  "persona": "warm",
  "no_data_reply": "I don't have information about that...",
  "escalation_threshold": 0.7,
  "branding": {
    "primary_color": "#007bff",
    "logo_url": "https://example.com/logo.png"
  }
}
```

### Conversation Flow
- Maintain rolling context windows (last N turns or 5K tokens)
- Conversation summarization when approaching limits
- Support for clarifying questions when required fields missing
- Structured outputs (JSON schema) for escalation integration

## Performance Optimization

### Caching Strategy
- Pre-compute responses for top 100 FAQ combinations per client
- Predictive caching based on user behavior patterns
- Response clustering for similar queries with variable slots

### Scalability Targets
- 500 simultaneous sessions per pod
- Horizontal scaling via Kubernetes HPA
- Support for 50+ clients within 12 months
- Handle 1K-10K messages/month per client

## Monitoring & Analytics

### Key Metrics
- Response latency and accuracy
- Cost per conversation and per client
- User satisfaction scores (CSAT)
- Escalation rates and resolution success
- Token usage and API costs

### Logging Requirements
- All model interactions with audit trail
- Conversation replay capability for debugging
- Cost attribution per client/conversation/query type
- Quality scoring for response helpfulness

## Essential File Structure

```
/
├── backend/
│   ├── app/
│   │   ├── core/           # FastAPI app, auth, config
│   │   ├── rag/            # RAG pipeline, embeddings
│   │   ├── models/         # Database models
│   │   ├── services/       # Business logic
│   │   └── api/            # REST endpoints
│   ├── tests/
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/     # Chat widget, UI components
│   │   ├── services/       # API client
│   │   └── types/          # TypeScript definitions
│   ├── package.json
│   └── widget/             # Embeddable script
├── infrastructure/
│   ├── docker/
│   ├── kubernetes/
│   └── terraform/
├── docs/
│   ├── CLAUDE.md          # This file
│   ├── PLANNING.md        # Project roadmap
│   ├── TASK.md           # Living task tracker
│   └── context/
│       ├── Q.md          # Requirements questions
│       └── A.md          # Detailed answers
└── scripts/
    ├── deploy.sh
    ├── migrate.sh
    └── test.sh
```

## Documentation Requirements

**CRITICAL**: All documentation files must be kept current and consulted before any development work:

1. **PLANNING.md** - Always check for current implementation roadmap
2. **TASK.md** - Update with all tasks (past, present, future) as living document
3. **CLAUDE.md** - This file, updated with new patterns and learnings
4. **README.md** - User-facing documentation when created

The documentation represents the single source of truth for the project and must never be neglected.

## Development Priorities

### MVP (6 weeks)
1. RAG pipeline with vector search
2. Multi-tenant client isolation
3. Embeddable React widget
4. Basic admin dashboard
5. GDPR compliance and logging

### Phase 2 Features
- A/B testing infrastructure
- Sentiment analysis pipeline
- Advanced escalation logic
- Rich media support
- Voice integration

### Architectural Abstractions
- LLM interface abstraction for model switching
- Storage abstraction for multi-cloud deployment
- Authentication abstraction for various client needs
- Monitoring abstraction for different analytics providers

## Common Pitfalls to Avoid

1. **Vector Search Quality**: Chunking strategy is critical for retrieval performance
2. **Cost Optimization**: Token usage can spiral without careful prompt engineering
3. **Model Switching**: Abstract LLM interface early for flexibility
4. **Compliance Complexity**: GDPR implementation often takes longer than expected
5. **Feature Creep**: Focus on customer service excellence over cool AI features
6. **Vendor Lock-in**: Design for multi-cloud and multi-model flexibility

**CRITICAL**: All documentation files must be kept current and consulted before any development work:

1. **PLANNING.md** - Always check for current implementation roadmap
2. **TASK.md** - Update with all tasks (past, present, future) as living document
3. **CLAUDE.md** - This file, updated with new patterns and learnings
4. **README.md** - User-facing documentation when created

The documentation represents the single source of truth for the project and must never be neglected.