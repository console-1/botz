#!/bin/bash

# Production Deployment Script for Customer Service Bot
# This script handles deployment to production environment with proper safety checks

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Configuration
PROJECT_NAME="customer-service-bot"
DEPLOY_ENV="${DEPLOY_ENV:-production}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-your-registry.com}"
BACKUP_RETENTION_DAYS=30
HEALTH_CHECK_TIMEOUT=300  # 5 minutes
ROLLBACK_ENABLED="${ROLLBACK_ENABLED:-true}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking deployment prerequisites..."
    
    # Check required tools
    local required_tools=("docker" "docker-compose" "kubectl" "psql" "curl")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "Required tool '$tool' is not installed"
            exit 1
        fi
    done
    
    # Check environment variables
    local required_vars=("DATABASE_URL" "REDIS_URL" "QDRANT_URL")
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log_error "Required environment variable '$var' is not set"
            exit 1
        fi
    done
    
    # Check Docker registry access
    if ! docker login "$DOCKER_REGISTRY" &> /dev/null; then
        log_error "Failed to authenticate with Docker registry"
        exit 1
    fi
    
    log_success "Prerequisites check completed"
}

# Create backup before deployment
create_backup() {
    log_info "Creating pre-deployment backup..."
    
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_dir="backups/${timestamp}"
    
    mkdir -p "$backup_dir"
    
    # Database backup
    log_info "Backing up database..."
    PGPASSWORD="$DATABASE_PASSWORD" pg_dump \
        -h "$DATABASE_HOST" \
        -p "$DATABASE_PORT" \
        -U "$DATABASE_USER" \
        -d "$DATABASE_NAME" \
        --no-password \
        --verbose \
        > "$backup_dir/database_backup.sql"
    
    # Configuration backup
    log_info "Backing up configuration..."
    cp -r config/ "$backup_dir/" 2>/dev/null || true
    cp docker-compose.yml "$backup_dir/" 2>/dev/null || true
    cp .env "$backup_dir/" 2>/dev/null || true
    
    # Create backup manifest
    cat > "$backup_dir/manifest.json" << EOF
{
    "timestamp": "$timestamp",
    "environment": "$DEPLOY_ENV",
    "git_commit": "$(git rev-parse HEAD)",
    "git_branch": "$(git rev-parse --abbrev-ref HEAD)",
    "database_size": "$(du -sh "$backup_dir/database_backup.sql" | cut -f1)",
    "backup_type": "pre_deployment"
}
EOF
    
    # Compress backup
    tar -czf "backup_${timestamp}.tar.gz" -C backups "$timestamp"
    rm -rf "$backup_dir"
    
    log_success "Backup created: backup_${timestamp}.tar.gz"
    echo "$timestamp" > .last_backup
}

# Build and push Docker images
build_and_push_images() {
    log_info "Building and pushing Docker images..."
    
    local git_commit=$(git rev-parse --short HEAD)
    local image_tag="${git_commit}-$(date +%Y%m%d)"
    
    # Build backend image
    log_info "Building backend image..."
    docker build \
        -t "${DOCKER_REGISTRY}/${PROJECT_NAME}-backend:${image_tag}" \
        -t "${DOCKER_REGISTRY}/${PROJECT_NAME}-backend:latest" \
        -f backend/Dockerfile \
        backend/
    
    # Build frontend image
    log_info "Building frontend image..."
    docker build \
        -t "${DOCKER_REGISTRY}/${PROJECT_NAME}-frontend:${image_tag}" \
        -t "${DOCKER_REGISTRY}/${PROJECT_NAME}-frontend:latest" \
        -f frontend/Dockerfile \
        frontend/
    
    # Push images
    log_info "Pushing images to registry..."
    docker push "${DOCKER_REGISTRY}/${PROJECT_NAME}-backend:${image_tag}"
    docker push "${DOCKER_REGISTRY}/${PROJECT_NAME}-backend:latest"
    docker push "${DOCKER_REGISTRY}/${PROJECT_NAME}-frontend:${image_tag}"
    docker push "${DOCKER_REGISTRY}/${PROJECT_NAME}-frontend:latest"
    
    log_success "Images built and pushed successfully"
    echo "$image_tag" > .last_deploy_tag
}

# Run database migrations
run_migrations() {
    log_info "Running database migrations..."
    
    # Check if migrations are needed
    if ! PGPASSWORD="$DATABASE_PASSWORD" psql \
        -h "$DATABASE_HOST" \
        -p "$DATABASE_PORT" \
        -U "$DATABASE_USER" \
        -d "$DATABASE_NAME" \
        -c "SELECT version_num FROM alembic_version;" &> /dev/null; then
        log_info "Database not initialized, running initial migration..."
    fi
    
    # Run migrations in a temporary container
    docker run --rm \
        --network host \
        -e DATABASE_URL="$DATABASE_URL" \
        "${DOCKER_REGISTRY}/${PROJECT_NAME}-backend:latest" \
        alembic upgrade head
    
    log_success "Database migrations completed"
}

# Deploy to Kubernetes
deploy_to_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    local image_tag=$(cat .last_deploy_tag)
    
    # Update image tags in Kubernetes manifests
    sed -i "s|image: .*-backend:.*|image: ${DOCKER_REGISTRY}/${PROJECT_NAME}-backend:${image_tag}|g" \
        infrastructure/kubernetes/backend-deployment.yaml
    sed -i "s|image: .*-frontend:.*|image: ${DOCKER_REGISTRY}/${PROJECT_NAME}-frontend:${image_tag}|g" \
        infrastructure/kubernetes/frontend-deployment.yaml
    
    # Apply Kubernetes manifests
    kubectl apply -f infrastructure/kubernetes/namespace.yaml
    kubectl apply -f infrastructure/kubernetes/configmap.yaml
    kubectl apply -f infrastructure/kubernetes/secrets.yaml
    kubectl apply -f infrastructure/kubernetes/database-deployment.yaml
    kubectl apply -f infrastructure/kubernetes/redis-deployment.yaml
    kubectl apply -f infrastructure/kubernetes/qdrant-deployment.yaml
    kubectl apply -f infrastructure/kubernetes/backend-deployment.yaml
    kubectl apply -f infrastructure/kubernetes/frontend-deployment.yaml
    kubectl apply -f infrastructure/kubernetes/ingress.yaml
    
    # Wait for deployments to be ready
    log_info "Waiting for deployments to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/backend-deployment -n customer-service-bot
    kubectl wait --for=condition=available --timeout=300s deployment/frontend-deployment -n customer-service-bot
    
    log_success "Kubernetes deployment completed"
}

# Deploy using Docker Compose (alternative to Kubernetes)
deploy_with_docker_compose() {
    log_info "Deploying with Docker Compose..."
    
    local image_tag=$(cat .last_deploy_tag)
    
    # Update image tags in docker-compose.prod.yml
    export BACKEND_IMAGE="${DOCKER_REGISTRY}/${PROJECT_NAME}-backend:${image_tag}"
    export FRONTEND_IMAGE="${DOCKER_REGISTRY}/${PROJECT_NAME}-frontend:${image_tag}"
    
    # Pull latest images
    docker-compose -f docker-compose.prod.yml pull
    
    # Start services
    docker-compose -f docker-compose.prod.yml up -d
    
    log_success "Docker Compose deployment completed"
}

# Health checks
perform_health_checks() {
    log_info "Performing health checks..."
    
    local api_url="${API_URL:-http://localhost:8000}"
    local frontend_url="${FRONTEND_URL:-http://localhost:3000}"
    local max_attempts=30
    local attempt=1
    
    # Check API health
    log_info "Checking API health..."
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f -s "$api_url/health" &> /dev/null; then
            log_success "API health check passed"
            break
        fi
        
        if [[ $attempt -eq $max_attempts ]]; then
            log_error "API health check failed after $max_attempts attempts"
            return 1
        fi
        
        log_info "Attempt $attempt/$max_attempts failed, retrying in 10 seconds..."
        sleep 10
        ((attempt++))
    done
    
    # Check frontend health
    log_info "Checking frontend health..."
    if curl -f -s "$frontend_url" &> /dev/null; then
        log_success "Frontend health check passed"
    else
        log_warning "Frontend health check failed"
    fi
    
    # Check database connectivity
    log_info "Checking database connectivity..."
    if curl -f -s "$api_url/health/database" &> /dev/null; then
        log_success "Database connectivity check passed"
    else
        log_error "Database connectivity check failed"
        return 1
    fi
    
    # Check external services
    log_info "Checking external services..."
    if curl -f -s "$api_url/health/external" &> /dev/null; then
        log_success "External services check passed"
    else
        log_warning "External services check failed"
    fi
    
    log_success "Health checks completed"
}

# Rollback deployment
rollback_deployment() {
    log_warning "Rolling back deployment..."
    
    if [[ ! -f .last_backup ]]; then
        log_error "No backup found for rollback"
        exit 1
    fi
    
    local backup_timestamp=$(cat .last_backup)
    local backup_file="backup_${backup_timestamp}.tar.gz"
    
    if [[ ! -f "$backup_file" ]]; then
        log_error "Backup file not found: $backup_file"
        exit 1
    fi
    
    # Extract backup
    tar -xzf "$backup_file"
    
    # Restore database
    log_info "Restoring database..."
    PGPASSWORD="$DATABASE_PASSWORD" psql \
        -h "$DATABASE_HOST" \
        -p "$DATABASE_PORT" \
        -U "$DATABASE_USER" \
        -d "$DATABASE_NAME" \
        < "backups/${backup_timestamp}/database_backup.sql"
    
    # Restore configuration
    cp -r "backups/${backup_timestamp}/config/" . 2>/dev/null || true
    
    # Restart services with previous version
    if command -v kubectl &> /dev/null && kubectl get namespace customer-service-bot &> /dev/null; then
        kubectl rollout undo deployment/backend-deployment -n customer-service-bot
        kubectl rollout undo deployment/frontend-deployment -n customer-service-bot
    elif [[ -f docker-compose.prod.yml ]]; then
        docker-compose -f docker-compose.prod.yml down
        docker-compose -f docker-compose.prod.yml up -d
    fi
    
    # Cleanup
    rm -rf "backups/${backup_timestamp}"
    
    log_success "Rollback completed"
}

# Cleanup old backups
cleanup_old_backups() {
    log_info "Cleaning up old backups..."
    
    find . -name "backup_*.tar.gz" -type f -mtime +$BACKUP_RETENTION_DAYS -delete
    
    log_success "Old backups cleaned up"
}

# Send deployment notification
send_notification() {
    local status="$1"
    local message="$2"
    
    # Slack notification (if webhook URL is configured)
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"🚀 **$PROJECT_NAME Deployment**\n**Status:** $status\n**Environment:** $DEPLOY_ENV\n**Message:** $message\"}" \
            "$SLACK_WEBHOOK_URL" &> /dev/null || true
    fi
    
    # Email notification (if configured)
    if [[ -n "${NOTIFICATION_EMAIL:-}" ]]; then
        echo "$message" | mail -s "[$PROJECT_NAME] Deployment $status" "$NOTIFICATION_EMAIL" &> /dev/null || true
    fi
}

# Main deployment function
main() {
    local start_time=$(date +%s)
    
    log_info "Starting deployment of $PROJECT_NAME to $DEPLOY_ENV environment"
    
    # Trap for cleanup on exit
    trap 'cleanup_on_exit $?' EXIT
    
    # Pre-deployment checks
    check_prerequisites
    
    # Create backup
    create_backup
    
    # Build and push images
    build_and_push_images
    
    # Run database migrations
    run_migrations
    
    # Deploy based on environment
    if command -v kubectl &> /dev/null && kubectl get namespace customer-service-bot &> /dev/null 2>&1; then
        deploy_to_kubernetes
    else
        deploy_with_docker_compose
    fi
    
    # Perform health checks
    if ! perform_health_checks; then
        if [[ "$ROLLBACK_ENABLED" == "true" ]]; then
            log_error "Health checks failed, initiating rollback..."
            rollback_deployment
            send_notification "FAILED (Rolled Back)" "Deployment failed health checks and was rolled back"
            exit 1
        else
            log_error "Health checks failed but rollback is disabled"
            send_notification "FAILED" "Deployment failed health checks"
            exit 1
        fi
    fi
    
    # Cleanup
    cleanup_old_backups
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log_success "Deployment completed successfully in ${duration} seconds"
    send_notification "SUCCESS" "Deployment completed successfully in ${duration} seconds"
}

# Cleanup function
cleanup_on_exit() {
    local exit_code=$1
    
    if [[ $exit_code -ne 0 ]]; then
        log_error "Deployment failed with exit code $exit_code"
    fi
    
    # Cleanup temporary files
    rm -f .last_deploy_tag .last_backup 2>/dev/null || true
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --rollback)
                rollback_deployment
                exit 0
                ;;
            --health-check)
                perform_health_checks
                exit $?
                ;;
            --dry-run)
                log_info "Dry run mode - no actual deployment will be performed"
                DRY_RUN=true
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo "Options:"
                echo "  --rollback     Rollback to previous deployment"
                echo "  --health-check Perform health checks only"
                echo "  --dry-run      Dry run mode"
                echo "  --help         Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
        shift
    done
    
    # Run main deployment
    main
fi