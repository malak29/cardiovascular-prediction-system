# Cardiovascular Disease Prediction System - Deployment Guide

## Overview

This comprehensive deployment guide covers all aspects of deploying the Cardiovascular Disease Prediction System in various environments, from local development to production-ready Kubernetes clusters.

**System Version**: v1.0.0  
**Last Updated**: January 15, 2024  
**Deployment Environments**: Local, Docker, Kubernetes, Cloud (AWS/GCP/Azure)

## Table of Contents

- [Prerequisites](#prerequisites)
- [Architecture Overview](#architecture-overview)
- [Environment Setup](#environment-setup)
- [Local Development Deployment](#local-development-deployment)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Database Setup](#database-setup)
- [Security Configuration](#security-configuration)
- [Monitoring and Logging](#monitoring-and-logging)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)
- [Maintenance](#maintenance)

## Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 4 cores
- RAM: 8GB
- Storage: 50GB SSD
- Network: 100Mbps

**Recommended Production:**
- CPU: 8+ cores
- RAM: 16GB+
- Storage: 200GB+ SSD
- Network: 1Gbps+

### Software Prerequisites

**Required Software:**
- Docker 24.0+
- Docker Compose 2.20+
- Kubernetes 1.28+ (for K8s deployment)
- kubectl 1.28+
- Python 3.11+
- Node.js 18+
- PostgreSQL 15+
- Redis 7+

**Optional Tools:**
- Helm 3.12+
- Terraform 1.5+
- AWS CLI 2.13+
- Git 2.40+

### Access Requirements

**Repository Access:**
```bash
git clone https://github.com/yourorg/cvd-prediction-system.git
cd cvd-prediction-system
```

**Registry Access:**
- Docker Hub or private registry credentials
- Container image pull permissions

**Cloud Access (if applicable):**
- AWS/GCP/Azure account with appropriate permissions
- Service account keys or IAM roles configured

## Architecture Overview

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │────│   Frontend      │────│   Backend API   │
│   (Nginx/ALB)   │    │   (React)       │    │   (FastAPI)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                              ┌─────────────────────────┼─────────────────────────┐
                              │                         │                         │
                    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
                    │   PostgreSQL    │    │     Redis       │    │   ML Models     │
                    │   (Database)    │    │    (Cache)      │    │   (Storage)     │
                    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Component Dependencies

1. **Frontend (React)** → Backend API
2. **Backend API (FastAPI)** → PostgreSQL, Redis, ML Models
3. **Database (PostgreSQL)** → Persistent storage
4. **Cache (Redis)** → Session storage, model caching
5. **ML Models** → Trained model artifacts

## Environment Setup

### Environment Variables

Create environment-specific configuration files:

#### `.env.local`
```bash
# Application
APP_NAME=CVD Prediction System
APP_ENV=development
DEBUG=true
LOG_LEVEL=DEBUG

# Database
DATABASE_URL=postgresql://cvduser:password@localhost:5432/cvd_db
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_POOL_SIZE=10

# Security
SECRET_KEY=your-secret-key-for-development
API_KEY_SALT=your-api-key-salt

# ML Models
MODEL_PATH=/app/ml_models
MODEL_CACHE_TTL=3600

# Frontend
REACT_APP_API_BASE_URL=http://localhost:8000/api/v1
REACT_APP_ENVIRONMENT=development
```

#### `.env.production`
```bash
# Application
APP_NAME=CVD Prediction System
APP_ENV=production
DEBUG=false
LOG_LEVEL=INFO

# Database (use strong credentials)
DATABASE_URL=postgresql://cvduser:STRONG_PASSWORD@postgres:5432/cvd_db
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=40

# Redis
REDIS_URL=redis://redis:6379/0
REDIS_POOL_SIZE=20

# Security (use strong keys)
SECRET_KEY=VERY_STRONG_SECRET_KEY_FOR_PRODUCTION
API_KEY_SALT=STRONG_API_KEY_SALT

# Performance
WORKERS=4
MAX_REQUESTS=1000
MAX_REQUESTS_JITTER=50

# Frontend
REACT_APP_API_BASE_URL=https://cvd-api.example.com/api/v1
REACT_APP_ENVIRONMENT=production
```

## Local Development Deployment

### Quick Start

1. **Clone and Setup**
```bash
git clone https://github.com/yourorg/cvd-prediction-system.git
cd cvd-prediction-system
cp .env.example .env.local
```

2. **Start Dependencies**
```bash
# Start PostgreSQL and Redis
docker-compose -f docker-compose.dev.yml up -d postgres redis

# Wait for services to be ready
sleep 10
```

3. **Backend Setup**
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run database migrations
alembic upgrade head

# Start backend server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

4. **Frontend Setup**
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

5. **Access Application**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Development Tools

**Database Management**
```bash
# Connect to database
docker exec -it cvd_postgres psql -U cvduser -d cvd_db

# Run migrations
cd backend && alembic upgrade head

# Create new migration
cd backend && alembic revision --autogenerate -m "Description"
```

**Testing**
```bash
# Backend tests
cd backend && python -m pytest

# Frontend tests  
cd frontend && npm test

# End-to-end tests
cd frontend && npm run test:e2e
```

## Docker Deployment

### Single-Host Docker Deployment

1. **Prepare Environment**
```bash
# Create production environment file
cp .env.example .env.production
# Edit .env.production with production values

# Create data directories
mkdir -p data/postgres data/redis data/models data/logs
```

2. **Build and Deploy**
```bash
# Build all services
docker-compose build

# Start all services
docker-compose -f docker-compose.prod.yml up -d

# Check service health
docker-compose ps
docker-compose logs -f
```

3. **Initialize Database**
```bash
# Run database migrations
docker-compose exec backend alembic upgrade head

# Load initial data (if available)
docker-compose exec backend python scripts/load_sample_data.py
```

### Docker Compose Configuration

**docker-compose.prod.yml**
```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: ${DATABASE_USER}
      POSTGRES_PASSWORD: ${DATABASE_PASSWORD}
      POSTGRES_DB: ${DATABASE_NAME}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-scripts:/docker-entrypoint-initdb.d
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DATABASE_USER}"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  backend:
    build:
      context: .
      dockerfile: infrastructure/docker/Dockerfile.backend
      target: production
    environment:
      - DATABASE_URL=postgresql://${DATABASE_USER}:${DATABASE_PASSWORD}@postgres:5432/${DATABASE_NAME}
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - model_data:/app/ml_models
      - ./data/logs:/app/logs
    ports:
      - "8000:8000"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  frontend:
    build:
      context: .
      dockerfile: infrastructure/docker/Dockerfile.frontend
      target: production
      args:
        - REACT_APP_API_BASE_URL=${REACT_APP_API_BASE_URL}
    ports:
      - "80:80"
    depends_on:
      backend:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  model_data:
```

## Kubernetes Deployment

### Prerequisites

1. **Kubernetes Cluster**
```bash
# Verify cluster access
kubectl cluster-info
kubectl get nodes

# Create namespace
kubectl create namespace cvd-prediction
kubectl config set-context --current --namespace=cvd-prediction
```

2. **Required Tools**
```bash
# Install Helm (if not installed)
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Add required Helm repositories
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update
```

### Step-by-Step Deployment

#### 1. Install Dependencies

**Install PostgreSQL**
```bash
helm install postgres bitnami/postgresql \
  --namespace cvd-prediction \
  --set auth.username=cvduser \
  --set auth.password=STRONG_PASSWORD \
  --set auth.database=cvd_db \
  --set primary.persistence.size=50Gi \
  --set primary.resources.requests.memory=1Gi \
  --set primary.resources.requests.cpu=500m
```

**Install Redis**
```bash
helm install redis bitnami/redis \
  --namespace cvd-prediction \
  --set auth.password=REDIS_PASSWORD \
  --set replica.replicaCount=1 \
  --set master.persistence.size=8Gi
```

**Install Nginx Ingress Controller**
```bash
helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --create-namespace \
  --set controller.replicaCount=2 \
  --set controller.nodeSelector."kubernetes\.io/os"=linux \
  --set defaultBackend.nodeSelector."kubernetes\.io/os"=linux
```

#### 2. Deploy Application

**Apply Configuration**
```bash
# Create secrets
kubectl create secret generic cvd-database-secret \
  --from-literal=username=cvduser \
  --from-literal=password=STRONG_PASSWORD

kubectl create secret generic cvd-app-secret \
  --from-literal=secret-key=YOUR_SECRET_KEY \
  --from-literal=api-key=YOUR_API_KEY

# Apply all manifests
kubectl apply -f infrastructure/k8s/
```

**Verify Deployment**
```bash
# Check pod status
kubectl get pods -w

# Check services
kubectl get services

# Check ingress
kubectl get ingress

# View logs
kubectl logs -l app=cvd-backend
kubectl logs -l app=cvd-frontend
```

#### 3. Database Initialization

```bash
# Run database migrations
kubectl exec -it deployment/cvd-backend -- alembic upgrade head

# Load initial data (if needed)
kubectl exec -it deployment/cvd-backend -- python scripts/load_initial_data.py
```

### Helm Chart Deployment (Alternative)

Create a Helm chart for easier deployment:

**values.yaml**
```yaml
image:
  backend:
    repository: ghcr.io/yourorg/cvd-backend
    tag: "v1.0.0"
  frontend:
    repository: ghcr.io/yourorg/cvd-frontend
    tag: "v1.0.0"

replicaCount:
  backend: 3
  frontend: 2

resources:
  backend:
    requests:
      memory: "512Mi"
      cpu: "250m"
    limits:
      memory: "1Gi"
      cpu: "500m"
  frontend:
    requests:
      memory: "128Mi"
      cpu: "100m"
    limits:
      memory: "256Mi"
      cpu: "200m"

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: cvd-prediction.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: cvd-tls-secret
      hosts:
        - cvd-prediction.example.com

database:
  host: postgres-postgresql
  port: 5432
  name: cvd_db
  username: cvduser
  
redis:
  host: redis-master
  port: 6379
```

**Deploy with Helm**
```bash
helm install cvd-prediction ./helm/cvd-prediction \
  --namespace cvd-prediction \
  --values values.yaml
```

## Cloud Deployment

### AWS Deployment

#### 1. EKS Cluster Setup

```bash
# Install eksctl
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin

# Create EKS cluster
eksctl create cluster --name cvd-prediction \
  --version 1.28 \
  --region us-west-2 \
  --nodegroup-name standard-workers \
  --node-type m5.large \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 10 \
  --managed

# Update kubeconfig
aws eks update-kubeconfig --region us-west-2 --name cvd-prediction
```

#### 2. AWS-Specific Configuration

**Install AWS Load Balancer Controller**
```bash
# Create IAM service account
eksctl create iamserviceaccount \
  --cluster=cvd-prediction \
  --namespace=kube-system \
  --name=aws-load-balancer-controller \
  --attach-policy-arn=arn:aws:iam::ACCOUNT:policy/AWSLoadBalancerControllerIAMPolicy \
  --override-existing-serviceaccounts \
  --approve

# Install controller
helm repo add eks https://aws.github.io/eks-charts
helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  -n kube-system \
  --set clusterName=cvd-prediction \
  --set serviceAccount.create=false \
  --set serviceAccount.name=aws-load-balancer-controller
```

**Configure RDS and ElastiCache**
```bash
# Create RDS instance
aws rds create-db-instance \
  --db-instance-identifier cvd-postgres \
  --db-instance-class db.r5.large \
  --engine postgres \
  --engine-version 15.4 \
  --master-username cvduser \
  --master-user-password STRONG_PASSWORD \
  --allocated-storage 100 \
  --storage-type gp2 \
  --vpc-security-group-ids sg-xxxxxxxx \
  --db-subnet-group-name cvd-db-subnet-group

# Create ElastiCache cluster
aws elasticache create-cache-cluster \
  --cache-cluster-id cvd-redis \
  --cache-node-type cache.r5.large \
  --engine redis \
  --num-cache-nodes 1 \
  --security-group-ids sg-xxxxxxxx \
  --subnet-group-name cvd-cache-subnet-group
```

#### 3. S3 for Model Storage

```bash
# Create S3 bucket for models
aws s3 mb s3://cvd-prediction-models

# Configure bucket policy for model access
aws s3api put-bucket-policy --bucket cvd-prediction-models --policy file://s3-policy.json
```

### GCP Deployment

#### 1. GKE Cluster Setup

```bash
# Create GKE cluster
gcloud container clusters create cvd-prediction \
  --zone us-central1-a \
  --machine-type n1-standard-2 \
  --num-nodes 3 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 10 \
  --enable-autorepair \
  --enable-autoupgrade

# Get cluster credentials
gcloud container clusters get-credentials cvd-prediction --zone us-central1-a
```

#### 2. GCP-Specific Services

**Cloud SQL for PostgreSQL**
```bash
# Create Cloud SQL instance
gcloud sql instances create cvd-postgres \
  --database-version=POSTGRES_15 \
  --tier=db-custom-2-4096 \
  --region=us-central1 \
  --storage-type=SSD \
  --storage-size=100GB

# Create database and user
gcloud sql databases create cvd_db --instance=cvd-postgres
gcloud sql users create cvduser --instance=cvd-postgres --password=STRONG_PASSWORD
```

**Cloud Storage for Models**
```bash
# Create bucket
gsutil mb gs://cvd-prediction-models

# Set permissions
gsutil iam ch serviceAccount:cvd-sa@PROJECT.iam.gserviceaccount.com:objectViewer gs://cvd-prediction-models
```

### Azure Deployment

#### 1. AKS Cluster Setup

```bash
# Create resource group
az group create --name cvd-prediction --location eastus

# Create AKS cluster
az aks create \
  --resource-group cvd-prediction \
  --name cvd-prediction-cluster \
  --node-count 3 \
  --enable-autoscaler \
  --min-count 1 \
  --max-count 10 \
  --generate-ssh-keys

# Get cluster credentials
az aks get-credentials --resource-group cvd-prediction --name cvd-prediction-cluster
```

## Database Setup

### PostgreSQL Configuration

#### Production Configuration

**postgresql.conf**
```ini
# Memory settings
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB

# Checkpoint settings
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100

# Connection settings
max_connections = 100
max_prepared_transactions = 0

# Logging
log_destination = 'stderr'
log_min_duration_statement = 1000
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '
log_statement = 'ddl'

# Replication (if needed)
wal_level = replica
max_wal_senders = 3
wal_keep_size = 16MB
```

#### Database Initialization

```sql
-- Create database and user
CREATE DATABASE cvd_db;
CREATE USER cvduser WITH ENCRYPTED PASSWORD 'STRONG_PASSWORD';
GRANT ALL PRIVILEGES ON DATABASE cvd_db TO cvduser;

-- Grant schema privileges
\c cvd_db;
GRANT ALL ON SCHEMA public TO cvduser;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO cvduser;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO cvduser;
```

#### Backup and Recovery

**Automated Backups**
```bash
#!/bin/bash
# backup-postgres.sh

BACKUP_DIR="/backups/postgres"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="cvd_db_backup_${TIMESTAMP}.sql"

# Create backup
pg_dump -h postgres -U cvduser -d cvd_db > "${BACKUP_DIR}/${BACKUP_FILE}"

# Compress backup
gzip "${BACKUP_DIR}/${BACKUP_FILE}"

# Upload to cloud storage (optional)
aws s3 cp "${BACKUP_DIR}/${BACKUP_FILE}.gz" s3://cvd-backups/postgres/

# Cleanup old backups (keep last 7 days)
find "${BACKUP_DIR}" -name "*.gz" -mtime +7 -delete
```

**Recovery**
```bash
# Restore from backup
psql -h postgres -U cvduser -d cvd_db < backup_file.sql
```

### Redis Configuration

#### Production Redis Configuration

**redis.conf**
```ini
# Memory settings
maxmemory 512mb
maxmemory-policy allkeys-lru

# Persistence
appendonly yes
appendfsync everysec
save 900 1
save 300 10
save 60 10000

# Network
timeout 0
tcp-keepalive 300
tcp-backlog 511

# Security
requirepass STRONG_REDIS_PASSWORD

# Performance
hash-max-ziplist-entries 512
hash-max-ziplist-value 64
list-max-ziplist-size -2
set-max-intset-entries 512
zset-max-ziplist-entries 128
zset-max-ziplist-value 64
```

## Security Configuration

### SSL/TLS Configuration

#### Certificate Management with Let's Encrypt

**cert-manager Installation**
```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Create ClusterIssuer
cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

### Network Security

#### Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: cvd-network-policy
  namespace: cvd-prediction
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  # Allow ingress from nginx
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
  # Allow backend to database
  - from:
    - podSelector:
        matchLabels:
          app: cvd-backend
    ports:
    - protocol: TCP
      port: 5432
  egress:
  # Allow DNS
  - to: []
    ports:
    - protocol: UDP
      port: 53
  # Allow HTTPS outbound
  - to: []
    ports:
    - protocol: TCP
      port: 443
```

### Authentication and Authorization

#### JWT Configuration

```python
# backend/app/core/security.py
import jwt
from datetime import datetime, timedelta
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class SecurityManager:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30

    def create_access_token(self, data: dict):
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def verify_token(self, token: str):
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.PyJWTError:
            return None
```

## Monitoring and Logging

### Prometheus Monitoring

Refer to `/monitoring/prometheus.yml` for complete configuration. Key metrics to monitor:

- **Application Metrics**
  - Request rate and latency
  - Error rates
  - Prediction accuracy
  - Model inference time

- **Infrastructure Metrics**  
  - CPU and memory usage
  - Database connections
  - Cache hit rates
  - Disk usage

### Logging Configuration

Centralized logging with structured JSON logs:

```yaml
# logging-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: logging-config
data:
  filebeat.yml: |
    filebeat.inputs:
    - type: container
      paths:
        - /var/log/containers/*cvd*.log
      processors:
      - add_kubernetes_metadata:
          host: ${NODE_NAME}
          matchers:
          - logs_path:
              logs_path: "/var/log/containers/"
    
    output.elasticsearch:
      hosts: ["elasticsearch:9200"]
      index: "cvd-logs-%{+yyyy.MM.dd}"
```

### Health Monitoring

**Health Check Script**
```python
# scripts/health_check.py
import requests
import sys
import time

def check_health():
    endpoints = [
        "http://localhost:8000/health",
        "http://localhost:8000/api/v1/health/detailed"
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(endpoint, timeout=10)
            if response.status_code != 200:
                print(f"Health check failed for {endpoint}")
                return False
        except Exception as e:
            print(f"Health check error for {endpoint}: {e}")
            return False
    
    return True

if __name__ == "__main__":
    if not check_health():
        sys.exit(1)
    print("All health checks passed")
```

## Performance Tuning

### Backend Optimization

**Gunicorn Configuration**
```python
# gunicorn.conf.py
bind = "0.0.0.0:8000"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
preload_app = True
keepalive = 2
```

**Database Connection Pooling**
```python
# backend/app/core/database.py
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True,
    pool_recycle=3600
)
```

### Frontend Optimization

**Nginx Configuration**
```nginx
server {
    listen 80;
    server_name localhost;
    root /usr/share/nginx/html;
    index index.html index.htm;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/javascript
        application/xml+rss
        application/json;

    # Cache static assets
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        try_files $uri =404;
    }

    # Handle React Router
    location / {
        try_files $uri $uri/ /index.html;
    }

    # API proxy with optimizations
    location /api/ {
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # Buffering
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }
}
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Database Connection Issues

**Problem**: Cannot connect to PostgreSQL
**Solution**:
```bash
# Check database status
kubectl get pods -l app=postgres
kubectl logs -l app=postgres

# Test connection
kubectl exec -it deployment/cvd-backend -- psql $DATABASE_URL -c "SELECT 1"

# Check network connectivity
kubectl exec -it deployment/cvd-backend -- nc -zv postgres 5432
```

#### 2. Model Loading Failures

**Problem**: ML models not loading
**Solution**:
```bash
# Check model files
kubectl exec -it deployment/cvd-backend -- ls -la /app/ml_models/

# Check model permissions
kubectl exec -it deployment/cvd-backend -- python -c "
import joblib
try:
    model = joblib.load('/app/ml_models/model.pkl')
    print('Model loaded successfully')
except Exception as e:
    print(f'Model loading failed: {e}')
"
```

#### 3. High Memory Usage

**Problem**: Backend consuming too much memory
**Solution**:
```bash
# Check memory usage
kubectl top pods

# Analyze memory profile
kubectl exec -it deployment/cvd-backend -- python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB')
"

# Adjust resource limits
kubectl patch deployment cvd-backend -p '{"spec":{"template":{"spec":{"containers":[{"name":"cvd-backend","resources":{"limits":{"memory":"2Gi"}}}]}}}}'
```

#### 4. Frontend Build Failures

**Problem**: Frontend build fails during deployment
**Solution**:
```bash
# Check build logs
docker build -t cvd-frontend -f infrastructure/docker/Dockerfile.frontend .

# Check Node.js version
docker run --rm node:18-alpine node --version

# Clear npm cache
docker run --rm -v $(pwd)/frontend:/app -w /app node:18-alpine npm cache clean --force
```

### Log Analysis

**Common Log Patterns**
```bash
# Backend error logs
kubectl logs -l app=cvd-backend | grep ERROR

# Database connection logs
kubectl logs -l app=cvd-backend | grep -i "database\|postgres\|connection"

# API request logs
kubectl logs -l app=cvd-backend | grep -E "POST|GET|PUT|DELETE" | tail -20

# Performance logs
kubectl logs -l app=cvd-backend | grep -i "slow\|timeout\|performance"
```

## Maintenance

### Regular Maintenance Tasks

#### Daily Tasks
```bash
#!/bin/bash
# daily-maintenance.sh

# Check system health
kubectl get pods -n cvd-prediction
kubectl top nodes
kubectl top pods -n cvd-prediction

# Check recent logs for errors
kubectl logs -l app=cvd-backend --since=24h | grep -i error | wc -l

# Backup database
pg_dump -h $DB_HOST -U $DB_USER $DB_NAME | gzip > /backups/cvd_db_$(date +%Y%m%d).sql.gz
```

#### Weekly Tasks
```bash
#!/bin/bash
# weekly-maintenance.sh

# Update container images (staging first)
kubectl set image deployment/cvd-backend cvd-backend=latest -n cvd-staging
kubectl rollout status deployment/cvd-backend -n cvd-staging

# Clean up old backups
find /backups -name "*.sql.gz" -mtime +30 -delete

# Review resource usage and adjust limits if needed
kubectl describe nodes | grep -A 5 "Allocated resources"
```

#### Monthly Tasks
```bash
#!/bin/bash
# monthly-maintenance.sh

# Security updates
kubectl get pods -o jsonpath='{.items[*].spec.containers[*].image}' | tr ' ' '\n' | sort -u
# Review and update base images in Dockerfiles

# Performance review
# Analyze monitoring data and optimize resource allocation

# Certificate renewal check
kubectl get certificates -n cvd-prediction
```

### Backup and Disaster Recovery

**Backup Strategy**
```bash
#!/bin/bash
# backup-strategy.sh

# Database backup
pg_dump -h $DB_HOST -U $DB_USER $DB_NAME > backup.sql

# Model artifacts backup
tar -czf models-backup.tar.gz /app/ml_models/

# Configuration backup
kubectl get configmaps,secrets -o yaml > config-backup.yaml

# Upload to cloud storage
aws s3 sync /backups/ s3://cvd-backups/$(date +%Y-%m-%d)/
```

**Disaster Recovery Plan**
1. **Database Recovery**: Restore from latest backup
2. **Application Recovery**: Redeploy from container registry
3. **Data Recovery**: Restore model artifacts from backup
4. **Configuration Recovery**: Apply saved Kubernetes manifests

### Upgrade Process

**Rolling Update Process**
```bash
# 1. Deploy to staging
kubectl set image deployment/cvd-backend cvd-backend=v1.1.0 -n cvd-staging

# 2. Run smoke tests
kubectl exec -it deployment/cvd-backend -n cvd-staging -- python scripts/smoke_tests.py

# 3. Deploy to production with zero downtime
kubectl set image deployment/cvd-backend cvd-backend=v1.1.0 -n cvd-production

# 4. Monitor rollout
kubectl rollout status deployment/cvd-backend -n cvd-production

# 5. Verify health
curl -f https://cvd-prediction.example.com/health
```

---

## Support and Resources

- **Documentation**: https://docs.cvd-prediction.com
- **Support**: support@cvd-prediction.com  
- **Status Page**: https://status.cvd-prediction.com
- **GitHub Issues**: https://github.com/yourorg/cvd-prediction-system/issues

---

*This deployment guide is regularly updated. For the latest version, visit our documentation portal.*