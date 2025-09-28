#!/bin/bash
set -e

# Script to securely build Docker images without exposing secrets in build output

if [ $# -ne 1 ]; then
    echo "Usage: $0 <environment>"
    echo "Environments: development, staging, production"
    exit 1
fi

ENV=$1

# Validate environment
if [[ ! "$ENV" =~ ^(development|staging|production)$ ]]; then
    echo "Invalid environment. Must be one of: development, staging, production"
    exit 1
fi

echo "Building Docker image for $ENV environment"

# Check if env file exists
ENV_FILE=".env.$ENV"
if [ ! -f "$ENV_FILE" ]; then
    echo "Warning: $ENV_FILE not found. Creating from .env.example"
    if [ ! -f .env.example ]; then
        echo "Error: .env.example not found"
        exit 1
    fi
    cp .env.example "$ENV_FILE"
    echo "Please update $ENV_FILE with your configuration before running the container"
fi

echo "Loading environment variables from $ENV_FILE (secrets masked)"

# Securely load environment variables
set -a
source "$ENV_FILE"
set +a

# Print confirmation with masked values
echo "Environment: $ENV"
echo "Database host: ${POSTGRES_HOST:-${DB_HOST:-Not set}}"
echo "Database port: ${POSTGRES_PORT:-${DB_PORT:-Not set}}"
echo "Database name: ${POSTGRES_DB:-${DB_NAME:-Not set}}"
echo "Database user: ${POSTGRES_USER:-${DB_USER:-Not set}}"
echo "API keys: ******** (masked for security)"

# Build the Docker image with secrets but without showing them in console output
docker build --no-cache \
    --build-arg APP_ENV="$ENV" \
    --build-arg LLM_API_KEY="$LLM_API_KEY" \
    --build-arg LANGFUSE_PUBLIC_KEY="$LANGFUSE_PUBLIC_KEY" \
    --build-arg LANGFUSE_SECRET_KEY="$LANGFUSE_SECRET_KEY" \
    --build-arg JWT_SECRET_KEY="$JWT_SECRET_KEY" \
    -t fastapi-langgraph-template:"$ENV" .

echo "Docker image fastapi-langgraph-template:$ENV built successfully"
