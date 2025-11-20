#!/bin/bash
set -e  # Exit if any command fails

LOG_FILE="/home/ubuntu/start_docker.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "$(date) - Starting Docker deployment script..."

AWS_REGION="eu-north-1"
ECR_REPO="467105738571.dkr.ecr.eu-north-1.amazonaws.com/yt-chrome-plugin:latest"
CONTAINER_NAME="Analysis-app"
HOST_PORT=80
CONTAINER_PORT=5000

echo "$(date) - Logging in to ECR..."
aws ecr get-login-password --region eu-north-1 | docker login --username AWS --password-stdin 467105738571.dkr.ecr.eu-north-1.amazonaws.com

echo "$(date) - Pulling latest Docker image..."
docker pull "$ECR_REPO"

echo "$(date) - Stopping old container if exists..."
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    docker stop "$CONTAINER_NAME"
fi

echo "$(date) - Removing old container if exists..."
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    docker rm "$CONTAINER_NAME"
fi

echo "$(date) - Removing unused images..."
docker system prune -af

echo "$(date) - Starting new container..."
docker run -d \
  --name "$CONTAINER_NAME" \
  -p ${HOST_PORT}:${CONTAINER_PORT} \
  --restart unless-stopped \
  "$ECR_REPO"

echo "$(date) - Container started successfully."
docker ps -a | grep "$CONTAINER_NAME"
echo "$(date) - Deployment completed."
