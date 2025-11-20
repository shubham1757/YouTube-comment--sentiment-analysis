#!/bin/bash

set -e  # Exit script if any command fails
export DEBIAN_FRONTEND=noninteractive

# Update package lists and install required tools
sudo apt-get update -y
sudo apt-get install -y unzip curl

# Install Docker if not already installed
if ! command -v docker &> /dev/null; then
    echo "Docker not found. Installing Docker..."
    sudo apt-get install -y docker.io
    sudo systemctl enable docker
    sudo systemctl start docker
else
    echo "Docker already installed. Skipping installation."
fi

# Install AWS CLI if not already installed
if ! command -v aws &> /dev/null; then
    echo "AWS CLI not found. Installing AWS CLI..."
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/home/ubuntu/awscliv2.zip"
    unzip -o /home/ubuntu/awscliv2.zip -d /home/ubuntu/
    sudo /home/ubuntu/aws/install
else
    echo "AWS CLI already installed. Skipping installation."
fi

# Add ubuntu user to docker group
sudo usermod -aG docker ubuntu

# Clean up
rm -rf /home/ubuntu/awscliv2.zip /home/ubuntu/aws

echo "Installation complete."
