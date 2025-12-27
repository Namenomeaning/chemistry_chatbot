#!/bin/bash
# EC2 Setup Script for Amazon Linux 2023
# Run this script on a fresh or existing EC2 instance
# Safe to run multiple times (idempotent)

set -e

echo "=== EC2 Setup Script ==="
echo ""

# Update system packages
echo "=== Updating system packages ==="
sudo dnf update -y

# Install Docker if not already installed
if command -v docker &> /dev/null; then
    echo "=== Docker already installed ==="
    docker --version
else
    echo "=== Installing Docker ==="
    sudo dnf install -y docker
fi

# Start and enable Docker service
echo "=== Ensuring Docker service is running ==="
sudo systemctl start docker 2>/dev/null || true
sudo systemctl enable docker 2>/dev/null || true

# Add user to docker group if not already
if groups $USER | grep -q '\bdocker\b'; then
    echo "=== User already in docker group ==="
else
    echo "=== Adding user to docker group ==="
    sudo usermod -aG docker $USER
fi

# Install Docker Compose plugin if not already installed or outdated
echo "=== Checking Docker Compose ==="
sudo mkdir -p /usr/libexec/docker/cli-plugins

COMPOSE_VERSION=$(docker compose version 2>/dev/null | grep -oP 'v\d+\.\d+\.\d+' || echo "v0.0.0")
if [[ "$COMPOSE_VERSION" < "v2.20.0" ]]; then
    echo "=== Installing/Updating Docker Compose plugin ==="
    sudo curl -SL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-$(uname -m)" \
      -o /usr/libexec/docker/cli-plugins/docker-compose
    sudo chmod +x /usr/libexec/docker/cli-plugins/docker-compose
else
    echo "=== Docker Compose already up to date ==="
fi

# Install Docker Buildx plugin if not already installed or outdated
echo "=== Checking Docker Buildx ==="
BUILDX_VERSION=$(docker buildx version 2>/dev/null | grep -oP 'v\d+\.\d+\.\d+' || echo "v0.0.0")
if [[ "$BUILDX_VERSION" < "v0.17.0" ]]; then
    echo "=== Installing/Updating Docker Buildx plugin ==="
    sudo curl -SL "https://github.com/docker/buildx/releases/latest/download/buildx-$(uname -s | tr '[:upper:]' '[:lower:]')-amd64" \
      -o /usr/libexec/docker/cli-plugins/docker-buildx
    sudo chmod +x /usr/libexec/docker/cli-plugins/docker-buildx
else
    echo "=== Docker Buildx already up to date ==="
fi

# Restart Docker to apply changes
echo "=== Restarting Docker ==="
sudo systemctl restart docker

# Install Git if not already installed
if command -v git &> /dev/null; then
    echo "=== Git already installed ==="
else
    echo "=== Installing Git ==="
    sudo dnf install -y git
fi

# Verify installations
echo ""
echo "=== Verifying installations ==="
docker --version
docker compose version
docker buildx version
git --version

echo ""
echo "=== Setup Complete ==="
echo ""

# Check if user needs to re-login for docker group
if ! docker ps &> /dev/null; then
    echo "IMPORTANT: Log out and log back in for docker group changes to take effect"
    echo ""
    echo "Run: exit"
    echo "Then SSH back in"
else
    echo "Docker is ready to use!"
fi

echo ""
echo "Next steps:"
echo "1. Clone your repo: git clone https://github.com/Namenomeaning/chemi_chatbot.git"
echo "2. cd chemi_chatbot"
echo "3. Create .env file with your API keys"
echo "4. Run: ./deploy/init-ssl.sh chemi.vn your@email.com"
echo "5. Run: docker compose up -d --build"
echo ""
echo "Your site will be at: https://chemi.vn"
