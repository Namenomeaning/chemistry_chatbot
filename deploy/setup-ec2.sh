#!/bin/bash
# EC2 Setup Script for Amazon Linux 2023
# Run this script on a fresh EC2 instance

set -e

echo "=== Updating system packages ==="
sudo dnf update -y

echo "=== Installing Docker ==="
sudo dnf install -y docker

echo "=== Starting Docker service ==="
sudo systemctl start docker
sudo systemctl enable docker

echo "=== Adding user to docker group ==="
sudo usermod -aG docker $USER

echo "=== Installing Docker Compose plugin ==="
sudo mkdir -p /usr/libexec/docker/cli-plugins
sudo curl -SL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-$(uname -m)" \
  -o /usr/libexec/docker/cli-plugins/docker-compose
sudo chmod +x /usr/libexec/docker/cli-plugins/docker-compose

echo "=== Restarting Docker ==="
sudo systemctl restart docker

echo "=== Installing Git ==="
sudo dnf install -y git

echo "=== Verifying installations ==="
docker --version
docker compose version
git --version

echo ""
echo "=== Setup Complete ==="
echo "IMPORTANT: Log out and log back in for docker group changes to take effect"
echo ""
echo "Next steps:"
echo "1. Log out: exit"
echo "2. SSH back in"
echo "3. Clone your repo: git clone <your-repo-url>"
echo "4. cd chemistry_chatbot"
echo "5. Create .env file: nano .env"
echo "   Add: GROQ_API_KEY=your-key"
echo "6. Run: ./deploy/init-ssl.sh chemi.vn your@email.com"
echo "7. Run: docker compose up -d --build"
echo ""
echo "Your site will be at: https://chemi.vn"
