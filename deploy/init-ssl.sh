#!/bin/bash
# Initialize Let's Encrypt SSL Certificate
# Usage: ./deploy/init-ssl.sh [domain] [email]
# Safe to run multiple times (idempotent)

set -e

DOMAIN=${1:-chemi.vn}
EMAIL=${2:-admin@chemi.vn}

echo "=== SSL Setup Script ==="
echo "Domain: $DOMAIN"
echo "Email: $EMAIL"
echo ""

# Cleanup function
cleanup() {
    echo "=== Cleaning up ==="
    docker stop nginx-temp 2>/dev/null || true
    docker rm nginx-temp 2>/dev/null || true
    rm -f nginx-temp.conf 2>/dev/null || true
}

# Set trap to cleanup on exit or error
trap cleanup EXIT

# Check if SSL certificate already exists
if [ -d "certbot/conf/live/$DOMAIN" ]; then
    echo "=== SSL certificate already exists for $DOMAIN ==="
    echo "To renew, delete certbot/conf/live/$DOMAIN and run again"
    echo ""
    read -p "Do you want to continue anyway? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# Create directories
echo "=== Creating directories ==="
mkdir -p certbot/conf certbot/www

# Stop any running containers that might use port 80
echo "=== Stopping containers using port 80 ==="
docker stop chemi-nginx 2>/dev/null || true
docker stop nginx-temp 2>/dev/null || true
docker rm nginx-temp 2>/dev/null || true

# Wait a moment for port to be released
sleep 2

# Create temporary nginx config for ACME challenge
echo "=== Creating temporary nginx config ==="
cat > nginx-temp.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    server {
        listen 80;
        server_name _;

        location /.well-known/acme-challenge/ {
            root /var/www/certbot;
        }

        location / {
            return 200 'OK';
            add_header Content-Type text/plain;
        }
    }
}
EOF

echo "=== Starting temporary nginx for ACME challenge ==="
docker run -d --name nginx-temp \
    -p 80:80 \
    -v $(pwd)/nginx-temp.conf:/etc/nginx/nginx.conf:ro \
    -v $(pwd)/certbot/www:/var/www/certbot \
    nginx:alpine

# Wait for nginx to start
sleep 3

# Verify nginx is running
if ! docker ps | grep -q nginx-temp; then
    echo "ERROR: Failed to start temporary nginx"
    exit 1
fi

echo "=== Requesting SSL certificate from Let's Encrypt ==="
docker run --rm \
    -v $(pwd)/certbot/conf:/etc/letsencrypt \
    -v $(pwd)/certbot/www:/var/www/certbot \
    certbot/certbot certonly --webroot \
    -w /var/www/certbot \
    -d $DOMAIN \
    --email $EMAIL \
    --agree-tos \
    --no-eff-email \
    --non-interactive \
    --force-renewal

echo ""
echo "=== SSL Certificate obtained successfully! ==="
echo ""

# Update nginx.conf with SSL configuration
echo "=== Updating nginx.conf for HTTPS ==="
cat > nginx.conf << EOF
events {
    worker_connections 1024;
}

http {
    upstream app {
        server app:8000;
    }

    # HTTP - redirect to HTTPS + ACME challenge
    server {
        listen 80;
        server_name $DOMAIN;

        location /.well-known/acme-challenge/ {
            root /var/www/certbot;
        }

        location / {
            return 301 https://\$host\$request_uri;
        }
    }

    # HTTPS
    server {
        listen 443 ssl;
        server_name $DOMAIN;

        ssl_certificate /etc/letsencrypt/live/$DOMAIN/fullchain.pem;
        ssl_certificate_key /etc/letsencrypt/live/$DOMAIN/privkey.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;

        client_max_body_size 10M;

        location / {
            proxy_pass http://app;
            proxy_http_version 1.1;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
            proxy_read_timeout 120s;
            proxy_connect_timeout 10s;
        }
    }
}
EOF

# Update docker-compose.yml with SSL configuration
echo "=== Updating docker-compose.yml for SSL ==="
cat > docker-compose.yml << 'EOF'
services:
  app:
    build: .
    container_name: chemi-app
    env_file: .env
    restart: unless-stopped
    expose:
      - "8000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3

  nginx:
    image: nginx:alpine
    container_name: chemi-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./certbot/conf:/etc/letsencrypt:ro
      - ./certbot/www:/var/www/certbot:ro
    depends_on:
      app:
        condition: service_healthy
    restart: unless-stopped

  certbot:
    image: certbot/certbot
    container_name: chemi-certbot
    volumes:
      - ./certbot/conf:/etc/letsencrypt
      - ./certbot/www:/var/www/certbot
    entrypoint: "/bin/sh -c 'trap exit TERM; while :; do certbot renew; sleep 12h & wait $${!}; done;'"
    restart: unless-stopped
EOF

echo ""
echo "=== Configuration updated ==="
echo ""
echo "Now run: docker compose up -d --build"
echo ""
echo "Your site will be available at: https://$DOMAIN"
