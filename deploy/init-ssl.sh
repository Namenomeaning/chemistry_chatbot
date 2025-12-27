#!/bin/bash
# Initialize Let's Encrypt SSL Certificate
# Usage: ./deploy/init-ssl.sh your-domain.com your@email.com

set -e

DOMAIN=${1:-chemi.vn}
EMAIL=${2:-admin@chemi.vn}

if [ -z "$1" ]; then
    echo "Using default domain: chemi.vn"
fi

echo "=== Setting up SSL for $DOMAIN ==="

# Create directories
mkdir -p certbot/conf certbot/www

# Update nginx.conf with domain
sed -i "s/chemi.vn/$DOMAIN/g" nginx.conf

# Create temporary nginx config for ACME challenge
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
    -v $(pwd)/certbot/www:/var/www/certbot:ro \
    nginx:alpine

sleep 3

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
    --non-interactive

echo "=== Stopping temporary nginx ==="
docker stop nginx-temp
docker rm nginx-temp
rm nginx-temp.conf

echo ""
echo "=== SSL Certificate obtained successfully! ==="
echo ""
echo "Now run: docker compose up -d --build"
echo ""
echo "Your site will be available at: https://$DOMAIN"
