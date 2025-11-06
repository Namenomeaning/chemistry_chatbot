#!/bin/bash
# MinIO bucket setup script

set -e

echo "Waiting for MinIO to be ready..."
sleep 10
echo "✓ MinIO should be ready"

# Configure mc (MinIO Client)
mc alias set local http://minio:9000 minioadmin minioadmin

# Create buckets
echo "Creating buckets..."
mc mb local/chemistry-images --ignore-existing
mc mb local/chemistry-audio --ignore-existing
echo "✓ Buckets created"

# Set public read policy
echo "Setting public read policies..."
mc anonymous set download local/chemistry-images
mc anonymous set download local/chemistry-audio
echo "✓ Policies applied"

echo "✓ MinIO setup completed successfully!"
