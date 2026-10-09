#!/bin/bash
# Deploy or update the demo on the EC2 server.
# Run on the server:  sudo bash /opt/rag-demo/deploy.sh
# Safe to re-run: it always deploys the most recently pushed image in ECR,
# and reads the secrets from Parameter Store using the server's IAM role.
set -euo pipefail
mkdir -p /opt/rag-demo
cd /opt/rag-demo

# 1. Facts about this server, read from AWS
TOKEN=$(curl -sX PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 300")
AWS_DEFAULT_REGION=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/placement/region)
export AWS_DEFAULT_REGION
IP=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/public-ipv4)
REGISTRY=$(aws sts get-caller-identity --query Account --output text).dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com
TAG=$(aws ecr describe-images --repository-name rag-demo \
  --query 'sort_by(imageDetails[?imageTags], &imagePushedAt)[-1].imageTags[0]' --output text)

# 2. Settings for Docker Compose
cat > .env <<EOF
IMAGE=$REGISTRY/rag-demo:$TAG
SITE_ADDRESS=$(echo "$IP" | tr . -).sslip.io
EOF

# 3. Secrets, readable only by root
umask 077
{
  echo "API_KEY=$(aws ssm get-parameter --name /rag-demo/API_KEY --with-decryption --query Parameter.Value --output text)"
  echo "HOSTED_API_KEY=$(aws ssm get-parameter --name /rag-demo/HOSTED_API_KEY --with-decryption --query Parameter.Value --output text)"
} > .env.secrets
umask 022

# 4. Configuration files
cat > compose.yaml <<'EOF'
services:
  app:
    image: ${IMAGE}
    env_file: .env.secrets
    environment:
      FORWARDED_ALLOW_IPS: "*"
    mem_limit: 1g
    restart: unless-stopped
  caddy:
    image: caddy:2
    environment:
      SITE_ADDRESS: ${SITE_ADDRESS}
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile:ro
      - caddy_data:/data
      - caddy_config:/config
    restart: unless-stopped
    depends_on:
      - app
volumes:
  caddy_data:
  caddy_config:
EOF

cat > Caddyfile <<'EOF'
{$SITE_ADDRESS} {
    redir / /ui/
    reverse_proxy app:8000
}
EOF

# 5. Pull and (re)start
aws ecr get-login-password | docker login --username AWS --password-stdin "$REGISTRY"
docker compose up -d
docker compose ps
echo "Deployed $REGISTRY/rag-demo:$TAG at https://$(echo "$IP" | tr . -).sslip.io"
