# Docker Workflow

## Prerequisites (WSL2)

1. Install Docker Engine packages inside Ubuntu:
   ```bash
   sudo apt update
   sudo apt install -y docker.io docker-compose-plugin
   ```
2. Add your user to the `docker` group so you can run Docker without `sudo`:
   ```bash
   sudo usermod -aG docker $USER
   newgrp docker
   ```

## Local Development

The default `docker-compose.yaml` builds local `:dev` images tagged under `chrislawrencedev/*`.

```bash
# Build/update dev images
docker compose -f docker-compose.yaml build

# Start the full stack with live code mounts
docker compose -f docker-compose.yaml up -d

# Tail logs or restart specific services as needed
docker compose -f docker-compose.yaml logs -f api
docker compose -f docker-compose.yaml restart frontend
```

When you are done:
```bash
docker compose -f docker-compose.yaml down
```

## Build & Push Images

All helper targets live in the root `Makefile`. We use **GitHub Container Registry (GHCR)** exclusively.

### GitHub Container Registry (GHCR)

**Note:** CLI authentication is required even if you've connected GitHub in your IDE.

1. **Authenticate with GHCR:**
   ```bash
   # First, create a Personal Access Token (PAT) at:
   # https://github.com/settings/tokens
   # Required permissions: write:packages, read:packages
   
   # Then login (replace YOUR_TOKEN with your PAT):
   docker login ghcr.io -u Lawrence908 -p YOUR_TOKEN
   
   # Or use the helper command for instructions:
   make login-ghcr
   ```

2. **Build and push production images:**
   ```bash
   # Build production images tagged for GHCR (:latest and :git-<sha>)
   make build-ghcr
   
   # Push to GHCR
   make push-ghcr
   ```

Images will be available at `ghcr.io/lawrence908/astrid-<service>:<tag>`. The username is automatically lowercased for Docker image names (Docker requirement).

## Deploy / Update on Server (Hephaestus)

### Initial Setup

1. **SSH into your server and create the deployment directory:**
   ```bash
   ssh user@hephaestus
   mkdir -p /mnt/storage/astrid
   cd /mnt/storage/astrid
   ```

2. **Create required directories for bind mounts (storage drive):**
   ```bash
   sudo mkdir -p /mnt/storage/apps/astrid/{redis_data,logs,organizr_config,certs}
   sudo chown -R $USER:$USER /mnt/storage/astrid
   ```

3. **Copy deployment files to the server:**
   ```bash
   # From your local machine:
   scp docker-compose.prod.yml user@hephaestus:/mnt/storage/apps/astrid/
   scp .env.prod user@hephaestus:/mnt/storage/apps/astrid/
   scp -r certs/* user@hephaestus:/mnt/storage/apps/astrid/certs/
   ```

4. **Authenticate with GHCR on the server:**
   ```bash
   # On the server:
   docker login ghcr.io -u Lawrence908 -p YOUR_GITHUB_PAT
   ```

5. **Pull images and start the stack:**
   ```bash
   cd /mnt/storage/astrid
   docker compose -f docker-compose.prod.yml pull
   docker compose -f docker-compose.prod.yml up -d
   ```

6. **Verify services are running:**
   ```bash
   docker compose -f docker-compose.prod.yml ps
   docker compose -f docker-compose.prod.yml logs -f
   ```

### Updating Services

When you push new images from your workstation:
```bash
# On the server:
cd /mnt/storage/astrid
docker compose -f docker-compose.prod.yml pull
docker compose -f docker-compose.prod.yml up -d
```

### Storage Location

All persistent data is stored on `/mnt/storage/apps/astrid/` (your storage drive):
- `redis_data/` - Redis database files
- `logs/` - Application logs from all services
- `organizr_config/` - Organizr configuration
- `certs/` - SSL certificates (read-only in containers)

**Note:** If your storage drive is mounted at a different path, update the `device:` paths in `docker-compose.prod.yml` volumes section.

### Stopping the Stack

```bash
cd /mnt/storage/astrid
docker compose -f docker-compose.prod.yml down
```

To also remove volumes (⚠️ deletes data):
```bash
docker compose -f docker-compose.prod.yml down -v
```



