# Server Deployment Guide

Quick reference for deploying AstrID on Hephaestus or any Linux server.

## Prerequisites

- Docker and Docker Compose installed on the server
- **2TB storage drive mounted at `/mnt/storage`** - Used for all container deployments to preserve OS disk space
- GitHub Personal Access Token (PAT) with `read:packages` permission
  - Use `GITHUB_CONTAINER_ACCESS_TOKEN2` from `github/hephaestus-infra/.env` (has `write:packages` and `repo` privileges)
- Production environment variables ready (`.env.prod` in `/mnt/storage/apps/astrid/`)

## Server Directory Structure

On your server, create this structure:

```
/mnt/storage/apps/astrid/
├── docker-compose.prod.yml    # Compose file (copied from repo)
├── .env.prod                  # Production environment variables
└── certs/                     # SSL certificates
    └── prod-ca-2021.crt
```

**All deployment files go in `/mnt/storage/apps/astrid/`** - this matches your symlink pattern for apps and is the working directory for running `docker compose` commands.

## Quick Start

### 1. Prepare Server Directories

```bash
# SSH into your server
ssh user@hephaestus

# Create deployment directory on storage drive
sudo mkdir -p /mnt/storage/apps/astrid/{redis_data,logs,organizr_config,certs}
sudo chown -R $USER:$USER /mnt/storage/apps/astrid
cd /mnt/storage/apps/astrid
```

### 2. Copy Deployment Files

From your local machine:

```bash
# Copy compose file
scp docker-compose.prod.yml user@hephaestus:/mnt/storage/apps/astrid/

# Copy environment file
# Create .env.prod in the AstrID repo root (or copy from a template)
# Then copy it to the deployment directory on server
scp .env.prod user@hephaestus:/mnt/storage/apps/astrid/

# Copy SSL certificates
scp -r certs/* user@hephaestus:/mnt/storage/apps/astrid/certs/
```

**Note:** `.env.prod` should be:
- Created in the AstrID repo root (for version control, but add to `.gitignore` if it contains secrets)
- Copied to `/mnt/storage/apps/astrid/` on the server (where docker-compose.prod.yml reads it)

### 3. Authenticate with GHCR

On the server:

```bash
# Login to GitHub Container Registry
# Use GITHUB_CONTAINER_ACCESS_TOKEN2 from github/hephaestus-infra/.env
docker login ghcr.io -u Lawrence908 -p GITHUB_CONTAINER_ACCESS_TOKEN2

# (Already done if you've authenticated before)
```

### 4. Deploy

```bash
cd /mnt/storage/apps/astrid

# Pull all images from GHCR
docker compose -f docker-compose.prod.yml pull

# Start all services
docker compose -f docker-compose.prod.yml up -d

# Check status
docker compose -f docker-compose.prod.yml ps

# View logs
docker compose -f docker-compose.prod.yml logs -f
```

## Updating Services

When new images are pushed to GHCR:

```bash
cd /mnt/storage/apps/astrid
docker compose -f docker-compose.prod.yml pull
docker compose -f docker-compose.prod.yml up -d
```

## Service Ports

- **API**: `9001` → `http://hephaestus:9001`
- **Frontend**: `9002` → `http://hephaestus:9002`
- **MLflow**: `9003` → `http://hephaestus:9003`
- **Prefect**: `9004` → `http://hephaestus:9004`
- **Organizr**: `9005` → `http://hephaestus:9005`

## Storage Locations

All data is stored on `/mnt/storage/apps/astrid/`:

- `redis_data/` - Redis database persistence
- `logs/` - Application logs
- `organizr_config/` - Organizr settings
- `certs/` - SSL certificates

## Troubleshooting

### Check service health:
```bash
docker compose -f docker-compose.prod.yml ps
```

### View logs for a specific service:
```bash
docker compose -f docker-compose.prod.yml logs -f api
docker compose -f docker-compose.prod.yml logs -f worker
```

### Restart a service:
```bash
docker compose -f docker-compose.prod.yml restart api
```

### Stop all services:
```bash
docker compose -f docker-compose.prod.yml down
```

### Remove everything (including volumes - ⚠️ deletes data):
```bash
docker compose -f docker-compose.prod.yml down -v
```

## Workflow: Old vs New Approach

### Old Approach (Source Code on Server)
Previously, you would:
1. Clone the GitHub repo to `/mnt/storage/`
2. Run `docker-compose-homelab.yml` from the repo directory
3. Containers would build/run from source code

**Issues:**
- Source code takes up space on storage drive
- Containers built on server (slower, uses more resources)
- Harder to version control deployments

### New Approach (Container Images Only)
Now, you:
1. **Develop locally** on your PC (WSL2) - clone repos, make changes, test
2. **Build images locally** → `make build-ghcr`
3. **Push to GHCR** → `make push-ghcr`
4. **On server: Pull and run** → Just pull images, no source code needed

**Benefits:**
- No source code needed on server (just compose file + .env.prod)
- Faster deployments (pre-built images)
- Cleaner separation: dev on PC, deploy on server
- Easy rollbacks (just pull different image tags)

**Where containers run:**
- **Containers themselves**: Run in memory (temporary)
- **Docker images**: Stored on OS drive by default (`/var/lib/docker`)
- **Application data**: Stored on storage drive (`/mnt/storage/apps/astrid/`) via bind mounts

**Optional: Move Docker data root to storage drive** (see section below)

## Custom Storage Path

If your storage drive is mounted elsewhere, edit `docker-compose.prod.yml` and update the `device:` paths in the volumes section:

```yaml
volumes:
  redis_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /mnt/storage/apps/astrid/redis_data  # Update all to /mnt/storage/apps/astrid/
  logs:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /mnt/storage/apps/astrid/logs
  organizr_config:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /mnt/storage/apps/astrid/organizr_config
```

## Optional: Move Docker Data Root to Storage Drive

By default, Docker stores images, containers, and volumes in `/var/lib/docker` (OS drive). To move everything to your storage drive:

**⚠️ Warning:** This requires stopping Docker and moving data. Do this before deploying containers.

```bash
# 1. Stop Docker
sudo systemctl stop docker

# 2. Move existing Docker data (if any)
sudo mv /var/lib/docker /mnt/storage/docker

# 3. Create symlink (or configure daemon.json)
sudo ln -s /mnt/storage/docker /var/lib/docker

# OR configure daemon.json (recommended):
sudo mkdir -p /etc/docker
echo '{"data-root": "/mnt/storage/docker"}' | sudo tee /etc/docker/daemon.json

# 4. Start Docker
sudo systemctl start docker

# 5. Verify
docker info | grep "Docker Root Dir"
```

**Note:** Application data (Redis, logs, etc.) is already on storage drive via bind mounts, so this is optional. It mainly affects where Docker stores pulled images.

