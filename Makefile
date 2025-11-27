REGISTRY ?= chrislawrencedev
PROJECT ?= astrid
GITHUB_USER ?= Lawrence908
GHCR_REGISTRY = ghcr.io
# Convert GitHub username to lowercase for Docker image names (Docker requires lowercase)
GHCR_USER := $(shell echo $(GITHUB_USER) | tr '[:upper:]' '[:lower:]')
SERVICES := \
	api:Dockerfile.api.optimized:. \
	worker:Dockerfile.worker:. \
	prefect-worker:Dockerfile.prefect.optimized:. \
	prefect-server:Dockerfile.prefect-server:. \
	mlflow:Dockerfile.mlflow:. \
	frontend:frontend/Dockerfile:.
GIT_SHA := $(shell git rev-parse --short HEAD 2>/dev/null || date +%s)

.PHONY: build-dev push-dev build-prod push-prod build-tag push-tag login login-ghcr build-ghcr push-ghcr tag-ghcr push-ghcr-tag

build-dev:
	docker compose -f docker-compose.yaml build

push-dev:
	docker compose -f docker-compose.yaml push

build-prod:
	@for tag in latest git-$(GIT_SHA); do \
		$(MAKE) build-tag TAG=$$tag; \
	done

push-prod:
	@for tag in latest git-$(GIT_SHA); do \
		$(MAKE) push-tag TAG=$$tag; \
	done

build-tag:
ifndef TAG
	$(error TAG is not set)
endif
	@set -e; \
	for entry in $(SERVICES); do \
		service=$${entry%%:*}; \
		rest=$${entry#*:}; \
		dockerfile=$${rest%%:*}; \
		context=$${rest#*:}; \
		echo "==> Building $(REGISTRY)/$(PROJECT)-$$service:$(TAG)"; \
		docker build -f $$dockerfile -t $(REGISTRY)/$(PROJECT)-$$service:$(TAG) $$context; \
	done

push-tag:
ifndef TAG
	$(error TAG is not set)
endif
	@set -e; \
	for entry in $(SERVICES); do \
		service=$${entry%%:*}; \
		echo "==> Pushing $(REGISTRY)/$(PROJECT)-$$service:$(TAG)"; \
		docker push $(REGISTRY)/$(PROJECT)-$$service:$(TAG); \
	done

# GitHub Container Registry targets
build-ghcr:
	@for tag in latest git-$(GIT_SHA); do \
		$(MAKE) build-ghcr-tag TAG=$$tag; \
	done

build-ghcr-tag:
ifndef TAG
	$(error TAG is not set)
endif
	@set -e; \
	for entry in $(SERVICES); do \
		service=$${entry%%:*}; \
		rest=$${entry#*:}; \
		dockerfile=$${rest%%:*}; \
		context=$${rest#*:}; \
		echo "==> Building $(GHCR_REGISTRY)/$(GHCR_USER)/$(PROJECT)-$$service:$(TAG)"; \
		docker build -f $$dockerfile -t $(GHCR_REGISTRY)/$(GHCR_USER)/$(PROJECT)-$$service:$(TAG) $$context; \
	done

push-ghcr:
	@for tag in latest git-$(GIT_SHA); do \
		$(MAKE) push-ghcr-tag TAG=$$tag; \
	done

push-ghcr-tag:
ifndef TAG
	$(error TAG is not set)
endif
	@set -e; \
	for entry in $(SERVICES); do \
		service=$${entry%%:*}; \
		echo "==> Pushing $(GHCR_REGISTRY)/$(GHCR_USER)/$(PROJECT)-$$service:$(TAG)"; \
		docker push $(GHCR_REGISTRY)/$(GHCR_USER)/$(PROJECT)-$$service:$(TAG); \
	done

# Tag existing Docker Hub images for GHCR (useful if already built)
tag-ghcr:
	@for tag in latest git-$(GIT_SHA); do \
		$(MAKE) tag-ghcr-tag TAG=$$tag; \
	done

tag-ghcr-tag:
ifndef TAG
	$(error TAG is not set)
endif
	@set -e; \
	for entry in $(SERVICES); do \
		service=$${entry%%:*}; \
		echo "==> Tagging $(REGISTRY)/$(PROJECT)-$$service:$(TAG) -> $(GHCR_REGISTRY)/$(GHCR_USER)/$(PROJECT)-$$service:$(TAG)"; \
		docker tag $(REGISTRY)/$(PROJECT)-$$service:$(TAG) $(GHCR_REGISTRY)/$(GHCR_USER)/$(PROJECT)-$$service:$(TAG); \
	done

login:
	@echo "Run 'docker login' before pushing images to Docker Hub (credentials not stored)."
	@echo "  docker login"

login-ghcr:
	@echo "To authenticate with GitHub Container Registry:"
	@echo "  1. Create a Personal Access Token (PAT) at https://github.com/settings/tokens"
	@echo "     with 'write:packages' and 'read:packages' permissions"
	@echo "  2. Run: docker login $(GHCR_REGISTRY) -u $(GITHUB_USER) -p YOUR_TOKEN"
	@echo ""
	@echo "Note: Images will be tagged as $(GHCR_REGISTRY)/$(GHCR_USER)/* (lowercase username)"
	@echo "Or use your GitHub password (PAT recommended):"
	@echo "  docker login $(GHCR_REGISTRY) -u $(GITHUB_USER)"
