#!/usr/bin/env python3
"""
Linear Ticket Creator for AstrID Project
Creates all tickets from the project plan automatically.

Usage:
    python create_linear_tickets.py
"""

import time

import requests


class LinearTicketCreator:
    def __init__(self, api_key: str, team_id: str):
        self.api_key = api_key
        self.team_id = team_id
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.endpoint = "https://api.linear.app/graphql"

        # Cache for labels and projects
        self.labels_cache = {}
        self.projects_cache = {}

    def get_labels(self) -> dict[str, str]:
        """Fetch all labels and cache them by name."""
        if self.labels_cache:
            return self.labels_cache

        # Updated query for current Linear API
        query = """
        query GetLabels($teamId: String!) {
          team(key: $teamId) {
            labels {
              nodes {
                id
                name
              }
            }
          }
        }
        """

        variables = {"teamId": self.team_id}

        response = requests.post(
            self.endpoint,
            headers=self.headers,
            json={"query": query, "variables": variables},
        )

        if response.status_code == 200:
            data = response.json()
            team_data = data.get("data", {}).get("team", {})
            if team_data:
                labels = team_data.get("labels", {}).get("nodes", [])
                self.labels_cache = {label["name"]: label["id"] for label in labels}
                return self.labels_cache
            else:
                raise Exception("Team not found or no access")
        else:
            raise Exception(f"Failed to fetch labels: {response.status_code}")

    def get_projects(self) -> dict[str, str]:
        """Fetch all projects and cache them by name."""
        if self.projects_cache:
            return self.projects_cache

        # Updated query for current Linear API
        query = """
        query GetProjects($teamId: String!) {
          team(key: $teamId) {
            projects {
              nodes {
                id
                name
              }
            }
          }
        }
        """

        variables = {"teamId": self.team_id}

        response = requests.post(
            self.endpoint,
            headers=self.headers,
            json={"query": query, "variables": variables},
        )

        if response.status_code == 200:
            data = response.json()
            team_data = data.get("data", {}).get("team", {})
            if team_data:
                projects = team_data.get("projects", {}).get("nodes", [])
                self.projects_cache = {
                    project["name"]: project["id"] for project in projects
                }
                return self.projects_cache
            else:
                raise Exception("Team not found or no access")
        else:
            raise Exception(f"Failed to fetch projects: {response.status_code}")

    def create_ticket(
        self,
        title: str,
        description: str,
        priority: int,
        label_names: list[str],
        project_name: str,
        estimated_time: str,
        dependencies: str,
    ) -> str | None:
        """Create a single ticket in Linear."""

        # Get label IDs
        labels = self.get_labels()
        label_ids = []
        for label_name in label_names:
            if label_name in labels:
                label_ids.append(labels[label_name])
            else:
                print(f"Warning: Label '{label_name}' not found")

        # Get project ID
        projects = self.get_projects()
        if project_name not in projects:
            print(f"Error: Project '{project_name}' not found")
            return None

        project_id = projects[project_name]

        # Create the ticket
        mutation = """
        mutation CreateIssue($input: IssueCreateInput!) {
          issueCreate(input: $input) {
            success
            issue {
              id
              title
              number
            }
            errors {
              message
            }
          }
        }
        """

        variables = {
            "input": {
                "title": title,
                "description": description,
                "priority": priority,
                "labelIds": label_ids,
                "projectId": project_id,
                "teamKey": self.team_id,
            }
        }

        response = requests.post(
            self.endpoint,
            headers=self.headers,
            json={"query": mutation, "variables": variables},
        )

        if response.status_code == 200:
            data = response.json()
            result = data.get("data", {}).get("issueCreate", {})

            if result.get("success"):
                issue = result["issue"]
                print(f"✅ Created: {issue['title']} (#{issue['number']})")
                return issue["id"]
            else:
                errors = result.get("errors", [])
                print(f"❌ Failed to create '{title}': {errors}")
                return None
        else:
            print(f"❌ HTTP error creating '{title}': {response.status_code}")
            return None

    def create_all_tickets(self):
        """Create all tickets from the project plan."""

        # Ticket definitions
        tickets = [
            {
                "title": "Development Environment Setup",
                "description": "Set up complete development environment for AstrID project\n\nSubtasks:\n• Install Python 3.11 and uv package manager\n• Configure pre-commit hooks (Ruff, Black, MyPy)\n• Set up Docker development environment\n• Configure environment variables and secrets",
                "priority": 1,
                "labels": ["infrastructure", "high-priority"],
                "project": "ASTRID-INFRA",
                "estimated_time": "2 days",
                "dependencies": "",
            },
            {
                "title": "Database Setup and Migrations",
                "description": "Design and implement database schema with migration system\n\nSubtasks:\n• Design database schema for observations, detections, etc.\n• Implement SQLAlchemy 2 models\n• Create Alembic migration scripts\n• Set up test database configuration",
                "priority": 1,
                "labels": ["infrastructure", "database", "high-priority"],
                "project": "ASTRID-INFRA",
                "estimated_time": "3 days",
                "dependencies": "INFRA-001",
            },
            {
                "title": "INFRA-003: Cloud Storage Integration",
                "description": "Configure cloud storage for datasets and artifacts\n\nSubtasks:\n• Configure Cloudflare R2 (S3-compatible) storage\n• Implement storage client with content addressing\n• Set up DVC for dataset versioning\n• Configure MLflow artifact storage",
                "priority": 2,
                "labels": ["infrastructure"],
                "project": "ASTRID-INFRA",
                "estimated_time": "2 days",
                "dependencies": "INFRA-001",
            },
            {
                "title": "AUTH-001: Supabase Integration",
                "description": "Implement authentication and authorization system\n\nSubtasks:\n• Set up Supabase project\n• Implement JWT authentication\n• Create role-based access control\n• Add API key management",
                "priority": 2,
                "labels": ["infrastructure", "security"],
                "project": "ASTRID-INFRA",
                "estimated_time": "2 days",
                "dependencies": "INFRA-001",
            },
            {
                "title": "Observation Models and Services",
                "description": "Implement core observation domain models and business logic\n\nSubtasks:\n• Implement Observation domain models\n• Create observation repository interface\n• Implement observation service layer\n• Add observation validation logic",
                "priority": 1,
                "labels": ["core-domain", "high-priority"],
                "project": "ASTRID-CORE",
                "estimated_time": "3 days",
                "dependencies": "INFRA-002",
            },
            {
                "title": "OBS-002: Survey Integration",
                "description": "Integrate with external astronomical survey APIs\n\nSubtasks:\n• Integrate with MAST API for observations\n• Integrate with SkyView for image data\n• Implement survey-specific adapters\n• Add observation metadata extraction",
                "priority": 2,
                "labels": ["core-domain", "integration"],
                "project": "ASTRID-CORE",
                "estimated_time": "4 days",
                "dependencies": "OBS-001",
            },
            {
                "title": "OBS-003: FITS Processing Pipeline",
                "description": "Implement FITS file processing and WCS handling\n\nSubtasks:\n• Implement FITS file reading and writing\n• Add WCS (World Coordinate System) handling\n• Create image metadata extraction\n• Implement star catalog integration",
                "priority": 2,
                "labels": ["core-domain", "data-processing"],
                "project": "ASTRID-CORE",
                "estimated_time": "3 days",
                "dependencies": "OBS-001",
            },
            {
                "title": "PREP-001: Image Preprocessing Services",
                "description": "Implement image calibration and preprocessing pipeline\n\nSubtasks:\n• Implement bias/dark/flat calibration\n• Add WCS alignment and registration\n• Create image quality assessment\n• Implement preprocessing pipeline orchestration",
                "priority": 2,
                "labels": ["core-domain", "image-processing"],
                "project": "ASTRID-CORE",
                "estimated_time": "4 days",
                "dependencies": "OBS-003",
            },
            {
                "title": "PREP-002: Astronomical Image Processing",
                "description": "Advanced image processing with OpenCV and scikit-image\n\nSubtasks:\n• Integrate OpenCV for image manipulation\n• Add scikit-image for advanced processing\n• Implement image normalization and scaling\n• Create preprocessing result storage",
                "priority": 3,
                "labels": ["core-domain", "image-processing"],
                "project": "ASTRID-CORE",
                "estimated_time": "3 days",
                "dependencies": "PREP-001",
            },
            {
                "title": "DIFF-001: Image Differencing Algorithms",
                "description": "Implement image differencing algorithms for anomaly detection\n\nSubtasks:\n• Implement ZOGY algorithm\n• Add classic differencing methods\n• Create reference image selection logic\n• Implement difference image generation",
                "priority": 2,
                "labels": ["core-domain", "algorithm"],
                "project": "ASTRID-CORE",
                "estimated_time": "4 days",
                "dependencies": "PREP-001",
            },
            {
                "title": "DIFF-002: Source Extraction",
                "description": "Extract and analyze sources from difference images\n\nSubtasks:\n• Integrate SEP for source extraction\n• Add photutils for additional analysis\n• Implement candidate filtering\n• Create candidate scoring system",
                "priority": 2,
                "labels": ["core-domain", "algorithm"],
                "project": "ASTRID-CORE",
                "estimated_time": "3 days",
                "dependencies": "DIFF-001",
            },
            {
                "title": "DET-001: U-Net Model Integration",
                "description": "Integrate existing U-Net model into new architecture\n\nSubtasks:\n• Port existing U-Net model to new architecture\n• Implement model loading and inference\n• Add confidence scoring\n• Create model performance tracking",
                "priority": 2,
                "labels": ["ml", "model-integration"],
                "project": "ASTRID-ML",
                "estimated_time": "3 days",
                "dependencies": "DIFF-002",
            },
            {
                "title": "DET-002: Anomaly Detection Pipeline",
                "description": "Complete anomaly detection service implementation\n\nSubtasks:\n• Implement detection service layer\n• Add detection validation logic\n• Create detection result storage\n• Implement detection metrics calculation",
                "priority": 2,
                "labels": ["ml", "pipeline"],
                "project": "ASTRID-ML",
                "estimated_time": "3 days",
                "dependencies": "DET-001",
            },
            {
                "title": "CUR-001: Human Validation System",
                "description": "Create human validation interface for detected anomalies\n\nSubtasks:\n• Create validation interface\n• Implement curator management\n• Add validation event tracking\n• Create feedback collection system",
                "priority": 3,
                "labels": ["core-domain", "ui"],
                "project": "ASTRID-CORE",
                "estimated_time": "4 days",
                "dependencies": "DET-002",
            },
            {
                "title": "CAT-001: Data Cataloging",
                "description": "Implement data cataloging and export functionality\n\nSubtasks:\n• Implement catalog entry creation\n• Add analytics and reporting\n• Create data export functionality\n• Implement catalog search and filtering",
                "priority": 3,
                "labels": ["core-domain", "data"],
                "project": "ASTRID-CORE",
                "estimated_time": "3 days",
                "dependencies": "CUR-001",
            },
            {
                "title": "API-001: Core API Endpoints",
                "description": "Implement core API endpoints for all domains\n\nSubtasks:\n• Implement observations endpoints\n• Add detections endpoints\n• Create streaming endpoints (SSE)\n• Add health check and monitoring endpoints",
                "priority": 1,
                "labels": ["api", "high-priority"],
                "project": "ASTRID-API",
                "estimated_time": "4 days",
                "dependencies": "OBS-001,DET-002",
            },
            {
                "title": "API-002: API Documentation and Testing",
                "description": "Comprehensive API documentation and testing\n\nSubtasks:\n• Add comprehensive API documentation\n• Implement API testing suite\n• Add API versioning\n• Create API rate limiting",
                "priority": 2,
                "labels": ["api", "testing"],
                "project": "ASTRID-API",
                "estimated_time": "2 days",
                "dependencies": "API-001",
            },
            {
                "title": "UI-001: Next.js Dashboard Setup",
                "description": "Set up Next.js dashboard with authentication\n\nSubtasks:\n• Set up Next.js project with TypeScript\n• Implement Tailwind CSS styling\n• Create responsive layout components\n• Add authentication integration",
                "priority": 3,
                "labels": ["ui", "frontend"],
                "project": "ASTRID-API",
                "estimated_time": "3 days",
                "dependencies": "API-001",
            },
            {
                "title": "UI-002: Dashboard Features",
                "description": "Implement core dashboard functionality\n\nSubtasks:\n• Create observation overview dashboard\n• Implement detection visualization\n• Add real-time streaming updates\n• Create user management interface",
                "priority": 3,
                "labels": ["ui", "frontend"],
                "project": "ASTRID-API",
                "estimated_time": "4 days",
                "dependencies": "UI-001",
            },
            {
                "title": "ML-001: MLflow Integration",
                "description": "Set up MLflow for experiment tracking and model management\n\nSubtasks:\n• Set up MLflow tracking server\n• Implement experiment tracking\n• Add model registry functionality\n• Create model versioning system",
                "priority": 2,
                "labels": ["ml", "infrastructure"],
                "project": "ASTRID-ML",
                "estimated_time": "2 days",
                "dependencies": "INFRA-003",
            },
            {
                "title": "ML-002: Model Training Pipeline",
                "description": "Automated model training and optimization workflows\n\nSubtasks:\n• Implement automated training workflows\n• Add hyperparameter optimization\n• Create model evaluation metrics\n• Implement model deployment automation",
                "priority": 3,
                "labels": ["ml", "pipeline"],
                "project": "ASTRID-ML",
                "estimated_time": "4 days",
                "dependencies": "ML-001",
            },
            {
                "title": "MLOPS-001: Model Serving",
                "description": "Production model serving and monitoring\n\nSubtasks:\n• Implement model inference endpoints\n• Add model performance monitoring\n• Create A/B testing framework\n• Implement model rollback capabilities",
                "priority": 3,
                "labels": ["ml", "mlops"],
                "project": "ASTRID-ML",
                "estimated_time": "3 days",
                "dependencies": "ML-002",
            },
            {
                "title": "WORK-001: Workflow Orchestration",
                "description": "Set up Prefect for workflow orchestration\n\nSubtasks:\n• Set up Prefect server\n• Implement observation processing flows\n• Add model training workflows\n• Create monitoring and alerting",
                "priority": 2,
                "labels": ["workflow", "orchestration"],
                "project": "ASTRID-WORK",
                "estimated_time": "3 days",
                "dependencies": "INFRA-001",
            },
            {
                "title": "WORK-002: Dramatiq Workers",
                "description": "Implement background processing workers\n\nSubtasks:\n• Implement observation ingestion workers\n• Add preprocessing workers\n• Create differencing workers\n• Implement detection workers",
                "priority": 2,
                "labels": ["workflow", "background"],
                "project": "ASTRID-WORK",
                "estimated_time": "4 days",
                "dependencies": "WORK-001",
            },
            {
                "title": "TEST-001: Test Framework Setup",
                "description": "Set up comprehensive testing infrastructure\n\nSubtasks:\n• Configure pytest with async support\n• Set up test database fixtures\n• Implement mock services\n• Add test coverage reporting",
                "priority": 1,
                "labels": ["testing", "high-priority"],
                "project": "ASTRID-TEST",
                "estimated_time": "2 days",
                "dependencies": "INFRA-001",
            },
            {
                "title": "TEST-002: Test Implementation",
                "description": "Implement comprehensive test coverage\n\nSubtasks:\n• Write unit tests for all domains\n• Add integration tests for API\n• Implement end-to-end tests\n• Add performance and load tests",
                "priority": 2,
                "labels": ["testing"],
                "project": "ASTRID-TEST",
                "estimated_time": "5 days",
                "dependencies": "TEST-001",
            },
            {
                "title": "QUAL-001: Code Quality Tools",
                "description": "Configure code quality and formatting tools\n\nSubtasks:\n• Configure Ruff for linting\n• Set up MyPy for type checking\n• Implement Black code formatting\n• Add pre-commit hooks",
                "priority": 2,
                "labels": ["testing", "quality"],
                "project": "ASTRID-TEST",
                "estimated_time": "1 day",
                "dependencies": "INFRA-001",
            },
            {
                "title": "DEPLOY-001: Docker Setup",
                "description": "Containerize all services for deployment\n\nSubtasks:\n• Create API Dockerfile\n• Create worker Dockerfile\n• Set up Docker Compose for development\n• Implement health checks",
                "priority": 2,
                "labels": ["deployment", "docker"],
                "project": "ASTRID-DEPLOY",
                "estimated_time": "2 days",
                "dependencies": "API-001",
            },
            {
                "title": "DEPLOY-002: Production Setup",
                "description": "Configure production environment and monitoring\n\nSubtasks:\n• Configure production environment\n• Set up monitoring and logging\n• Implement backup and recovery\n• Add performance monitoring",
                "priority": 3,
                "labels": ["deployment", "production"],
                "project": "ASTRID-DEPLOY",
                "estimated_time": "3 days",
                "dependencies": "DEPLOY-001",
            },
            {
                "title": "CI-001: GitHub Actions",
                "description": "Set up automated CI/CD pipeline\n\nSubtasks:\n• Set up automated testing\n• Add code quality checks\n• Implement automated deployment\n• Add security scanning",
                "priority": 2,
                "labels": ["deployment", "ci-cd"],
                "project": "ASTRID-DEPLOY",
                "estimated_time": "2 days",
                "dependencies": "TEST-001",
            },
            {
                "title": "DOC-001: Technical Documentation",
                "description": "Create comprehensive technical documentation\n\nSubtasks:\n• Write API documentation\n• Create architecture documentation\n• Add deployment guides\n• Write user manuals",
                "priority": 3,
                "labels": ["documentation"],
                "project": "ASTRID-DOCS",
                "estimated_time": "4 days",
                "dependencies": "API-001",
            },
            {
                "title": "TRAIN-001: User Training",
                "description": "Create user training materials and onboarding\n\nSubtasks:\n• Create user training materials\n• Implement onboarding process\n• Add help and support documentation\n• Create video tutorials",
                "priority": 4,
                "labels": ["documentation", "training"],
                "project": "ASTRID-DOCS",
                "estimated_time": "3 days",
                "dependencies": "DOC-001",
            },
        ]

        print(f"🚀 Creating {len(tickets)} tickets in Linear...")
        print("=" * 60)

        created_count = 0
        failed_count = 0

        for i, ticket in enumerate(tickets, 1):
            print(f"\n[{i}/{len(tickets)}] Creating: {ticket['title']}")

            ticket_id = self.create_ticket(
                title=ticket["title"],
                description=ticket["description"],
                priority=ticket["priority"],
                label_names=ticket["labels"],
                project_name=ticket["project"],
                estimated_time=ticket["estimated_time"],
                dependencies=ticket["dependencies"],
            )

            if ticket_id:
                created_count += 1
            else:
                failed_count += 1

            # Rate limiting - be nice to Linear's API
            time.sleep(0.5)

        print("\n" + "=" * 60)
        print(f"✅ Successfully created: {created_count} tickets")
        if failed_count > 0:
            print(f"❌ Failed to create: {failed_count} tickets")
        print("🎉 Ticket creation complete!")


def main():
    """Main function to run the ticket creator."""

    print("🎫 Linear Ticket Creator for AstrID Project")
    print("=" * 50)

    # Get configuration from user
    api_key = input("Enter your Linear API key: ").strip()
    if not api_key:
        print("❌ API key is required!")
        return

    team_id = input("Enter your Linear team ID: ").strip()
    if not team_id:
        print("❌ Team ID is required!")
        return

    print(f"\n🔑 Using API key: {api_key[:8]}...")
    print(f"👥 Using team ID: {team_id}")

    # Confirm before proceeding
    confirm = input("\nProceed with creating all tickets? (y/N): ").strip().lower()
    if confirm != "y":
        print("❌ Cancelled!")
        return

    try:
        creator = LinearTicketCreator(api_key, team_id)
        creator.create_all_tickets()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure:")
        print("   • Your API key is valid")
        print("   • Your team ID is correct")
        print("   • You have permission to create tickets")
        print("   • All labels and projects exist")


if __name__ == "__main__":
    main()
