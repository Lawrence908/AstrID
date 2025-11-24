#!/usr/bin/env python3
"""
Training Data Pipeline Test (ASTR-113)

This script tests the complete training data pipeline from observations →
preprocessing → differencing → detection → training dataset creation.

Usage:
    python test_training_pipeline.py

Prerequisites:
- Services running: api, worker, prefect, mlflow, redis
- Database migrated (includes training tables)
- Environment variables set for Supabase and R2
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configuration
API_BASE = "http://127.0.0.1:8000"
MLFLOW_UI = "http://localhost:9003"
PREFECT_UI = "http://localhost:9004"

# Test parameters
TEST_SURVEY_ID = "hst"
TEST_COUNT = 5  # Start small for testing
TEST_DATE_RANGE = {"start": "2024-01-01T00:00:00", "end": "2024-01-02T23:59:59"}

# Authentication
AUTH_TOKEN = None
AUTH_HEADERS = {}


def authenticate_user(email: str, password: str) -> bool:
    """Authenticate user and get JWT token."""
    global AUTH_TOKEN, AUTH_HEADERS

    url = f"{API_BASE}/signin"
    payload = {"email": email, "password": password}

    print(f"Authenticating user: {email}")

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()

        result = response.json()
        auth_data = result.get("data", {})

        AUTH_TOKEN = auth_data.get("access_token")
        AUTH_HEADERS = {
            "Authorization": f"Bearer {AUTH_TOKEN}",
            "Content-Type": "application/json",
        }

        user_info = auth_data.get("user", {})
        print(f"✅ Successfully authenticated as: {user_info.get('email', 'Unknown')}")
        print(f"   - User ID: {user_info.get('id', 'Unknown')}")
        print(f"   - Role: {user_info.get('role', 'Unknown')}")

        return True

    except requests.exceptions.RequestException as e:
        print(f"❌ Authentication failed: {str(e)}")
        if hasattr(e, "response") and e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"   Error details: {error_detail}")
            except Exception:
                print(f"   Response: {e.response.text}")
        return False


def check_service_health(url: str, service_name: str) -> bool:
    """Check if a service is healthy."""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"✅ {service_name}: Healthy")
            return True
        else:
            print(f"❌ {service_name}: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ {service_name}: {str(e)}")
        return False


def ingest_test_observations() -> dict[str, Any]:
    """Ingest test observations using the API."""
    if not AUTH_TOKEN:
        return {
            "success": False,
            "error": "Not authenticated. Please run authenticate_user() first.",
        }

    url = f"{API_BASE}/observations/ingest/batch-random"
    payload = {"count": TEST_COUNT, "survey_id": TEST_SURVEY_ID}

    print(f"Ingesting {TEST_COUNT} test observations for survey {TEST_SURVEY_ID}...")

    try:
        response = requests.post(url, json=payload, headers=AUTH_HEADERS, timeout=30)
        response.raise_for_status()

        result = response.json()
        observations = result.get("data", [])

        print(f"✅ Successfully ingested {len(observations)} observations")

        # Store observation IDs for later use
        observation_ids = [obs["id"] for obs in observations]

        return {
            "success": True,
            "count": len(observations),
            "observation_ids": observation_ids,
            "observations": observations,
        }

    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to ingest observations: {str(e)}")
        return {"success": False, "error": str(e)}


def run_preprocessing(observation_ids: list[str]) -> dict[str, Any]:
    """Run preprocessing on the ingested observations."""
    if not observation_ids:
        return {"success": False, "error": "No observation IDs provided"}

    url = f"{API_BASE}/preprocessing/process/batch"
    payload = {"observation_ids": observation_ids, "force_reprocess": False}

    print(f"Running preprocessing on {len(observation_ids)} observations...")

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()

        result = response.json()
        preprocess_runs = result.get("data", [])

        print(f"✅ Successfully processed {len(preprocess_runs)} observations")

        return {
            "success": True,
            "count": len(preprocess_runs),
            "preprocess_runs": preprocess_runs,
        }

    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to run preprocessing: {str(e)}")
        return {"success": False, "error": str(e)}


def run_differencing(observation_ids: list[str]) -> dict[str, Any]:
    """Run differencing on the preprocessed observations."""
    if not observation_ids:
        return {"success": False, "error": "No observation IDs provided"}

    url = f"{API_BASE}/differencing/process/batch"
    payload = {"observation_ids": observation_ids, "reference_strategy": "skyview"}

    print(f"Running differencing on {len(observation_ids)} observations...")

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()

        result = response.json()
        difference_runs = result.get("data", [])

        print(f"✅ Successfully created {len(difference_runs)} difference images")

        return {
            "success": True,
            "count": len(difference_runs),
            "difference_runs": difference_runs,
        }

    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to run differencing: {str(e)}")
        return {"success": False, "error": str(e)}


def run_detection(observation_ids: list[str]) -> dict[str, Any]:
    """Run ML detection on the difference images."""
    if not observation_ids:
        return {"success": False, "error": "No observation IDs provided"}

    url = f"{API_BASE}/detection/batch-process"
    payload = {"observation_ids": observation_ids, "model_id": "latest"}

    print(f"Running ML detection on {len(observation_ids)} observations...")

    try:
        response = requests.post(url, json=payload, timeout=180)
        response.raise_for_status()

        result = response.json()
        detections = result.get("data", [])

        print(f"✅ Successfully processed {len(detections)} detections")

        # Count detections with confidence > 0
        high_confidence = [d for d in detections if d.get("confidence_score", 0) > 0.5]
        print(f"   - {len(high_confidence)} detections with confidence > 0.5")

        return {
            "success": True,
            "count": len(detections),
            "high_confidence_count": len(high_confidence),
            "detections": detections,
        }

    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to run detection: {str(e)}")
        return {"success": False, "error": str(e)}


def create_training_dataset() -> dict[str, Any]:
    """Create training dataset from the processed detections."""
    if not AUTH_TOKEN:
        return {
            "success": False,
            "error": "Not authenticated. Please run authenticate_user() first.",
        }

    url = f"{API_BASE}/training/datasets/collect"
    payload = {
        "survey_ids": [TEST_SURVEY_ID],
        "start": TEST_DATE_RANGE["start"],
        "end": TEST_DATE_RANGE["end"],
        "confidence_threshold": 0.5,  # Lower threshold for testing
        "max_samples": 100,
        "name": f"test_{TEST_SURVEY_ID}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    }

    print(f"Creating training dataset for survey {TEST_SURVEY_ID}...")
    print(f"Date range: {payload['start']} to {payload['end']}")
    print(f"Confidence threshold: {payload['confidence_threshold']}")

    try:
        response = requests.post(url, json=payload, headers=AUTH_HEADERS, timeout=60)
        response.raise_for_status()

        result = response.json()
        dataset_info = result.get("data", {})

        print("✅ Successfully created training dataset")
        print(f"   - Dataset ID: {dataset_info.get('dataset_id')}")
        print(f"   - Name: {dataset_info.get('name')}")
        print(f"   - Total samples: {dataset_info.get('total')}")
        print(
            f"   - Quality score: {dataset_info.get('quality', {}).get('quality_score', 0):.3f}"
        )

        return {
            "success": True,
            "dataset_id": dataset_info.get("dataset_id"),
            "dataset_info": dataset_info,
        }

    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to create training dataset: {str(e)}")
        return {"success": False, "error": str(e)}


def list_training_datasets() -> dict[str, Any]:
    """List all available training datasets."""
    url = f"{API_BASE}/training/datasets"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        result = response.json()
        datasets = result.get("data", [])

        print(f"✅ Found {len(datasets)} training datasets")

        for i, dataset in enumerate(datasets, 1):
            print(f"\n{i}. {dataset.get('name')}")
            print(f"   - ID: {dataset.get('id')}")
            print(f"   - Samples: {dataset.get('total_samples')}")
            print(f"   - Quality: {dataset.get('quality_score', 0):.3f}")
            print(f"   - Status: {dataset.get('status')}")
            print(f"   - Created: {dataset.get('created_at')}")

        return {"success": True, "datasets": datasets}

    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to list datasets: {str(e)}")
        return {"success": False, "error": str(e)}


def main():
    """Run the complete training data pipeline test."""
    print("=" * 60)
    print("TRAINING DATA PIPELINE TEST (ASTR-113)")
    print("=" * 60)

    # Step 0: Authentication
    print("\n🔐 Step 0: Authentication")
    print("Please provide your credentials:")
    email = input("Email: ").strip()
    password = input("Password: ").strip()

    if not authenticate_user(email, password):
        print("\n❌ Authentication failed. Cannot proceed.")
        return

    # Step 1: Health Checks
    print("\n🔍 Step 1: Health Checks")
    services = {"API": f"{API_BASE}/health", "MLflow": MLFLOW_UI, "Prefect": PREFECT_UI}

    all_healthy = True
    for name, url in services.items():
        if not check_service_health(url, name):
            all_healthy = False

    if not all_healthy:
        print("\n⚠️  Some services are not healthy. Please check docker-compose.")
        print("Run: docker-compose up -d api worker prefect mlflow redis")
        return

    print("\n🎉 All services are healthy!")

    # Step 2: Ingest Observations
    print("\n🔍 Step 2: Ingest Test Observations")
    ingestion_result = ingest_test_observations()

    if not ingestion_result["success"]:
        print(f"\n❌ Cannot proceed without observations: {ingestion_result['error']}")
        return

    observation_ids = ingestion_result["observation_ids"]
    print(f"\nObservation IDs: {observation_ids}")

    # Step 3: Run Preprocessing
    print("\n🔍 Step 3: Run Preprocessing")
    preprocessing_result = run_preprocessing(observation_ids)

    if not preprocessing_result["success"]:
        print(f"\n❌ Preprocessing failed: {preprocessing_result['error']}")
        return

    # Step 4: Run Differencing
    print("\n🔍 Step 4: Run Differencing")
    differencing_result = run_differencing(observation_ids)

    if not differencing_result["success"]:
        print(f"\n❌ Differencing failed: {differencing_result['error']}")
        return

    # Step 5: Run Detection
    print("\n🔍 Step 5: Run ML Detection")
    detection_result = run_detection(observation_ids)

    if not detection_result["success"]:
        print(f"\n❌ Detection failed: {detection_result['error']}")
        return

    print(f"\nTotal detections: {detection_result['count']}")
    print(f"High confidence detections: {detection_result['high_confidence_count']}")

    # Step 6: Create Training Dataset
    print("\n🔍 Step 6: Create Training Dataset")
    dataset_result = create_training_dataset()

    if not dataset_result["success"]:
        print(f"\n❌ Training dataset creation failed: {dataset_result['error']}")
        return

    dataset_id = dataset_result["dataset_id"]
    dataset_info = dataset_result["dataset_info"]

    # Step 7: List All Datasets
    print("\n🔍 Step 7: List All Training Datasets")
    list_result = list_training_datasets()

    if list_result["success"]:
        datasets = list_result["datasets"]
        datasets_with_samples = [d for d in datasets if d.get("total_samples", 0) > 0]

        if datasets_with_samples:
            print(f"\n🎉 Found {len(datasets_with_samples)} datasets with samples!")
            print("\nReady for training:")
            for dataset in datasets_with_samples:
                print(
                    f"- {dataset['name']} (ID: {dataset['id']}) - {dataset['total_samples']} samples"
                )
        else:
            print("\n⚠️  No datasets have samples yet.")
            print("This suggests the data pipeline needs to be run first.")

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE TEST SUMMARY")
    print("=" * 60)

    if dataset_info.get("total", 0) > 0:
        print("\n🎉 SUCCESS: Training data pipeline is working!")
        print(f"\nReady for training with dataset: {dataset_id}")
        print("\nNext steps:")
        print("1. Open the training notebook: notebooks/training/model_training.ipynb")
        print(f"2. Use dataset_id: {dataset_id}")
        print("3. Replace synthetic data with real data loading")
    else:
        print("\n⚠️  PARTIAL SUCCESS: Pipeline ran but no training samples found")
        print("\nTry:")
        print("- Lowering confidence_threshold (try 0.3 or 0.1)")
        print("- Expanding date range (try last 30 days)")
        print("- Checking if detection pipeline ran successfully")
        print("- Verifying observations were ingested and preprocessed")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
