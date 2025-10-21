#!/usr/bin/env python3
"""
Test script for the updated reference dataset endpoint with R2 storage.

This script tests the reference dataset creation endpoint to verify that
FITS files are now being uploaded to R2 storage instead of saved locally.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import after path setup
from src.domains.observations.ingestion.services.data_ingestion import (  # noqa: E402
    DataIngestionService,
)
from src.infrastructure.storage.config import StorageConfig  # noqa: E402


async def test_reference_dataset_r2_storage():
    """Test the reference dataset creation with R2 storage."""
    print("Testing Reference Dataset Creation with R2 Storage")
    print("=" * 60)

    try:
        # Initialize the service
        print("1. Initializing DataIngestionService...")
        service = DataIngestionService()
        print("✓ Service initialized")

        # Test parameters
        ra = 83.633
        dec = 22.0145
        size = 0.25
        pixels = 512
        surveys = ["DSS2 Red"]

        print("\n2. Creating reference dataset...")
        print(f"   RA: {ra}°, Dec: {dec}°")
        print(f"   Size: {size}°, Pixels: {pixels}x{pixels}")
        print(f"   Surveys: {surveys}")

        # Create the reference dataset
        result = await service.create_reference_dataset(
            ra=ra,
            dec=dec,
            size=size,
            pixels=pixels,
            surveys=surveys,
        )

        print("\n3. Results:")
        print("✓ Reference dataset created successfully!")

        # Check if R2 storage was used
        if "r2_object_key" in result:
            print(f"✓ Uploaded to R2: {result['r2_object_key']}")
            print(f"✓ R2 URL: {result['r2_url']}")
            print(f"✓ Bucket: {result['bucket']}")

            # Show metadata
            if "metadata" in result:
                print(f"✓ Metadata: {len(result['metadata'])} fields")
                for key, value in result["metadata"].items():
                    print(f"   {key}: {value}")
        else:
            print("⚠️  R2 upload failed, using local storage as fallback")
            if "error" in result:
                print(f"   Error: {result['error']}")
            if "local_path" in result:
                print(f"   Local path: {result['local_path']}")

        # Show all result fields
        print(f"\n4. Full result keys: {list(result.keys())}")

        return result

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return None


async def test_storage_config():
    """Test storage configuration."""
    print("\nTesting Storage Configuration")
    print("=" * 40)

    try:
        # Set SSL verification to False for development testing
        # In production, this should be True with proper certificates
        os.environ["CLOUDFLARE_R2_VERIFY_SSL"] = "false"

        config = StorageConfig.from_env()
        print("✓ Storage config loaded from environment")

        # Check required fields
        required_fields = [
            "r2_account_id",
            "r2_access_key_id",
            "r2_secret_access_key",
            "r2_endpoint_url",
            "r2_bucket_name",
        ]

        missing_fields = []
        for field in required_fields:
            value = getattr(config, field, None)
            if not value:
                missing_fields.append(field)
            else:
                # Mask sensitive values
                if "key" in field.lower() or "secret" in field.lower():
                    masked_value = f"{value[:8]}..." if len(value) > 8 else "***"
                    print(f"✓ {field}: {masked_value}")
                else:
                    print(f"✓ {field}: {value}")

        if missing_fields:
            print(f"⚠️  Missing required fields: {missing_fields}")
            print("   Set these environment variables:")
            for field in missing_fields:
                env_var = field.upper()
                print(f"   - {env_var}")
        else:
            print("✓ All required configuration fields present")

        return len(missing_fields) == 0

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


async def main():
    """Main test function."""
    print("AstrID R2 Reference Dataset Test")
    print("=" * 50)

    # Test 1: Storage configuration
    config_ok = await test_storage_config()

    if not config_ok:
        print("\n⚠️  Storage configuration issues detected.")
        print("   Please set the required environment variables before testing.")
        return

    # Test 2: Reference dataset creation
    result = await test_reference_dataset_r2_storage()

    if result:
        print("\n🎉 Test completed successfully!")

        # Summary
        if "r2_object_key" in result:
            print("✅ R2 storage integration working correctly")
        else:
            print("⚠️  R2 storage not working, using local fallback")
    else:
        print("\n❌ Test failed")


if __name__ == "__main__":
    asyncio.run(main())
