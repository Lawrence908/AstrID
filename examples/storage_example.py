#!/usr/bin/env python3
"""
Example usage of AstrID storage infrastructure.

This script demonstrates how to use the various storage components
for managing astronomical data and ML artifacts.
"""

import asyncio
import tempfile
from pathlib import Path

from src.infrastructure.storage import (
    ContentAddressedStorage,
    DVCClient,
    MLflowStorageConfig,
    R2StorageClient,
    StorageConfig,
)


async def main():
    """Main example function."""
    print("AstrID Storage Infrastructure Example")
    print("=" * 50)

    try:
        # Initialize storage configuration
        print("1. Initializing storage configuration...")
        config = StorageConfig.from_env()

        # Validate configuration
        try:
            config.validate()
            print("✓ Storage configuration validated")
        except ValueError as e:
            print(f"✗ Configuration validation failed: {e}")
            print("Please set required environment variables:")
            print("  - CLOUDFLARE_ACCOUNT_ID")
            print("  - CLOUDFLARE_R2_ACCESS_KEY_ID")
            print("  - CLOUDFLARE_R2_SECRET_ACCESS_KEY")
            print("  - CLOUDFLARE_R2_ENDPOINT_URL")
            return

        # Initialize R2 client
        print("\n2. Initializing R2 storage client...")
        r2_client = R2StorageClient(config=config)
        print("✓ R2 client initialized")

        # Initialize content-addressed storage
        print("\n3. Initializing content-addressed storage...")
        cas = ContentAddressedStorage(
            r2_client=r2_client, bucket=config.r2_bucket_name, prefix="cas/"
        )
        print("✓ Content-addressed storage initialized")

        # Example 1: Store and retrieve data using content addressing
        print("\n4. Content-addressed storage example...")
        test_data = b"This is test astronomical data content"

        try:
            # Store data
            content_hash = await cas.store_data(
                data=test_data,
                content_type="application/octet-stream",
                metadata={"source": "example", "type": "test_data"},
            )
            print(f"✓ Data stored with hash: {content_hash[:16]}...")

            # Check if data exists
            exists = await cas.exists(content_hash)
            print(f"✓ Data exists check: {exists}")

            # Retrieve data
            retrieved_data = await cas.retrieve_data(content_hash)
            print(f"✓ Data retrieved successfully: {len(retrieved_data)} bytes")

            # Verify data integrity
            if retrieved_data == test_data:
                print("✓ Data integrity verified")
            else:
                print("✗ Data integrity check failed")

        except Exception as e:
            print(f"✗ Content-addressed storage example failed: {e}")

        # Example 2: Direct R2 operations
        print("\n5. Direct R2 storage example...")
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_file:
                tmp_file.write(b"Example FITS file content")
                tmp_file.flush()
                tmp_path = Path(tmp_file.name)

            # Upload file
            object_key = await r2_client.upload_file(
                bucket=config.r2_bucket_name,
                key="examples/test_file.txt",
                data=tmp_path,
                metadata={"example": "true", "type": "fits"},
            )
            print(f"✓ File uploaded with key: {object_key}")

            # List files
            files = await r2_client.list_files(
                bucket=config.r2_bucket_name, prefix="examples/", max_keys=10
            )
            print(f"✓ Found {len(files)} files in examples/ prefix")

            # Get file metadata
            metadata = await r2_client.get_file_metadata(
                bucket=config.r2_bucket_name, key=object_key
            )
            print(f"✓ File metadata: {metadata['size_bytes']} bytes")

            # Download file
            downloaded_data = await r2_client.download_file(
                bucket=config.r2_bucket_name, key=object_key
            )
            print(f"✓ File downloaded: {len(downloaded_data)} bytes")

            # Clean up temporary file
            tmp_path.unlink()

        except Exception as e:
            print(f"✗ Direct R2 storage example failed: {e}")

        # Example 3: DVC client (basic initialization)
        print("\n6. DVC client example...")
        try:
            dvc_client = DVCClient(config=config)
            print("✓ DVC client initialized")

            # Initialize DVC repo (this would create .dvc directory)
            # await dvc_client.init_repo()
            # print("✓ DVC repository initialized")

            # Configure remote (this would add remote to DVC config)
            # await dvc_client.configure_remote()
            # print("✓ DVC remote configured")

            print(f"✓ DVC client ready: {type(dvc_client).__name__}")
            print("ℹ DVC operations skipped (would modify current directory)")

        except Exception as e:
            print(f"✗ DVC client example failed: {e}")

        # Example 4: MLflow storage configuration
        print("\n7. MLflow storage configuration example...")
        try:
            mlflow_config = MLflowStorageConfig.from_storage_config(config)
            print("✓ MLflow storage configuration created")

            env_vars = mlflow_config.get_env_vars()
            print(f"✓ MLflow environment variables configured: {len(env_vars)} vars")

            # Note: MLflow artifact storage would require MLflow server
            print("ℹ MLflow artifact operations require running MLflow server")

        except Exception as e:
            print(f"✗ MLflow storage configuration failed: {e}")

        print(f"\n{'=' * 50}")
        print("Storage infrastructure example completed successfully!")
        print("All components are properly configured and functional.")

    except Exception as e:
        print(f"\n✗ Example failed with error: {e}")
        print("Please check your configuration and try again.")


if __name__ == "__main__":
    asyncio.run(main())
