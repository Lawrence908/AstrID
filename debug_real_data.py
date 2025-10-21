#!/usr/bin/env python3
"""
Debug script to check real data availability
"""

import asyncio
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def check_real_data():
    """Check if real data is available in the database."""
    print("🔍 Checking real data availability...")

    try:
        # Import required modules
        from src.core.db.session import AsyncSessionLocal
        from src.domains.ml.training_data.services import (
            TrainingDataCollectionParams,
            TrainingDataCollector,
        )
        from src.infrastructure.storage.r2_client import R2StorageClient

        # Create database session
        db_session = AsyncSessionLocal()
        r2_client = R2StorageClient()

        try:
            # Set up collection parameters
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)

            collection_params = TrainingDataCollectionParams(
                survey_ids=["hst", "jwst", "skyview"],
                date_range=(start_date, end_date),
                confidence_threshold=0.6,
                max_samples=100,
                validation_status="validated",
            )

            print("📊 Checking for data with params:")
            print(f"   Date range: {start_date.date()} to {end_date.date()}")
            print(f"   Confidence threshold: {collection_params.confidence_threshold}")
            print(f"   Max samples: {collection_params.max_samples}")

            # Try to collect data
            collector = TrainingDataCollector(db_session, r2_client)
            samples = await collector.collect_training_data(collection_params)

            if samples:
                print(f"✅ Found {len(samples)} training samples!")
                print("   Real data is available for training")

                # Show sample details
                for i, sample in enumerate(samples[:3]):  # Show first 3
                    print(f"   Sample {i+1}: {sample.labels}")

                return True
            else:
                print("❌ No training samples found")
                print("   Reasons could be:")
                print("   • No observations in database")
                print("   • No detections with sufficient confidence")
                print("   • No validated detections available")
                return False

        finally:
            await db_session.close()

    except Exception as e:
        print(f"❌ Error checking real data: {e}")
        logger.error(f"Real data check failed: {e}", exc_info=True)
        return False


async def check_database_content():
    """Check what's actually in the database."""
    print("\n🗄️ Checking database content...")

    try:
        from sqlalchemy import func, select

        from src.core.db.session import AsyncSessionLocal
        from src.domains.detection.models import Detection
        from src.domains.observations.models import Observation, Survey

        db_session = AsyncSessionLocal()

        try:
            # Check observations
            obs_count = await db_session.execute(select(func.count(Observation.id)))
            obs_total = obs_count.scalar()
            print(f"📊 Observations in database: {obs_total}")

            # Check detections
            det_count = await db_session.execute(select(func.count(Detection.id)))
            det_total = det_count.scalar()
            print(f"🎯 Detections in database: {det_total}")

            # Check validated detections
            val_det_count = await db_session.execute(
                select(func.count(Detection.id)).where(Detection.is_validated)
            )
            val_det_total = val_det_count.scalar()
            print(f"✅ Validated detections: {val_det_total}")

            # Check surveys
            survey_count = await db_session.execute(select(func.count(Survey.id)))
            survey_total = survey_count.scalar()
            print(f"🔭 Surveys in database: {survey_total}")

            if obs_total == 0:
                print("\n💡 Database is empty - this explains the real data fallback")
                print("   To enable real data training:")
                print("   1. Add observations using the ingestion service")
                print("   2. Run detection pipeline to create detections")
                print("   3. Validate detections through curation process")

        finally:
            await db_session.close()

    except Exception as e:
        print(f"❌ Error checking database: {e}")
        logger.error(f"Database check failed: {e}", exc_info=True)


async def main():
    print("🚀 AstrID Real Data Debug")
    print("=" * 40)

    data_available = await check_real_data()
    await check_database_content()

    print("\n🎯 Summary:")
    print(f"   Real data available: {'✅' if data_available else '❌'}")
    print(
        f"   Training mode: {'Real data' if data_available else 'Synthetic fallback'}"
    )


if __name__ == "__main__":
    asyncio.run(main())
