"""Tests for ASTR-92 Dramatiq Workers implementation."""

from datetime import datetime
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

from src.adapters.workers.config import (
    WorkerConfig,
    WorkerManager,
    WorkerType,
    get_task_queues,
    get_worker_config,
)
from src.adapters.workers.curation.curation_workers import CurationWorker
from src.adapters.workers.detection.detection_workers import DetectionWorker
from src.adapters.workers.differencing.differencing_workers import DifferencingWorker
from src.adapters.workers.ingestion.observation_workers import (
    ObservationIngestionWorker,
)
from src.adapters.workers.monitoring import WorkerMonitor
from src.adapters.workers.preprocessing.preprocessing_workers import PreprocessingWorker


@pytest.fixture
def worker_config():
    """Create worker configuration for testing."""
    return WorkerConfig(
        broker_url="redis://localhost:6379/0",
        result_backend="redis://localhost:6379/1",
        max_retries=3,
        retry_delay=1000,
        worker_timeout=300,
        max_memory=1024,
        max_cpu=80,
        concurrency=4,
        prefetch_multiplier=2,
    )


@pytest.fixture
def task_queues():
    """Create task queues for testing."""
    return get_task_queues()


@pytest.fixture
def worker_manager_instance():
    """Create worker manager instance for testing."""
    return WorkerManager()


@pytest.fixture
def worker_monitor_instance():
    """Create worker monitor instance for testing."""
    return WorkerMonitor()


@pytest.fixture
def observation_ingestion_worker():
    """Create observation ingestion worker for testing."""
    return ObservationIngestionWorker()


@pytest.fixture
def preprocessing_worker():
    """Create preprocessing worker for testing."""
    return PreprocessingWorker()


@pytest.fixture
def differencing_worker():
    """Create differencing worker for testing."""
    return DifferencingWorker()


@pytest.fixture
def detection_worker():
    """Create detection worker for testing."""
    return DetectionWorker()


@pytest.fixture
def curation_worker():
    """Create curation worker for testing."""
    return CurationWorker()


@pytest.fixture
def test_observation_data():
    """Create test observation data."""
    return {
        "survey_id": str(uuid4()),
        "observation_id": "TEST_OBS_001",
        "ra": 180.0,
        "dec": 45.0,
        "observation_time": datetime.now().isoformat(),
        "filter_band": "g",
        "exposure_time": 300.0,
        "fits_url": "https://example.com/test.fits",
        "pixel_scale": 0.5,
        "airmass": 1.2,
        "seeing": 1.0,
    }


class TestWorkerConfig:
    """Test worker configuration."""

    def test_worker_config_creation(self, worker_config):
        """Test creating worker configuration."""
        assert worker_config.broker_url == "redis://localhost:6379/0"
        assert worker_config.result_backend == "redis://localhost:6379/1"
        assert worker_config.max_retries == 3
        assert worker_config.retry_delay == 1000
        assert worker_config.worker_timeout == 300
        assert worker_config.max_memory == 1024
        assert worker_config.max_cpu == 80
        assert worker_config.concurrency == 4
        assert worker_config.prefetch_multiplier == 2

    def test_get_worker_config(self):
        """Test getting worker configuration from environment."""
        config = get_worker_config()
        assert isinstance(config, WorkerConfig)
        assert config.broker_url is not None
        assert config.result_backend is not None

    def test_get_task_queues(self, task_queues):
        """Test getting task queues configuration."""
        assert len(task_queues) == 6
        queue_names = [q.queue_name for q in task_queues]
        expected_queues = [
            "observation_ingestion",
            "preprocessing",
            "differencing",
            "detection",
            "curation",
            "notification",
        ]
        for expected_queue in expected_queues:
            assert expected_queue in queue_names

    def test_worker_type_enum(self):
        """Test worker type enumeration."""
        assert WorkerType.OBSERVATION_INGESTION.value == "observation_ingestion"
        assert WorkerType.PREPROCESSING.value == "preprocessing"
        assert WorkerType.DIFFERENCING.value == "differencing"
        assert WorkerType.DETECTION.value == "detection"
        assert WorkerType.CURATION.value == "curation"
        assert WorkerType.NOTIFICATION.value == "notification"


class TestWorkerManager:
    """Test worker manager functionality."""

    def test_worker_manager_creation(self, worker_manager_instance):
        """Test creating worker manager."""
        assert worker_manager_instance.broker is None
        assert worker_manager_instance.result_backend is None
        assert worker_manager_instance.actors == {}

    def test_setup_broker(self, worker_manager_instance, worker_config):
        """Test setting up broker."""
        with patch("src.adapters.workers.config.RedisBroker") as mock_broker:
            with patch("src.adapters.workers.config.RedisBackend") as mock_backend:
                with patch("src.adapters.workers.config.Results") as mock_results:
                    with patch("src.adapters.workers.config.dramatiq") as mock_dramatiq:
                        worker_manager_instance.setup_broker(worker_config)

                        mock_broker.assert_called_once_with(
                            url=worker_config.broker_url
                        )
                        mock_backend.assert_called_once_with(
                            url=worker_config.result_backend
                        )
                        mock_results.assert_called_once()
                        mock_dramatiq.set_broker.assert_called_once()

    def test_get_worker_metrics(self, worker_manager_instance):
        """Test getting worker metrics."""
        worker_id = "test_worker_1"
        metrics = worker_manager_instance.get_worker_metrics(worker_id)

        assert metrics is not None
        assert metrics.worker_id == worker_id
        assert metrics.worker_type == WorkerType.OBSERVATION_INGESTION
        assert metrics.status == "IDLE"
        assert metrics.tasks_processed == 0
        assert metrics.tasks_failed == 0

    def test_get_queue_status(self, worker_manager_instance):
        """Test getting queue status."""
        status = worker_manager_instance.get_queue_status()

        assert "queues" in status
        assert "total_actors" in status
        assert "broker_connected" in status
        assert isinstance(status["queues"], list)
        assert isinstance(status["total_actors"], int)
        assert isinstance(status["broker_connected"], bool)


class TestWorkerMonitor:
    """Test worker monitoring functionality."""

    def test_worker_monitor_creation(self, worker_monitor_instance):
        """Test creating worker monitor."""
        assert worker_monitor_instance.metrics_history == {}
        assert worker_monitor_instance.start_time is not None

    def test_get_worker_metrics(self, worker_monitor_instance):
        """Test getting worker metrics."""
        worker_id = "test_worker_1"
        metrics = worker_monitor_instance.get_worker_metrics(worker_id)

        assert metrics is not None
        assert metrics.worker_id == worker_id
        assert metrics.worker_type == WorkerType.OBSERVATION_INGESTION
        assert metrics.status == "IDLE"
        assert metrics.memory_usage >= 0
        assert metrics.cpu_usage >= 0

    def test_get_worker_health(self, worker_monitor_instance):
        """Test getting worker health."""
        health = worker_monitor_instance.get_worker_health()

        assert "status" in health
        assert "total_workers" in health
        assert "healthy_workers" in health
        assert "health_ratio" in health
        assert "timestamp" in health
        assert health["status"] in ["healthy", "degraded", "unhealthy", "error"]

    def test_get_queue_status(self, worker_monitor_instance):
        """Test getting queue status."""
        status = worker_monitor_instance.get_queue_status()

        assert "queues" in status
        assert "total_actors" in status
        assert "broker_connected" in status
        assert "timestamp" in status

    def test_get_performance_metrics(self, worker_monitor_instance):
        """Test getting performance metrics."""
        metrics = worker_monitor_instance.get_performance_metrics(time_window_hours=1)

        assert "time_window_hours" in metrics
        assert "total_tasks_processed" in metrics
        assert "total_tasks_failed" in metrics
        assert "failure_rate" in metrics
        assert "timestamp" in metrics
        assert metrics["time_window_hours"] == 1

    def test_clear_old_metrics(self, worker_monitor_instance):
        """Test clearing old metrics."""
        # Add some test metrics
        worker_monitor_instance.metrics_history["worker1"] = [
            Mock(last_heartbeat="2020-01-01T00:00:00"),
            Mock(last_heartbeat="2020-01-02T00:00:00"),
        ]

        cleared_count = worker_monitor_instance.clear_old_metrics(days_to_keep=1)
        assert cleared_count >= 0


class TestObservationIngestionWorker:
    """Test observation ingestion worker."""

    @pytest.mark.asyncio
    async def test_validate_observation_data(
        self, observation_ingestion_worker, test_observation_data
    ):
        """Test observation data validation."""
        result = await observation_ingestion_worker.validate_observation_data(
            test_observation_data
        )

        assert result["valid"] is True
        assert len(result["errors"]) == 0

    @pytest.mark.asyncio
    async def test_validate_observation_data_invalid(
        self, observation_ingestion_worker
    ):
        """Test observation data validation with invalid data."""
        invalid_data = {
            "survey_id": "invalid-uuid",
            "observation_id": "TEST",
            "ra": 400.0,  # Invalid RA
            "dec": 100.0,  # Invalid Dec
            "observation_time": "invalid-datetime",
        }

        result = await observation_ingestion_worker.validate_observation_data(
            invalid_data
        )

        assert result["valid"] is False
        assert len(result["errors"]) > 0

    @pytest.mark.asyncio
    async def test_process_observation_metadata(
        self, observation_ingestion_worker, test_observation_data
    ):
        """Test observation metadata processing."""
        result = await observation_ingestion_worker.process_observation_metadata(
            test_observation_data
        )

        assert "original_data" in result
        assert "processed_at" in result
        assert "processing_version" in result
        assert "quality_indicators" in result

    @pytest.mark.asyncio
    async def test_store_observation_files(
        self, observation_ingestion_worker, test_observation_data
    ):
        """Test observation file storage."""
        result = await observation_ingestion_worker.store_observation_files(
            test_observation_data
        )

        assert "fits_file_path" in result
        assert "thumbnail_path" in result
        assert "storage_metadata" in result


class TestPreprocessingWorker:
    """Test preprocessing worker."""

    @pytest.mark.asyncio
    async def test_apply_calibration(self, preprocessing_worker):
        """Test calibration application."""
        observation_id = uuid4()
        calibration_frames = {
            "bias_frame": "bias.fits",
            "dark_frame": "dark.fits",
            "flat_frame": "flat.fits",
        }

        result = await preprocessing_worker.apply_calibration(
            observation_id, calibration_frames
        )

        assert result["observation_id"] == str(observation_id)
        assert result["calibration_applied"] is True
        assert "bias_correction" in result
        assert "dark_correction" in result
        assert "flat_correction" in result

    @pytest.mark.asyncio
    async def test_align_observation(self, preprocessing_worker):
        """Test observation alignment."""
        observation_id = uuid4()
        reference_id = uuid4()

        result = await preprocessing_worker.align_observation(
            observation_id, reference_id
        )

        assert result["observation_id"] == str(observation_id)
        assert result["reference_id"] == str(reference_id)
        assert result["alignment_applied"] is True
        assert "wcs_correction" in result
        assert "registration_quality" in result

    @pytest.mark.asyncio
    async def test_assess_quality(self, preprocessing_worker):
        """Test quality assessment."""
        observation_id = uuid4()

        result = await preprocessing_worker.assess_quality(observation_id)

        assert result["observation_id"] == str(observation_id)
        assert "overall_quality_score" in result
        assert "background_analysis" in result
        assert "noise_analysis" in result
        assert "cosmic_ray_analysis" in result


class TestDifferencingWorker:
    """Test differencing worker."""

    @pytest.mark.asyncio
    async def test_apply_differencing_algorithm(self, differencing_worker):
        """Test differencing algorithm application."""
        observation_id = uuid4()
        algorithm = "zogy"

        result = await differencing_worker.apply_differencing_algorithm(
            observation_id, algorithm
        )

        assert result["observation_id"] == str(observation_id)
        assert result["algorithm"] == algorithm
        assert "difference_id" in result
        assert "difference_image_path" in result
        assert "algorithm_parameters" in result

    @pytest.mark.asyncio
    async def test_validate_difference_image(self, differencing_worker):
        """Test difference image validation."""
        difference_id = "test_diff_001"

        result = await differencing_worker.validate_difference_image(difference_id)

        assert result["difference_id"] == difference_id
        assert result["validation_passed"] is True
        assert "quality_metrics" in result
        assert "statistical_metrics" in result
        assert "spatial_metrics" in result

    @pytest.mark.asyncio
    async def test_extract_sources(self, differencing_worker):
        """Test source extraction."""
        difference_id = "test_diff_001"

        result = await differencing_worker.extract_sources(difference_id)

        assert result["difference_id"] == difference_id
        assert "source_count" in result
        assert "candidates" in result
        assert "extraction_parameters" in result
        assert "quality_metrics" in result


class TestDetectionWorker:
    """Test detection worker."""

    @pytest.mark.asyncio
    async def test_run_ml_inference(self, detection_worker):
        """Test ML model inference."""
        difference_id = "test_diff_001"
        model_id = "unet_v1.0.0"

        result = await detection_worker.run_ml_inference(difference_id, model_id)

        assert result["difference_id"] == difference_id
        assert result["model_id"] == model_id
        assert "detection_id" in result
        assert "detection_results" in result
        assert "model_metrics" in result

    @pytest.mark.asyncio
    async def test_validate_detections(self, detection_worker):
        """Test detection validation."""
        detection_id = "test_det_001"

        result = await detection_worker.validate_detections(detection_id)

        assert result["detection_id"] == detection_id
        assert result["validation_passed"] is True
        assert "quality_checks" in result
        assert "consistency_checks" in result
        assert "filtering_results" in result

    @pytest.mark.asyncio
    async def test_calculate_detection_metrics(self, detection_worker):
        """Test detection metrics calculation."""
        detection_id = "test_det_001"

        result = await detection_worker.calculate_detection_metrics(detection_id)

        assert result["detection_id"] == detection_id
        assert "performance_metrics" in result
        assert "processing_metrics" in result
        assert "resource_metrics" in result
        assert "detection_statistics" in result


class TestCurationWorker:
    """Test curation worker."""

    @pytest.mark.asyncio
    async def test_create_validation_events(self, curation_worker):
        """Test validation event creation."""
        detection_id = "test_det_001"

        result = await curation_worker.create_validation_events(detection_id)

        assert result["detection_id"] == detection_id
        assert "validation_events_created" in result
        assert "events" in result
        assert len(result["events"]) > 0

    @pytest.mark.asyncio
    async def test_generate_alerts(self, curation_worker):
        """Test alert generation."""
        detection_id = "test_det_001"

        result = await curation_worker.generate_alerts(detection_id)

        assert result["detection_id"] == detection_id
        assert "alerts_generated" in result
        assert "alerts" in result

    @pytest.mark.asyncio
    async def test_prepare_curation_interface(self, curation_worker):
        """Test curation interface preparation."""
        detection_id = "test_det_001"

        result = await curation_worker.prepare_curation_interface(detection_id)

        assert result["detection_id"] == detection_id
        assert result["interface_data_prepared"] is True
        assert "data" in result
        assert "detection_images" in result["data"]
        assert "context_images" in result["data"]
        assert "metadata" in result["data"]

    @pytest.mark.asyncio
    async def test_send_notifications(self, curation_worker):
        """Test notification sending."""
        detection_id = "test_det_001"

        result = await curation_worker.send_notifications(detection_id)

        assert result["detection_id"] == detection_id
        assert "notifications_sent" in result
        assert "channels" in result


class TestWorkerIntegration:
    """Test worker integration and workflow."""

    @pytest.mark.asyncio
    async def test_worker_workflow_integration(self, test_observation_data):
        """Test complete worker workflow integration."""
        # This would test the complete workflow from ingestion to curation
        # In a real test, you would mock the database and external services

        observation_worker = ObservationIngestionWorker()
        preprocessing_worker = PreprocessingWorker()
        differencing_worker = DifferencingWorker()
        detection_worker = DetectionWorker()
        curation_worker = CurationWorker()

        # Test data validation
        validation_result = await observation_worker.validate_observation_data(
            test_observation_data
        )
        assert validation_result["valid"] is True

        # Test metadata processing
        metadata_result = await observation_worker.process_observation_metadata(
            test_observation_data
        )
        assert "quality_indicators" in metadata_result

        # Test file storage
        storage_result = await observation_worker.store_observation_files(
            test_observation_data
        )
        assert "fits_file_path" in storage_result

        # Test preprocessing
        observation_id = uuid4()
        calibration_result = await preprocessing_worker.apply_calibration(
            observation_id, {}
        )
        assert calibration_result["calibration_applied"] is True

        # Test differencing
        _reference_id = uuid4()
        algorithm_result = await differencing_worker.apply_differencing_algorithm(
            observation_id, "zogy"
        )
        assert algorithm_result["algorithm"] == "zogy"

        # Test detection
        difference_id = algorithm_result["difference_id"]
        inference_result = await detection_worker.run_ml_inference(
            difference_id, "unet_v1"
        )
        assert inference_result["model_id"] == "unet_v1"

        # Test curation
        detection_id = inference_result["detection_id"]
        validation_events = await curation_worker.create_validation_events(detection_id)
        assert validation_events["validation_events_created"] > 0


class TestWorkerErrorHandling:
    """Test worker error handling."""

    @pytest.mark.asyncio
    async def test_worker_error_handling(self, observation_ingestion_worker):
        """Test worker error handling with invalid data."""
        invalid_data = {
            "survey_id": "invalid",
            "observation_id": "",
            "ra": -1000.0,
            "dec": 200.0,
            "observation_time": "invalid",
        }

        result = await observation_ingestion_worker.validate_observation_data(
            invalid_data
        )

        assert result["valid"] is False
        assert len(result["errors"]) > 0
        assert any(
            "RA must be between 0 and 360 degrees" in error
            for error in result["errors"]
        )
        assert any(
            "Dec must be between -90 and 90 degrees" in error
            for error in result["errors"]
        )

    @pytest.mark.asyncio
    async def test_worker_exception_handling(self, preprocessing_worker):
        """Test worker exception handling."""
        # Test with invalid observation ID
        invalid_observation_id = uuid4()

        # This should not raise an exception, but return a result
        result = await preprocessing_worker.assess_quality(invalid_observation_id)

        assert result["observation_id"] == str(invalid_observation_id)
        assert "overall_quality_score" in result


if __name__ == "__main__":
    pytest.main([__file__])
