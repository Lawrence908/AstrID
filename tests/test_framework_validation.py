"""Test framework validation - ensures all components work together."""

import asyncio
from typing import Any

import pytest

from tests.mocks import MockMLflowClient, MockPrefectClient, MockStorageClient
from tests.utils import APITestUtils, TestTimer, ValidationTestUtils


class TestFrameworkValidation:
    """Validate that the test framework components work correctly."""

    def test_timer_utility(self):
        """Test the timer utility."""
        timer = TestTimer()
        timer.start()
        # Simulate some work
        import time

        time.sleep(0.1)
        timer.stop()

        assert timer.elapsed > 0.05
        assert timer.elapsed < 0.5

    def test_timer_context_manager(self):
        """Test timer as context manager."""
        import time

        with TestTimer() as timer:
            time.sleep(0.1)

        assert timer.elapsed > 0.05
        assert timer.elapsed < 0.5

    def test_validation_utils(self):
        """Test validation utilities."""
        # Test UUID validation
        ValidationTestUtils.assert_valid_uuid("123e4567-e89b-12d3-a456-426614174000")

        with pytest.raises(AssertionError):
            ValidationTestUtils.assert_valid_uuid("invalid-uuid")

        # Test coordinate validation
        ValidationTestUtils.assert_valid_coordinates(180.0, 45.0)

        with pytest.raises(AssertionError):
            ValidationTestUtils.assert_valid_coordinates(400.0, 45.0)  # Invalid RA

        with pytest.raises(AssertionError):
            ValidationTestUtils.assert_valid_coordinates(180.0, 100.0)  # Invalid Dec

    def test_api_utils(self):
        """Test API utilities."""
        response_data = {
            "data": [{"id": 1}, {"id": 2}],
            "total": 2,
            "page": 1,
            "per_page": 10,
            "pages": 1,
        }

        # Should not raise
        APITestUtils.assert_pagination_response(response_data)

        # Missing field should raise
        incomplete_data = {"data": []}
        with pytest.raises(AssertionError):
            APITestUtils.assert_pagination_response(incomplete_data)

    def test_mock_storage_client(self):
        """Test mock storage client."""
        client = MockStorageClient()

        # Test basic operations
        assert len(client.files) == 0

        # Upload
        result = asyncio.run(client.upload_file("local/path.txt", "remote/path.txt"))
        assert result == "mock://storage/remote/path.txt"
        assert "remote/path.txt" in client.files

        # Download
        content = asyncio.run(client.download_file("remote/path.txt"))
        assert content == b"mock_file_content_local/path.txt"

        # File exists
        exists = asyncio.run(client.file_exists("remote/path.txt"))
        assert exists is True

        # List files
        files = asyncio.run(client.list_files())
        assert "remote/path.txt" in files

        # Delete
        deleted = asyncio.run(client.delete_file("remote/path.txt"))
        assert deleted is True
        assert "remote/path.txt" not in client.files

    def test_mock_mlflow_client(self):
        """Test mock MLflow client."""
        client = MockMLflowClient()

        # Create experiment
        exp_id = client.create_experiment("test_experiment")
        assert exp_id in client.experiments

        # Create run
        run = client.create_run(exp_id, "test_run")
        assert run.info.run_id in client.runs

        # Log parameters and metrics
        client.log_param("learning_rate", 0.01)
        client.log_metric("accuracy", 0.95)

        # Verify data was logged
        run_data = client.runs[run.info.run_id]
        assert "learning_rate" in run_data["params"]
        assert "accuracy" in run_data["metrics"]

        # End run
        client.end_run()
        assert client.current_run_id is None

    @pytest.mark.asyncio
    async def test_mock_prefect_client(self):
        """Test mock Prefect client."""
        client = MockPrefectClient()

        # Create flow
        flow = await client.create_flow("test_flow")
        assert flow["id"] in client.flows

        # Create flow run
        flow_run = await client.create_flow_run(flow_id=flow["id"], name="test_run")
        assert flow_run["id"] in client.flow_runs

        # Update state
        updated_run = await client.set_flow_run_state(flow_run["id"], "RUNNING")
        assert updated_run["state"]["type"] == "RUNNING"

        # Complete run
        completed_run = await client.set_flow_run_state(flow_run["id"], "COMPLETED")
        assert completed_run["state"]["type"] == "COMPLETED"

    def test_sample_data_fixtures(
        self,
        sample_observation_data: dict[str, Any],
        sample_survey_data: dict[str, Any],
        sample_detection_data: dict[str, Any],
    ):
        """Test that sample data fixtures work correctly."""
        # Observation data
        assert "id" in sample_observation_data
        assert "ra" in sample_observation_data
        assert "dec" in sample_observation_data
        assert isinstance(sample_observation_data["ra"], float)
        assert isinstance(sample_observation_data["dec"], float)

        # Survey data
        assert "id" in sample_survey_data
        assert "name" in sample_survey_data
        assert "filters" in sample_survey_data
        assert isinstance(sample_survey_data["filters"], list)

        # Detection data
        assert "id" in sample_detection_data
        assert "x" in sample_detection_data
        assert "y" in sample_detection_data
        assert isinstance(sample_detection_data["confidence"], float)

    @pytest.mark.asyncio
    async def test_database_fixtures(self, db_session):
        """Test database fixtures work correctly."""
        # This test verifies that the database session fixture works
        # In a real test, you would use the session to create/query data
        assert db_session is not None

        # Test that we can execute a simple query
        result = await db_session.execute("SELECT 1 as test_value")
        row = result.fetchone()
        assert row[0] == 1

    def test_mock_services_integration(
        self, mock_mlflow_client, mock_prefect_client, mock_storage_service
    ):
        """Test that mock services can be used together."""
        # This demonstrates how multiple mock services can be used in a single test
        assert mock_mlflow_client is not None
        assert mock_prefect_client is not None
        assert mock_storage_service is not None

        # Verify they have the expected interface
        assert hasattr(mock_mlflow_client, "create_experiment")
        assert hasattr(mock_prefect_client, "create_flow_run")
        assert hasattr(mock_storage_service, "upload")

    def test_error_simulation(self):
        """Test error simulation capabilities."""
        client = MockStorageClient()

        # Enable error simulation
        client.simulate_upload_error(True)

        # Upload should now fail
        with pytest.raises(Exception, match="Simulated upload error"):
            asyncio.run(client.upload_file("test.txt", "remote.txt"))

        # Disable error simulation
        client.simulate_upload_error(False)

        # Upload should work again
        result = asyncio.run(client.upload_file("test.txt", "remote.txt"))
        assert result == "mock://storage/remote.txt"

    def test_performance_timing(self, performance_timer):
        """Test performance timing fixture."""
        import time

        performance_timer.start()
        time.sleep(0.05)  # 50ms
        performance_timer.stop()

        assert performance_timer.elapsed > 0.04
        assert performance_timer.elapsed < 0.1
