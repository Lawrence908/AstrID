"""Service layer for observations domain."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.logging import configure_domain_logger
from src.domains.observations.crud import ObservationCRUD, SurveyCRUD
from src.domains.observations.events import (
    ObservationFailed,
    ObservationStatusChanged,
)
from src.domains.observations.ingestion.services import (
    DataIngestionService,
    ObservationBuilderService,
)
from src.domains.observations.schema import (
    ObservationCreate,
    ObservationListParams,
    ObservationRead,
    ObservationStatus,
    ObservationUpdate,
    SurveyCreate,
    SurveyListParams,
    SurveyRead,
    SurveyUpdate,
)
from src.domains.observations.validators import (
    ObservationValidationError,
    ObservationValidator,
)


class SurveyService:
    """Service layer for Survey operations."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.crud = SurveyCRUD()
        self.logger = configure_domain_logger("observations.survey")

    async def create_survey(self, survey_data: SurveyCreate) -> SurveyRead:
        """Create a new survey."""
        self.logger.info(
            f"Creating survey: name={survey_data.name}, description={survey_data.description}"
        )
        try:
            survey = await self.crud.create(self.db, survey_data)
            result: SurveyRead = SurveyRead.model_validate(survey)
            self.logger.info(
                f"Successfully created survey: id={result.id}, name={result.name}"
            )
            return result
        except Exception as e:
            self.logger.error(
                f"Failed to create survey: name={survey_data.name}, error={str(e)}"
            )
            raise

    async def get_survey_by_id(self, survey_id: UUID) -> SurveyRead | None:
        """Get survey by ID."""
        self.logger.debug(f"Retrieving survey by ID: {survey_id}")
        survey = await self.crud.get_by_id(self.db, survey_id)
        if not survey:
            self.logger.warning(f"Survey not found: id={survey_id}")
            return None
        result: SurveyRead = SurveyRead.model_validate(survey)
        self.logger.debug(f"Found survey: id={result.id}, name={result.name}")
        return result

    async def get_survey_by_name(self, name: str) -> SurveyRead | None:
        """Get survey by name."""
        self.logger.debug(f"Retrieving survey by name: {name}")
        survey = await self.crud.get_by_name(self.db, name)
        if not survey:
            self.logger.warning(f"Survey not found: name={name}")
            return None
        result: SurveyRead = SurveyRead.model_validate(survey)
        self.logger.debug(f"Found survey: id={result.id}, name={result.name}")
        return result

    async def list_surveys(
        self, params: SurveyListParams
    ) -> tuple[list[SurveyRead], int]:
        """List surveys with pagination."""
        self.logger.debug(
            f"Listing surveys: limit={params.limit}, offset={params.offset}, is_active={params.is_active}"
        )
        surveys, total = await self.crud.get_many(self.db, params)
        result = [SurveyRead.model_validate(survey) for survey in surveys]
        self.logger.debug(f"Retrieved {len(result)} surveys (total: {total})")
        return result, total

    async def update_survey(
        self, survey_id: UUID, survey_data: SurveyUpdate
    ) -> SurveyRead | None:
        """Update a survey."""
        self.logger.info(f"Updating survey: id={survey_id}")
        try:
            survey = await self.crud.update(self.db, survey_id, survey_data)
            if not survey:
                self.logger.warning(f"Survey not found for update: id={survey_id}")
                return None
            result: SurveyRead = SurveyRead.model_validate(survey)
            self.logger.info(
                f"Successfully updated survey: id={result.id}, name={result.name}"
            )
            return result
        except Exception as e:
            self.logger.error(
                f"Failed to update survey: id={survey_id}, error={str(e)}"
            )
            raise

    async def delete_survey(self, survey_id: UUID) -> bool:
        """Delete a survey."""
        self.logger.warning(f"Deleting survey: id={survey_id}")
        try:
            result = await self.crud.delete(self.db, survey_id)
            if result:
                self.logger.warning(f"Successfully deleted survey: id={survey_id}")
            else:
                self.logger.warning(f"Survey not found for deletion: id={survey_id}")
            return result
        except Exception as e:
            self.logger.error(
                f"Failed to delete survey: id={survey_id}, error={str(e)}"
            )
            raise

    async def activate_survey(self, survey_id: UUID) -> SurveyRead | None:
        """Activate a survey."""
        survey_data = SurveyUpdate(is_active=True)
        return await self.update_survey(survey_id, survey_data)

    async def deactivate_survey(self, survey_id: UUID) -> SurveyRead | None:
        """Deactivate a survey."""
        survey_data = SurveyUpdate(is_active=False)
        return await self.update_survey(survey_id, survey_data)


class ObservationService:
    """Service layer for Observation operations."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.crud = ObservationCRUD()
        self.logger = configure_domain_logger("observations.observation")

    async def create_observation(
        self, observation_data: ObservationCreate
    ) -> ObservationRead:
        """Create a new observation."""
        self.logger.info(
            f"Creating observation: survey_id={observation_data.survey_id}, observation_id={observation_data.observation_id}"
        )
        try:
            # Validate that the survey exists
            survey = await SurveyCRUD().get_by_id(self.db, observation_data.survey_id)
            if not survey:
                self.logger.error(f"Survey not found: id={observation_data.survey_id}")
                raise ValueError(
                    f"Survey with ID {observation_data.survey_id} not found"
                )

            # Check for duplicate observation
            existing = await self.crud.get_by_survey_and_observation_id(
                self.db, observation_data.survey_id, observation_data.observation_id
            )
            if existing:
                self.logger.error(
                    f"Duplicate observation: survey_id={observation_data.survey_id}, observation_id={observation_data.observation_id}"
                )
                raise ValueError(
                    f"Observation {observation_data.observation_id} already exists for survey {survey.name}"
                )

            observation = await self.crud.create(self.db, observation_data)
            result: ObservationRead = ObservationRead.model_validate(observation)
            self.logger.info(
                f"Successfully created observation: id={result.id}, survey_id={result.survey_id}, observation_id={result.observation_id}"
            )
            return result
        except Exception as e:
            self.logger.error(
                f"Failed to create observation: survey_id={observation_data.survey_id}, observation_id={observation_data.observation_id}, error={str(e)}"
            )
            raise

    async def get_observation_by_id(
        self, observation_id: UUID
    ) -> ObservationRead | None:
        """Get observation by ID."""
        self.logger.debug(f"Retrieving observation by ID: {observation_id}")
        observation = await self.crud.get_by_id(self.db, observation_id)
        if not observation:
            self.logger.warning(f"Observation not found: id={observation_id}")
            return None
        result: ObservationRead = ObservationRead.model_validate(observation)
        self.logger.debug(
            f"Found observation: id={result.id}, survey_id={result.survey_id}, observation_id={result.observation_id}"
        )
        return result

    async def get_observation_by_survey_and_id(
        self, survey_id: UUID, observation_id: str
    ) -> ObservationRead | None:
        """Get observation by survey ID and observation ID."""
        self.logger.debug(
            f"Retrieving observation: survey_id={survey_id}, observation_id={observation_id}"
        )
        observation = await self.crud.get_by_survey_and_observation_id(
            self.db, survey_id, observation_id
        )
        if not observation:
            self.logger.warning(
                f"Observation not found: survey_id={survey_id}, observation_id={observation_id}"
            )
            return None
        result: ObservationRead = ObservationRead.model_validate(observation)
        self.logger.debug(
            f"Found observation: id={result.id}, survey_id={result.survey_id}, observation_id={result.observation_id}"
        )
        return result

    async def list_observations(
        self, params: ObservationListParams
    ) -> tuple[list[ObservationRead], int]:
        """List observations with pagination and filtering."""
        self.logger.debug(
            f"Listing observations: limit={params.limit}, offset={params.offset}, status={params.status}, survey_id={params.survey_id}"
        )
        observations, total = await self.crud.get_many(self.db, params)
        result = [ObservationRead.model_validate(obs) for obs in observations]
        self.logger.debug(f"Retrieved {len(result)} observations (total: {total})")
        return result, total

    async def get_observations_in_region(
        self,
        ra_min: float,
        ra_max: float,
        dec_min: float,
        dec_max: float,
        limit: int = 100,
    ) -> list[ObservationRead]:
        """Get observations within a spatial region."""
        self.logger.debug(
            f"Getting observations in region: ra=[{ra_min}, {ra_max}], dec=[{dec_min}, {dec_max}], limit={limit}"
        )
        observations = await self.crud.get_by_spatial_region(
            self.db, ra_min, ra_max, dec_min, dec_max, limit
        )
        result = [ObservationRead.model_validate(obs) for obs in observations]
        self.logger.debug(f"Found {len(result)} observations in region")
        return result

    async def update_observation(
        self, observation_id: UUID, observation_data: ObservationUpdate
    ) -> ObservationRead | None:
        """Update an observation."""
        self.logger.info(f"Updating observation: id={observation_id}")
        try:
            observation = await self.crud.update(
                self.db, observation_id, observation_data
            )
            if not observation:
                self.logger.warning(
                    f"Observation not found for update: id={observation_id}"
                )
                return None
            result: ObservationRead = ObservationRead.model_validate(observation)
            self.logger.info(
                f"Successfully updated observation: id={result.id}, status={result.status}"
            )
            return result
        except Exception as e:
            self.logger.error(
                f"Failed to update observation: id={observation_id}, error={str(e)}"
            )
            raise

    async def delete_observation(self, observation_id: UUID) -> bool:
        """Delete an observation."""
        self.logger.warning(f"Deleting observation: id={observation_id}")
        try:
            result = await self.crud.delete(self.db, observation_id)
            if result:
                self.logger.warning(
                    f"Successfully deleted observation: id={observation_id}"
                )
            else:
                self.logger.warning(
                    f"Observation not found for deletion: id={observation_id}"
                )
            return result
        except Exception as e:
            self.logger.error(
                f"Failed to delete observation: id={observation_id}, error={str(e)}"
            )
            raise

    async def update_observation_status(
        self, observation_id: UUID, status: ObservationStatus
    ) -> ObservationRead | None:
        """Update observation status."""
        self.logger.info(
            f"Updating observation status: id={observation_id}, status={status.value}"
        )
        try:
            observation = await self.crud.update_status(
                self.db, observation_id, status.value
            )
            if not observation:
                self.logger.warning(
                    f"Observation not found for status update: id={observation_id}"
                )
                return None
            result: ObservationRead = ObservationRead.model_validate(observation)
            self.logger.info(
                f"Successfully updated observation status: id={result.id}, status={result.status}"
            )
            return result
        except Exception as e:
            self.logger.error(
                f"Failed to update observation status: id={observation_id}, status={status.value}, error={str(e)}"
            )
            raise

    async def mark_observation_as_ingested(
        self, observation_id: UUID
    ) -> ObservationRead | None:
        """Mark observation as ingested."""
        self.logger.info(f"Marking observation as ingested: id={observation_id}")
        return await self.update_observation_status(
            observation_id, ObservationStatus.INGESTED
        )

    async def mark_observation_as_preprocessing(
        self, observation_id: UUID
    ) -> ObservationRead | None:
        """Mark observation as preprocessing."""
        self.logger.info(f"Marking observation as preprocessing: id={observation_id}")
        return await self.update_observation_status(
            observation_id, ObservationStatus.PREPROCESSING
        )

    async def mark_observation_as_preprocessed(
        self, observation_id: UUID
    ) -> ObservationRead | None:
        """Mark observation as preprocessed."""
        self.logger.info(f"Marking observation as preprocessed: id={observation_id}")
        return await self.update_observation_status(
            observation_id, ObservationStatus.PREPROCESSED
        )

    async def mark_observation_as_differencing(
        self, observation_id: UUID
    ) -> ObservationRead | None:
        """Mark observation as differencing."""
        self.logger.info(f"Marking observation as differencing: id={observation_id}")
        return await self.update_observation_status(
            observation_id, ObservationStatus.DIFFERENCING
        )

    async def mark_observation_as_differenced(
        self, observation_id: UUID
    ) -> ObservationRead | None:
        """Mark observation as differenced."""
        self.logger.info(f"Marking observation as differenced: id={observation_id}")
        return await self.update_observation_status(
            observation_id, ObservationStatus.DIFFERENCED
        )

    async def mark_observation_as_failed(
        self, observation_id: UUID
    ) -> ObservationRead | None:
        """Mark observation as failed."""
        self.logger.error(f"Marking observation as failed: id={observation_id}")
        return await self.update_observation_status(
            observation_id, ObservationStatus.FAILED
        )

    async def mark_observation_as_archived(
        self, observation_id: UUID
    ) -> ObservationRead | None:
        """Mark observation as archived."""
        self.logger.info(f"Marking observation as archived: id={observation_id}")
        return await self.update_observation_status(
            observation_id, ObservationStatus.ARCHIVED
        )

    async def get_observations_by_status(
        self, status: ObservationStatus, limit: int = 100
    ) -> list[ObservationRead]:
        """Get observations by status."""
        self.logger.debug(
            f"Getting observations by status: status={status.value}, limit={limit}"
        )
        params = ObservationListParams(status=status, limit=limit)
        observations, _ = await self.list_observations(params)
        self.logger.debug(
            f"Found {len(observations)} observations with status {status.value}"
        )
        return observations

    async def get_observations_for_processing(
        self, limit: int = 100
    ) -> list[ObservationRead]:
        """Get observations that are ready for processing."""
        self.logger.debug(f"Getting observations for processing: limit={limit}")
        params = ObservationListParams(status=ObservationStatus.INGESTED, limit=limit)
        observations, _ = await self.list_observations(params)
        self.logger.debug(
            f"Found {len(observations)} observations ready for processing"
        )
        return observations

    # Ingestion methods

    async def ingest_observations_from_mast(
        self,
        survey_id: UUID,
        ra: float,
        dec: float,
        radius: float = 0.1,
        missions: list[str] | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[ObservationRead]:
        """Ingest observations from MAST for a specific sky position.

        Args:
            survey_id: Survey UUID to associate observations with
            ra: Right Ascension in degrees
            dec: Declination in degrees
            radius: Search radius in degrees
            missions: List of mission names to query
            start_time: Start of time range
            end_time: End of time range

        Returns:
            List of created observations
        """
        self.logger.info(
            f"Ingesting MAST observations for RA={ra:.4f}°, Dec={dec:.4f}°"
        )

        try:
            # Initialize ingestion services
            ingestion_service = DataIngestionService()
            builder_service = ObservationBuilderService()

            # Ingest observation data
            observation_creates = (
                await ingestion_service.ingest_observations_by_position(
                    ra=ra,
                    dec=dec,
                    radius=radius,
                    survey_id=survey_id,
                    missions=missions,
                    start_time=start_time,
                    end_time=end_time,
                )
            )

            # Create observation records
            observations = await builder_service.create_observations_from_ingestion(
                self.db, observation_creates
            )

            # Convert to read schemas
            result = [ObservationRead.model_validate(obs) for obs in observations]

            self.logger.info(f"Successfully ingested {len(result)} MAST observations")
            return result

        except Exception as e:
            self.logger.error(f"Failed to ingest MAST observations: {e}")
            raise

    async def create_reference_dataset(
        self,
        survey_id: UUID,
        ra: float,
        dec: float,
        size: float = 0.25,
        pixels: int = 512,
        surveys: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a complete reference dataset with image, catalog, and mask.

        Args:
            survey_id: Survey UUID
            ra: Right Ascension in degrees
            dec: Declination in degrees
            size: Image size in degrees
            pixels: Image size in pixels
            surveys: List of surveys to use

        Returns:
            Dictionary with dataset information
        """
        self.logger.info(f"Creating reference dataset for RA={ra:.4f}°, Dec={dec:.4f}°")

        try:
            ingestion_service = DataIngestionService()

            result = await ingestion_service.create_reference_dataset(
                ra=ra,
                dec=dec,
                size=size,
                pixels=pixels,
                surveys=surveys,
            )

            self.logger.info(
                f"Successfully created reference dataset: {result.get('r2_object_key', result.get('local_path'))}"
            )

            # Return the full result from ingestion service, which now includes R2 information
            return result

        except Exception as e:
            self.logger.error(f"Failed to create reference dataset: {e}")
            raise

    async def batch_ingest_random_observations(
        self,
        survey_id: UUID,
        count: int = 10,
        missions: list[str] | None = None,
        avoid_galactic_plane: bool = True,
    ) -> list[ObservationRead]:
        """Batch ingest observations from random sky positions.

        Args:
            survey_id: Survey UUID
            count: Number of random positions to query
            missions: List of missions to query
            avoid_galactic_plane: Whether to avoid galactic plane regions

        Returns:
            List of created observations
        """
        self.logger.info(f"Batch ingesting {count} random observations")

        try:
            ingestion_service = DataIngestionService()
            builder_service = ObservationBuilderService()

            # Batch ingest
            observation_creates = (
                await ingestion_service.batch_ingest_random_observations(
                    count=count,
                    survey_id=survey_id,
                    avoid_galactic_plane=avoid_galactic_plane,
                    missions=missions,
                )
            )

            # Create observation records
            observations = await builder_service.create_observations_from_ingestion(
                self.db, observation_creates
            )

            # Convert to read schemas
            result = [ObservationRead.model_validate(obs) for obs in observations]

            self.logger.info(f"Successfully batch ingested {len(result)} observations")
            return result

        except Exception as e:
            self.logger.error(f"Failed to batch ingest observations: {e}")
            raise

    async def ingest_from_fits_directory(
        self,
        survey_id: UUID,
        directory_path: str,
        file_pattern: str = "*.fits",
    ) -> list[ObservationRead]:
        """Ingest observations from FITS files in a directory.

        Args:
            survey_id: Survey UUID
            directory_path: Path to directory containing FITS files
            file_pattern: Glob pattern for files to process

        Returns:
            List of created observations
        """
        self.logger.info(f"Ingesting FITS files from directory: {directory_path}")

        try:
            builder_service = ObservationBuilderService()

            observations = await builder_service.create_observations_from_directory(
                db=self.db,
                survey_id=survey_id,
                directory_path=directory_path,
                file_pattern=file_pattern,
            )

            # Convert to read schemas
            result = [ObservationRead.model_validate(obs) for obs in observations]

            self.logger.info(
                f"Successfully ingested {len(result)} observations from directory"
            )
            return result

        except Exception as e:
            self.logger.error(f"Failed to ingest from directory: {e}")
            raise

    # Additional business logic methods

    async def validate_observation_data(self, data: ObservationCreate) -> bool:
        """Validate observation data using business rules.

        Args:
            data: Observation data to validate

        Returns:
            bool: True if validation passes

        Raises:
            ObservationValidationError: If validation fails
        """
        self.logger.debug(f"Validating observation data for {data.observation_id}")

        try:
            validator = ObservationValidator()

            # Convert to dict for validation
            data_dict = data.model_dump()
            validator.validate_observation_data(data_dict)

            self.logger.debug(
                f"Observation data validation passed for {data.observation_id}"
            )
            return True

        except ObservationValidationError as e:
            self.logger.error(
                f"Observation validation failed for {data.observation_id}: {e}"
            )
            raise

    async def calculate_observation_metrics(self, observation: ObservationRead) -> dict:
        """Calculate metrics and derived values for an observation.

        Args:
            observation: Observation to calculate metrics for

        Returns:
            dict: Calculated metrics
        """
        self.logger.debug(f"Calculating metrics for observation {observation.id}")

        try:
            # Get the full observation model for business logic methods
            obs_model = await self.crud.get_by_id(self.db, observation.id)
            if not obs_model:
                raise ValueError(f"Observation {observation.id} not found")

            metrics = {
                "observation_id": str(observation.id),
                "survey_id": str(observation.survey_id),
                "coordinates": {
                    "ra": observation.ra,
                    "dec": observation.dec,
                    "validation": obs_model.validate_coordinates(),
                },
                "calculated_airmass": obs_model.calculate_airmass(),
                "processing_status": obs_model.get_processing_status(),
                "sky_region": obs_model.get_sky_region_bounds(),
                "quality_metrics": {
                    "has_airmass": observation.airmass is not None,
                    "has_seeing": observation.seeing is not None,
                    "has_pixel_scale": observation.pixel_scale is not None,
                    "exposure_time_seconds": observation.exposure_time,
                    "filter_band": observation.filter_band,
                },
                "file_status": {
                    "has_fits_file": bool(observation.fits_file_path),
                    "has_thumbnail": bool(observation.thumbnail_path),
                    "fits_url": observation.fits_url,
                },
            }

            self.logger.debug(f"Calculated metrics for observation {observation.id}")
            return metrics

        except Exception as e:
            self.logger.error(
                f"Failed to calculate metrics for observation {observation.id}: {e}"
            )
            raise

    async def process_observation_status_change(
        self,
        observation_id: UUID,
        new_status: ObservationStatus,
        reason: str | None = None,
        changed_by: str | None = None,
    ) -> ObservationRead | None:
        """Process a status change with proper event handling and validation.

        Args:
            observation_id: Observation UUID
            new_status: New status to set
            reason: Reason for the status change
            changed_by: Who/what triggered the change

        Returns:
            ObservationRead: Updated observation or None if not found
        """
        self.logger.info(
            f"Processing status change for {observation_id} to {new_status.value}"
        )

        try:
            async with self.db.begin():  # Transaction management
                # Get current observation
                current_obs = await self.get_observation_by_id(observation_id)
                if not current_obs:
                    self.logger.warning(
                        f"Observation {observation_id} not found for status change"
                    )
                    return None

                old_status = current_obs.status

                # Validate status transition
                if not self._is_valid_status_transition(old_status, new_status):
                    raise ValueError(
                        f"Invalid status transition from {old_status} to {new_status}"
                    )

                # Update status
                updated_obs = await self.update_observation_status(
                    observation_id, new_status
                )
                if not updated_obs:
                    return None

                # Create status change event
                status_event = ObservationStatusChanged(
                    observation_id=observation_id,
                    survey_id=current_obs.survey_id,
                    old_status=old_status,
                    new_status=new_status,
                    changed_at=datetime.now(),
                    changed_by=changed_by,
                    reason=reason,
                )

                # Log event (in real implementation, this would be published to event bus)
                self.logger.info(f"Status change event: {status_event}")

                return updated_obs

        except Exception as e:
            self.logger.error(
                f"Failed to process status change for {observation_id}: {e}"
            )
            raise

    async def handle_observation_failure(
        self,
        observation_id: UUID,
        failed_stage: str,
        error_message: str,
        error_details: dict | None = None,
        is_retryable: bool = True,
    ) -> None:
        """Handle observation processing failure with proper error tracking.

        Args:
            observation_id: Observation UUID
            failed_stage: Processing stage that failed
            error_message: Error description
            error_details: Additional error details
            is_retryable: Whether the failure can be retried
        """
        self.logger.error(
            f"Handling failure for observation {observation_id} at stage {failed_stage}"
        )

        try:
            async with self.db.begin():  # Transaction management
                # Get current observation
                current_obs = await self.get_observation_by_id(observation_id)
                if not current_obs:
                    self.logger.warning(
                        f"Observation {observation_id} not found for failure handling"
                    )
                    return

                # Update status to failed
                await self.update_observation_status(
                    observation_id, ObservationStatus.FAILED
                )

                # Create failure event
                failure_event = ObservationFailed(
                    observation_id=observation_id,
                    survey_id=current_obs.survey_id,
                    failed_stage=failed_stage,
                    error_message=error_message,
                    error_details=error_details,
                    failed_at=datetime.now(),
                    previous_status=current_obs.status,
                    is_retryable=is_retryable,
                )

                # Log event (in real implementation, this would be published to event bus)
                self.logger.error(f"Failure event: {failure_event}")

        except Exception as e:
            self.logger.error(
                f"Failed to handle observation failure for {observation_id}: {e}"
            )
            raise

    def _is_valid_status_transition(
        self, old_status: ObservationStatus, new_status: ObservationStatus
    ) -> bool:
        """Validate if a status transition is allowed.

        Args:
            old_status: Current status
            new_status: Desired new status

        Returns:
            bool: True if transition is valid
        """
        # Define valid state transitions
        valid_transitions = {
            ObservationStatus.INGESTED: [
                ObservationStatus.PREPROCESSING,
                ObservationStatus.FAILED,
                ObservationStatus.ARCHIVED,
            ],
            ObservationStatus.PREPROCESSING: [
                ObservationStatus.PREPROCESSED,
                ObservationStatus.FAILED,
            ],
            ObservationStatus.PREPROCESSED: [
                ObservationStatus.DIFFERENCING,
                ObservationStatus.FAILED,
                ObservationStatus.ARCHIVED,
            ],
            ObservationStatus.DIFFERENCING: [
                ObservationStatus.DIFFERENCED,
                ObservationStatus.FAILED,
            ],
            ObservationStatus.DIFFERENCED: [
                ObservationStatus.ARCHIVED,
                ObservationStatus.FAILED,
            ],
            ObservationStatus.FAILED: [
                ObservationStatus.INGESTED,  # For retry
                ObservationStatus.ARCHIVED,
            ],
            ObservationStatus.ARCHIVED: [],  # Terminal state
        }

        allowed_transitions = valid_transitions.get(old_status, [])
        return new_status in allowed_transitions

    async def get_survey_observation_summary(self, survey_id: UUID) -> dict[str, Any]:
        """Get comprehensive summary of observations for a survey.

        Args:
            survey_id: Survey UUID

        Returns:
            dict: Survey observation summary
        """
        self.logger.debug(f"Getting observation summary for survey {survey_id}")

        try:
            # Get survey info
            survey = await SurveyCRUD().get_by_id(self.db, survey_id)
            if not survey:
                raise ValueError(f"Survey {survey_id} not found")

            # Get observation counts by status
            status_counts = {}
            for status in ObservationStatus:
                params = ObservationListParams(
                    survey_id=survey_id, status=status, limit=1
                )
                _, count = await self.crud.get_many(self.db, params)
                status_counts[status.value] = count

            # Get recent observations
            recent_params = ObservationListParams(survey_id=survey_id, limit=10)
            recent_obs, _ = await self.crud.get_many(self.db, recent_params)

            summary = {
                "survey": {
                    "id": str(survey.id),
                    "name": survey.name,
                    "description": survey.description,
                    "is_active": survey.is_active,
                },
                "observation_counts": {
                    "total": sum(status_counts.values()),
                    "by_status": status_counts,
                },
                "recent_observations": len(recent_obs),
                "processing_summary": {
                    "ready_for_processing": status_counts.get(
                        ObservationStatus.INGESTED.value, 0
                    ),
                    "in_progress": (
                        status_counts.get(ObservationStatus.PREPROCESSING.value, 0)
                        + status_counts.get(ObservationStatus.DIFFERENCING.value, 0)
                    ),
                    "completed": status_counts.get(
                        ObservationStatus.DIFFERENCED.value, 0
                    ),
                    "failed": status_counts.get(ObservationStatus.FAILED.value, 0),
                    "archived": status_counts.get(ObservationStatus.ARCHIVED.value, 0),
                },
            }

            self.logger.debug(f"Generated observation summary for survey {survey_id}")
            return summary

        except Exception as e:
            self.logger.error(
                f"Failed to get survey observation summary for {survey_id}: {e}"
            )
            raise
