"""Service layer for observations domain."""

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.logging import configure_domain_logger
from src.domains.observations.crud import ObservationCRUD, SurveyCRUD
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
