"""Differencing service layer for business logic."""

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.logging import configure_domain_logger
from src.domains.differencing.repository import (
    CandidateRepository,
    DifferenceRunRepository,
)
from src.domains.differencing.schema import CandidateCreate, DifferenceRunCreate


class DifferenceRunService:
    """Service for DifferenceRun business logic."""

    def __init__(self, db: AsyncSession):
        self.repository = DifferenceRunRepository(db)
        self.logger = configure_domain_logger("differencing.difference_run")

    async def create_difference_run(self, run_data: DifferenceRunCreate):
        """Create a new difference run with business logic."""
        self.logger.info(
            f"Creating difference run: observation_id={run_data.observation_id}, algorithm={run_data.algorithm}"
        )
        try:
            # TODO: Add validation logic, automatic status initialization, etc.
            result = await self.repository.create(run_data)
            self.logger.info(
                f"Successfully created difference run: id={result.id}, algorithm={result.algorithm}"
            )
            return result
        except Exception as e:
            self.logger.error(
                f"Failed to create difference run: observation_id={run_data.observation_id}, error={str(e)}"
            )
            raise

    async def get_difference_run(self, run_id: str):
        """Get difference run by ID."""
        self.logger.debug(f"Retrieving difference run by ID: {run_id}")
        result = await self.repository.get_by_id(run_id)
        if result:
            self.logger.debug(
                f"Found difference run: id={result.id}, algorithm={result.algorithm}, status={result.status}"
            )
        else:
            self.logger.warning(f"Difference run not found: id={run_id}")
        return result

    async def list_difference_runs(
        self,
        observation_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ):
        """List difference runs."""
        self.logger.debug(
            f"Listing difference runs: observation_id={observation_id}, status={status}, limit={limit}"
        )
        result = await self.repository.list(
            observation_id=observation_id, status=status, limit=limit, offset=offset
        )
        self.logger.debug(f"Retrieved {len(result)} difference runs")
        return result

    async def run_differencing(self, observation_id: str, algorithm: str = "zogy"):
        """Run differencing algorithm on an observation."""
        self.logger.info(
            f"Starting differencing: observation_id={observation_id}, algorithm={algorithm}"
        )
        try:
            # TODO: Implement actual differencing logic
            # This would typically:
            # 1. Load the observation data
            # 2. Find reference images
            # 3. Run the differencing algorithm
            # 4. Create difference run and candidate records
            # 5. Return results
            result = {
                "message": f"Differencing initiated for observation {observation_id}",
                "observation_id": observation_id,
                "algorithm": algorithm,
                "status": "queued",
            }
            self.logger.info(
                f"Differencing queued successfully: observation_id={observation_id}, algorithm={algorithm}"
            )
            return result
        except Exception as e:
            self.logger.error(
                f"Failed to start differencing: observation_id={observation_id}, algorithm={algorithm}, error={str(e)}"
            )
            raise


class CandidateService:
    """Service for Candidate business logic."""

    def __init__(self, db: AsyncSession):
        self.repository = CandidateRepository(db)
        self.logger = configure_domain_logger("differencing.candidate")

    async def create_candidate(self, candidate_data: CandidateCreate):
        """Create a new candidate with business logic."""
        self.logger.info(
            f"Creating candidate: difference_run_id={candidate_data.difference_run_id}, candidate_type={candidate_data.candidate_type}"
        )
        try:
            # TODO: Add validation logic, automatic scoring, etc.
            result = await self.repository.create(candidate_data)
            self.logger.info(
                f"Successfully created candidate: id={result.id}, type={result.candidate_type}, score={result.score}"
            )
            return result
        except Exception as e:
            self.logger.error(
                f"Failed to create candidate: difference_run_id={candidate_data.difference_run_id}, error={str(e)}"
            )
            raise

    async def get_candidate(self, candidate_id: str):
        """Get candidate by ID."""
        self.logger.debug(f"Retrieving candidate by ID: {candidate_id}")
        result = await self.repository.get_by_id(candidate_id)
        if result:
            self.logger.debug(
                f"Found candidate: id={result.id}, type={result.candidate_type}, score={result.score}"
            )
        else:
            self.logger.warning(f"Candidate not found: id={candidate_id}")
        return result

    async def list_candidates(
        self,
        difference_run_id: str | None = None,
        candidate_type: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ):
        """List candidates."""
        self.logger.debug(
            f"Listing candidates: difference_run_id={difference_run_id}, candidate_type={candidate_type}, status={status}, limit={limit}"
        )
        result = await self.repository.list(
            difference_run_id=difference_run_id,
            candidate_type=candidate_type,
            status=status,
            limit=limit,
            offset=offset,
        )
        self.logger.debug(f"Retrieved {len(result)} candidates")
        return result
