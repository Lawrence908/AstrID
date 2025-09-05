"""Preprocessing service layer for business logic."""

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.logging import configure_domain_logger
from src.domains.preprocessing.repository import PreprocessRunRepository
from src.domains.preprocessing.schema import PreprocessRunCreate


class PreprocessRunService:
    """Service for PreprocessRun business logic."""

    def __init__(self, db: AsyncSession):
        self.repository = PreprocessRunRepository(db)
        self.logger = configure_domain_logger("preprocessing.preprocess_run")

    async def create_preprocess_run(self, run_data: PreprocessRunCreate):
        """Create a new preprocessing run with business logic."""
        self.logger.info(
            f"Creating preprocessing run: observation_id={run_data.observation_id}, algorithm={run_data.algorithm}"
        )
        try:
            # TODO: Add validation logic, automatic status initialization, etc.
            result = await self.repository.create(run_data)
            self.logger.info(
                f"Successfully created preprocessing run: id={result.id}, algorithm={result.algorithm}"
            )
            return result
        except Exception as e:
            self.logger.error(
                f"Failed to create preprocessing run: observation_id={run_data.observation_id}, error={str(e)}"
            )
            raise

    async def get_preprocess_run(self, run_id: str):
        """Get preprocessing run by ID."""
        self.logger.debug(f"Retrieving preprocessing run by ID: {run_id}")
        result = await self.repository.get_by_id(run_id)
        if result:
            self.logger.debug(
                f"Found preprocessing run: id={result.id}, algorithm={result.algorithm}, status={result.status}"
            )
        else:
            self.logger.warning(f"Preprocessing run not found: id={run_id}")
        return result

    async def list_preprocess_runs(
        self,
        observation_id: str | None = None,
        status: str | None = None,
        algorithm: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ):
        """List preprocessing runs."""
        self.logger.debug(
            f"Listing preprocessing runs: observation_id={observation_id}, status={status}, algorithm={algorithm}, limit={limit}"
        )
        result = await self.repository.list(
            observation_id=observation_id,
            status=status,
            algorithm=algorithm,
            limit=limit,
            offset=offset,
        )
        self.logger.debug(f"Retrieved {len(result)} preprocessing runs")
        return result

    async def run_preprocessing(self, observation_id: str, algorithm: str = "standard"):
        """Run preprocessing algorithm on an observation."""
        self.logger.info(
            f"Starting preprocessing: observation_id={observation_id}, algorithm={algorithm}"
        )
        try:
            # TODO: Implement actual preprocessing logic
            # This would typically:
            # 1. Load the observation data
            # 2. Apply preprocessing steps (bias correction, flat fielding, etc.)
            # 3. Create preprocess run record
            # 4. Return results
            result = {
                "message": f"Preprocessing initiated for observation {observation_id}",
                "observation_id": observation_id,
                "algorithm": algorithm,
                "status": "queued",
            }
            self.logger.info(
                f"Preprocessing queued successfully: observation_id={observation_id}, algorithm={algorithm}"
            )
            return result
        except Exception as e:
            self.logger.error(
                f"Failed to start preprocessing: observation_id={observation_id}, algorithm={algorithm}, error={str(e)}"
            )
            raise
