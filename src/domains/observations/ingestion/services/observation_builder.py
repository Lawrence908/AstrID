"""Service for building observation records from ingested data."""

from datetime import UTC
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.adapters.imaging.fits_io import FITSProcessor
from src.core.logging import configure_domain_logger
from src.domains.observations.crud import ObservationCRUD, SurveyCRUD
from src.domains.observations.models import Observation
from src.domains.observations.schema import ObservationCreate, ObservationUpdate


class ObservationBuilderService:
    """Service for building and enriching observation records."""

    def __init__(self):
        """Initialize the observation builder service."""
        self.logger = configure_domain_logger("observations.builder")
        self.fits_processor = FITSProcessor()
        self.observation_crud = ObservationCRUD()
        self.survey_crud = SurveyCRUD()

    async def create_observations_from_ingestion(
        self, db: AsyncSession, observation_creates: list[ObservationCreate]
    ) -> list[Observation]:
        """Create observation records from ingested data.

        Args:
            db: Database session
            observation_creates: List of ObservationCreate objects

        Returns:
            List of created Observation objects
        """
        self.logger.info(f"Creating {len(observation_creates)} observation records")

        created_observations = []

        for obs_create in observation_creates:
            try:
                # Validate survey exists
                survey = await self.survey_crud.get_by_id(db, obs_create.survey_id)
                if not survey:
                    self.logger.error(f"Survey not found: {obs_create.survey_id}")
                    continue

                # Check for duplicate
                existing = await self.observation_crud.get_by_survey_and_observation_id(
                    db, obs_create.survey_id, obs_create.observation_id
                )
                if existing:
                    self.logger.warning(
                        f"Observation already exists: {obs_create.observation_id}"
                    )
                    continue

                # Create observation
                observation = await self.observation_crud.create(db, obs_create)
                created_observations.append(observation)

                self.logger.debug(f"Created observation: {observation.id}")

            except Exception as e:
                self.logger.error(
                    f"Failed to create observation {obs_create.observation_id}: {e}"
                )
                continue

        self.logger.info(
            f"Successfully created {len(created_observations)} observations"
        )
        return created_observations

    async def enrich_observation_from_fits(
        self,
        db: AsyncSession,
        observation_id: UUID,
        fits_file_path: str,
        r2_object_key: str | None = None,
    ) -> Observation | None:
        """Enrich observation record with FITS file metadata.

        Args:
            db: Database session
            observation_id: Observation UUID
            fits_file_path: Path to FITS file (local or R2)
            r2_object_key: R2 storage key if file is in cloud storage

        Returns:
            Updated observation or None if not found
        """
        self.logger.info(f"Enriching observation {observation_id} with FITS metadata")

        try:
            # Get observation
            observation = await self.observation_crud.get_by_id(db, observation_id)
            if not observation:
                self.logger.error(f"Observation not found: {observation_id}")
                return None

            # Extract FITS metadata
            fits_metadata = self._extract_fits_metadata(fits_file_path)

            # Create update object with FITS metadata
            update_data = ObservationUpdate(
                image_width=fits_metadata.get("image_width"),
                image_height=fits_metadata.get("image_height"),
                pixel_scale=fits_metadata.get("pixel_scale"),
                airmass=fits_metadata.get("airmass"),
                seeing=fits_metadata.get("seeing"),
                fits_file_path=r2_object_key or fits_file_path,
            )

            # Update observation
            updated_observation = await self.observation_crud.update(
                db, observation_id, update_data
            )

            if updated_observation:
                self.logger.info(f"Successfully enriched observation: {observation_id}")

            return updated_observation

        except Exception as e:
            self.logger.error(f"Failed to enrich observation {observation_id}: {e}")
            return None

    async def batch_enrich_observations(
        self,
        db: AsyncSession,
        observations_with_files: list[dict[str, Any]],
    ) -> list[Observation]:
        """Batch enrich multiple observations with FITS metadata.

        Args:
            db: Database session
            observations_with_files: List of dicts with 'observation_id' and 'fits_path' keys

        Returns:
            List of enriched observations
        """
        self.logger.info(f"Batch enriching {len(observations_with_files)} observations")

        enriched_observations = []

        for obs_data in observations_with_files:
            try:
                observation_id = obs_data["observation_id"]
                fits_path = obs_data["fits_path"]
                r2_key = obs_data.get("r2_object_key")

                enriched = await self.enrich_observation_from_fits(
                    db, observation_id, fits_path, r2_key
                )

                if enriched:
                    enriched_observations.append(enriched)

            except Exception as e:
                self.logger.error(f"Failed to enrich observation {obs_data}: {e}")
                continue

        self.logger.info(
            f"Successfully enriched {len(enriched_observations)} observations"
        )
        return enriched_observations

    async def create_observation_from_fits_file(
        self,
        db: AsyncSession,
        survey_id: UUID,
        fits_file_path: str,
        observation_id: str | None = None,
        additional_metadata: dict[str, Any] | None = None,
    ) -> Observation | None:
        """Create a complete observation record from a FITS file.

        This method combines ingestion and enrichment in one step for local FITS files.

        Args:
            db: Database session
            survey_id: Survey UUID
            fits_file_path: Path to FITS file
            observation_id: Optional observation ID (generated if None)
            additional_metadata: Additional metadata to include

        Returns:
            Created observation or None if failed
        """
        self.logger.info(f"Creating observation from FITS file: {fits_file_path}")

        try:
            # Extract FITS metadata using the clean adapter
            image_data, wcs, fits_metadata = self.fits_processor.read_fits(
                fits_file_path
            )

            # Generate observation ID if not provided
            if observation_id is None:
                file_name = Path(fits_file_path).stem
                observation_id = f"local_{file_name}"

            # Extract coordinate information from WCS if available
            ra, dec = 0.0, 0.0
            if wcs is not None:
                try:
                    # Get center coordinates
                    center_coord = wcs.pixel_to_world(
                        image_data.shape[1] // 2, image_data.shape[0] // 2
                    )
                    ra = float(center_coord.ra.degree)
                    dec = float(center_coord.dec.degree)
                except Exception:
                    pass

            # Extract observation time
            from datetime import datetime

            obs_time = datetime.now(UTC)
            if "DATE-OBS" in fits_metadata:
                try:
                    obs_time = datetime.fromisoformat(
                        fits_metadata["DATE-OBS"].replace("T", " ").replace("Z", "")
                    )
                except Exception:
                    pass

            # Create ObservationCreate object
            obs_create = ObservationCreate(
                survey_id=survey_id,
                observation_id=observation_id,
                ra=ra,
                dec=dec,
                observation_time=obs_time,
                filter_band=fits_metadata.get(
                    "FILTER", fits_metadata.get("BAND", "unknown")
                ),
                exposure_time=float(
                    fits_metadata.get("EXPTIME", fits_metadata.get("EXPOSURE", 0.0))
                ),
                fits_url=f"file://{fits_file_path}",
                # Include FITS-derived metadata
                image_width=image_data.shape[1] if image_data is not None else None,
                image_height=image_data.shape[0] if image_data is not None else None,
                pixel_scale=None,  # Would need WCS calculation
                airmass=float(fits_metadata.get("AIRMASS"))
                if fits_metadata.get("AIRMASS")
                else None,
                seeing=float(fits_metadata.get("SEEING"))
                if fits_metadata.get("SEEING")
                else None,
            )

            # Apply additional metadata if provided
            if additional_metadata:
                for key, value in additional_metadata.items():
                    if hasattr(obs_create, key):
                        setattr(obs_create, key, value)

            # Create observation
            observation = await self.observation_crud.create(db, obs_create)

            self.logger.info(f"Created observation from FITS: {observation.id}")
            return observation

        except Exception as e:
            self.logger.error(
                f"Failed to create observation from FITS {fits_file_path}: {e}"
            )
            return None

    # Removed _extract_fits_metadata method - now using FITSProcessor adapter

    async def create_observations_from_directory(
        self,
        db: AsyncSession,
        survey_id: UUID,
        directory_path: str,
        file_pattern: str = "*.fits",
    ) -> list[Observation]:
        """Create observations from all FITS files in a directory.

        Args:
            db: Database session
            survey_id: Survey UUID
            directory_path: Path to directory containing FITS files
            file_pattern: Glob pattern for files to process

        Returns:
            List of created observations
        """
        self.logger.info(f"Creating observations from directory: {directory_path}")

        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        fits_files = list(directory.glob(file_pattern))
        self.logger.info(f"Found {len(fits_files)} FITS files")

        created_observations = []

        for fits_file in fits_files:
            try:
                observation = await self.create_observation_from_fits_file(
                    db=db,
                    survey_id=survey_id,
                    fits_file_path=str(fits_file),
                )

                if observation:
                    created_observations.append(observation)

            except Exception as e:
                self.logger.error(f"Failed to process {fits_file}: {e}")
                continue

        self.logger.info(
            f"Created {len(created_observations)} observations from {len(fits_files)} files"
        )
        return created_observations
