"""Service for managing survey configurations."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.logging import configure_domain_logger
from src.domains.catalog.models import SystemConfig
from src.domains.observations.survey_config import (
    DEFAULT_SURVEY_CONFIGURATIONS,
    SurveyConfiguration,
    SurveyConfigurationCreate,
    SurveyConfigurationRead,
    SurveyConfigurationUpdate,
)


class SurveyConfigurationService:
    """Service for managing survey configurations using SystemConfig storage."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.logger = configure_domain_logger("observations.survey_config")

    async def create_configuration(
        self, config_data: SurveyConfigurationCreate
    ) -> SurveyConfigurationRead:
        """Create a new survey configuration."""
        self.logger.info(f"Creating survey configuration: {config_data.name}")

        try:
            # Convert to SurveyConfiguration for storage
            config = SurveyConfiguration(
                name=config_data.name,
                description=config_data.description,
                sky_regions=config_data.sky_regions,
                missions=config_data.missions,
                query_radius=config_data.query_radius,
                time_window_days=config_data.time_window_days,
                max_observations_per_run=config_data.max_observations_per_run,
                auto_download=config_data.auto_download,
                auto_preprocess=config_data.auto_preprocess,
                quality_threshold=config_data.quality_threshold,
                schedule_cron=config_data.schedule_cron,
                is_active=config_data.is_active,
            )

            # Store in SystemConfig table
            system_config = SystemConfig(
                key=f"survey_config:{config_data.name.lower().replace(' ', '_')}",
                value=config.model_dump(),
                description=f"Survey configuration: {config_data.name}",
                is_active=True,
            )

            self.db.add(system_config)
            await self.db.flush()
            await self.db.refresh(system_config)

            # Convert back to read format
            config_dict = system_config.value
            config_dict["id"] = system_config.id
            config_dict["created_at"] = system_config.created_at
            config_dict["updated_at"] = system_config.updated_at

            result = SurveyConfigurationRead(**config_dict)

            self.logger.info(f"Successfully created survey configuration: {result.id}")
            return result

        except Exception as e:
            await self.db.rollback()
            self.logger.error(f"Failed to create survey configuration: {e}")
            raise

    async def get_configuration(
        self, config_id: UUID
    ) -> SurveyConfigurationRead | None:
        """Get a survey configuration by ID."""
        self.logger.debug(f"Retrieving survey configuration: {config_id}")

        try:
            result = await self.db.execute(
                select(SystemConfig).where(SystemConfig.id == config_id)
            )
            system_config = result.scalar_one_or_none()

            if not system_config:
                self.logger.warning(f"Survey configuration not found: {config_id}")
                return None

            # Convert to read format
            config_dict = system_config.value
            config_dict["id"] = system_config.id
            config_dict["created_at"] = system_config.created_at
            config_dict["updated_at"] = system_config.updated_at

            return SurveyConfigurationRead(**config_dict)

        except Exception as e:
            self.logger.error(f"Error retrieving survey configuration: {e}")
            raise

    async def get_configuration_by_name(
        self, name: str
    ) -> SurveyConfigurationRead | None:
        """Get a survey configuration by name."""
        self.logger.debug(f"Retrieving survey configuration by name: {name}")

        try:
            key = f"survey_config:{name.lower().replace(' ', '_')}"
            result = await self.db.execute(
                select(SystemConfig).where(SystemConfig.key == key)
            )
            system_config = result.scalar_one_or_none()

            if not system_config:
                self.logger.warning(f"Survey configuration not found: {name}")
                return None

            # Convert to read format
            config_dict = system_config.value
            config_dict["id"] = system_config.id
            config_dict["created_at"] = system_config.created_at
            config_dict["updated_at"] = system_config.updated_at

            return SurveyConfigurationRead(**config_dict)

        except Exception as e:
            self.logger.error(f"Error retrieving survey configuration by name: {e}")
            raise

    async def list_configurations(
        self, is_active: bool | None = None, limit: int = 100, offset: int = 0
    ) -> list[SurveyConfigurationRead]:
        """List survey configurations."""
        self.logger.debug(
            f"Listing survey configurations: active={is_active}, limit={limit}"
        )

        try:
            query = select(SystemConfig).where(SystemConfig.key.like("survey_config:%"))

            if is_active is not None:
                query = query.where(SystemConfig.is_active == is_active)

            query = query.offset(offset).limit(limit)

            result = await self.db.execute(query)
            system_configs = result.scalars().all()

            configurations = []
            for system_config in system_configs:
                config_dict = system_config.value
                config_dict["id"] = system_config.id
                config_dict["created_at"] = system_config.created_at
                config_dict["updated_at"] = system_config.updated_at
                configurations.append(SurveyConfigurationRead(**config_dict))

            self.logger.info(f"Found {len(configurations)} survey configurations")
            return configurations

        except Exception as e:
            self.logger.error(f"Error listing survey configurations: {e}")
            raise

    async def update_configuration(
        self, config_id: UUID, update_data: SurveyConfigurationUpdate
    ) -> SurveyConfigurationRead | None:
        """Update an existing survey configuration."""
        self.logger.info(f"Updating survey configuration: {config_id}")

        try:
            result = await self.db.execute(
                select(SystemConfig).where(SystemConfig.id == config_id)
            )
            system_config = result.scalar_one_or_none()

            if not system_config:
                self.logger.warning(f"Survey configuration not found: {config_id}")
                return None

            # Get current configuration
            current_config = SurveyConfiguration(**system_config.value)

            # Apply updates
            update_dict = update_data.model_dump(exclude_unset=True)
            for field, value in update_dict.items():
                setattr(current_config, field, value)

            # Update system config
            system_config.value = current_config.model_dump()
            system_config.is_active = current_config.is_active

            await self.db.flush()
            await self.db.refresh(system_config)

            # Convert to read format
            config_dict = system_config.value
            config_dict["id"] = system_config.id
            config_dict["created_at"] = system_config.created_at
            config_dict["updated_at"] = system_config.updated_at

            result = SurveyConfigurationRead(**config_dict)

            self.logger.info(f"Successfully updated survey configuration: {config_id}")
            return result

        except Exception as e:
            await self.db.rollback()
            self.logger.error(f"Failed to update survey configuration: {e}")
            raise

    async def delete_configuration(self, config_id: UUID) -> bool:
        """Delete a survey configuration."""
        self.logger.info(f"Deleting survey configuration: {config_id}")

        try:
            result = await self.db.execute(
                select(SystemConfig).where(SystemConfig.id == config_id)
            )
            system_config = result.scalar_one_or_none()

            if not system_config:
                self.logger.warning(f"Survey configuration not found: {config_id}")
                return False

            await self.db.delete(system_config)
            await self.db.flush()

            self.logger.info(f"Successfully deleted survey configuration: {config_id}")
            return True

        except Exception as e:
            await self.db.rollback()
            self.logger.error(f"Failed to delete survey configuration: {e}")
            raise

    async def get_active_configurations(self) -> list[SurveyConfigurationRead]:
        """Get all active survey configurations."""
        return await self.list_configurations(is_active=True)

    async def initialize_default_configurations(self) -> list[SurveyConfigurationRead]:
        """Initialize default survey configurations if they don't exist."""
        self.logger.info("Initializing default survey configurations")

        created_configs = []

        for _name, config_data in DEFAULT_SURVEY_CONFIGURATIONS.items():
            try:
                # Check if configuration already exists
                existing = await self.get_configuration_by_name(config_data.name)
                if existing:
                    self.logger.debug(
                        f"Configuration already exists: {config_data.name}"
                    )
                    continue

                # Create new configuration
                config = await self.create_configuration(config_data)
                created_configs.append(config)
                self.logger.info(f"Created default configuration: {config_data.name}")

            except Exception as e:
                self.logger.error(
                    f"Failed to create default configuration {config_data.name}: {e}"
                )
                continue

        await self.db.commit()
        self.logger.info(f"Initialized {len(created_configs)} default configurations")
        return created_configs

    async def get_configuration_for_ingestion(self) -> SurveyConfigurationRead | None:
        """Get the primary configuration for observation ingestion."""
        # For now, return the first active configuration
        # In the future, this could be more sophisticated (e.g., based on priority)
        active_configs = await self.get_active_configurations()

        if not active_configs:
            self.logger.warning("No active survey configurations found")
            return None

        # Return the highest priority configuration (lowest priority number)
        primary_config = min(
            active_configs,
            key=lambda c: min(r.priority for r in c.sky_regions)
            if c.sky_regions
            else 10,
        )

        self.logger.info(f"Using configuration for ingestion: {primary_config.name}")
        return primary_config
