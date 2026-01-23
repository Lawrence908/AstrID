"""
Archive router for querying multiple astronomical archives.

This module provides a unified interface to query both MAST and IRSA archives,
automatically routing missions to the correct archive based on mission name.
"""

import logging
from datetime import datetime
from typing import Any

from src.adapters.external.irsa import IRSAClient
from src.adapters.external.mast import MASTClient

logger = logging.getLogger(__name__)

# Mission to archive mapping
MISSION_TO_ARCHIVE = {
    # MAST missions
    "HST": "MAST",
    "JWST": "MAST",
    "TESS": "MAST",
    "GALEX": "MAST",
    "SWIFT": "MAST",
    "PS1": "MAST",
    "SDSS": "MAST",
    "HLA": "MAST",
    "HLSP": "MAST",
    # IRSA missions
    "ZTF": "IRSA",
    "PTF": "IRSA",
    "WISE": "IRSA",
    "NEOWISE": "IRSA",
    "2MASS": "IRSA",
    "SPITZER": "IRSA",
}


class ArchiveRouter:
    """Routes queries to appropriate archive based on mission name."""

    def __init__(self, timeout: int = 120, test_mode: bool = False):
        """Initialize archive router.

        Args:
            timeout: Query timeout in seconds (default: 120 for large queries)
            test_mode: If True, use mock data instead of real queries
        """
        self.timeout = timeout
        self.test_mode = test_mode
        self.logger = logging.getLogger(__name__)

        # Initialize clients
        self.mast_client = MASTClient(timeout=timeout, test_mode=test_mode)
        self.irsa_client = IRSAClient(timeout=timeout, test_mode=test_mode)

    def get_archive_for_mission(self, mission: str) -> str | None:
        """Get the archive name for a given mission.

        Args:
            mission: Mission name (e.g., 'HST', 'ZTF')

        Returns:
            Archive name ('MAST' or 'IRSA') or None if unknown
        """
        mission_upper = mission.upper()
        return MISSION_TO_ARCHIVE.get(mission_upper)

    def split_missions_by_archive(
        self, missions: list[str] | None
    ) -> dict[str, list[str]]:
        """Split missions into groups by archive.

        Args:
            missions: List of mission names

        Returns:
            Dictionary mapping archive names to lists of missions
        """
        if not missions:
            return {"MAST": [], "IRSA": []}

        archive_groups: dict[str, list[str]] = {"MAST": [], "IRSA": []}
        unknown_missions = []

        for mission in missions:
            archive = self.get_archive_for_mission(mission)
            if archive:
                archive_groups[archive].append(mission)
            else:
                unknown_missions.append(mission)

        if unknown_missions:
            self.logger.warning(
                f"Unknown missions (will be skipped): {unknown_missions}"
            )

        return archive_groups

    async def query_observations_by_position(
        self,
        ra: float,
        dec: float,
        radius: float = 0.1,
        missions: list[str] | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        data_rights: str = "PUBLIC",
    ) -> list[dict[str, Any]]:
        """Query observations from appropriate archives based on mission names.

        This method automatically routes missions to the correct archive (MAST or IRSA)
        and queries them in parallel, then merges the results.

        Args:
            ra: Right ascension in degrees
            dec: Declination in degrees
            radius: Search radius in degrees
            missions: List of mission names (e.g., ['HST', 'ZTF', 'GALEX'])
            start_time: Start of observation time range
            end_time: End of observation time range
            data_rights: 'PUBLIC' or 'EXCLUSIVE'

        Returns:
            Combined list of observation metadata dictionaries from all archives
        """
        # Split missions by archive
        archive_groups = self.split_missions_by_archive(missions)

        all_observations = []

        # Query MAST missions
        if archive_groups["MAST"]:
            self.logger.info(
                f"Querying MAST for missions: {archive_groups['MAST']}"
            )
            try:
                mast_obs = await self.mast_client.query_observations_by_position(
                    ra=ra,
                    dec=dec,
                    radius=radius,
                    missions=archive_groups["MAST"],
                    start_time=start_time,
                    end_time=end_time,
                    data_rights=data_rights,
                )
                all_observations.extend(mast_obs)
                self.logger.info(f"MAST returned {len(mast_obs)} observations")
            except Exception as e:
                self.logger.error(f"Error querying MAST: {e}")

        # Query IRSA missions
        if archive_groups["IRSA"]:
            self.logger.info(
                f"Querying IRSA for missions: {archive_groups['IRSA']}"
            )
            try:
                irsa_obs = await self.irsa_client.query_observations_by_position(
                    ra=ra,
                    dec=dec,
                    radius=radius,
                    missions=archive_groups["IRSA"],
                    start_time=start_time,
                    end_time=end_time,
                    data_rights=data_rights,
                )
                all_observations.extend(irsa_obs)
                self.logger.info(f"IRSA returned {len(irsa_obs)} observations")
            except Exception as e:
                self.logger.error(f"Error querying IRSA: {e}")

        self.logger.info(
            f"Total observations from all archives: {len(all_observations)}"
        )

        return all_observations

    def get_supported_missions(self) -> dict[str, list[str]]:
        """Get list of supported missions by archive.

        Returns:
            Dictionary mapping archive names to lists of supported missions
        """
        missions_by_archive: dict[str, list[str]] = {"MAST": [], "IRSA": []}

        for mission, archive in MISSION_TO_ARCHIVE.items():
            missions_by_archive[archive].append(mission)

        return missions_by_archive
