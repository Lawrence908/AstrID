"""Star catalog integration for astrometric and photometric reference."""

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

logger = logging.getLogger(__name__)


@dataclass
class StarCatalogConfig:
    """Configuration for star catalog queries and caching."""

    catalogs: list[str]  # ['gaia', 'tycho2', 'usnob1']
    search_radius: float  # arcseconds
    magnitude_limit: float
    cache_size: int
    update_frequency: str  # 'daily', 'weekly', 'monthly'
    cache_directory: str | None = None
    max_stars_per_query: int = 10000
    coordinate_precision: float = 1e-6  # degrees
    magnitude_precision: float = 0.001


class StarCatalog:
    """Star catalog interface for astrometric and photometric reference."""

    def __init__(self, config: StarCatalogConfig):
        """Initialize star catalog with configuration.

        Args:
            config: Star catalog configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize cache
        self.cache_dir = Path(
            config.cache_directory or "~/.astrid/catalog_cache"
        ).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_db = self.cache_dir / "star_catalog.db"
        self._init_cache_db()

        # Catalog-specific configurations
        self.catalog_configs = {
            "gaia": {
                "base_url": "https://gea.esac.esa.int/tap-server/tap",
                "table": "gaiadr3.gaia_source",
                "ra_col": "ra",
                "dec_col": "dec",
                "mag_col": "phot_g_mean_mag",
                "id_col": "source_id",
                "pmra_col": "pmra",
                "pmdec_col": "pmdec",
                "parallax_col": "parallax",
            },
            "tycho2": {
                "base_url": "https://vizier.u-strasbg.fr/viz-bin/votable",
                "catalog": "I/259/tyc2",
                "ra_col": "RAmdeg",
                "dec_col": "DEmdeg",
                "mag_col": "VTmag",
                "id_col": "TYC",
                "bt_mag_col": "BTmag",
            },
            "usnob1": {
                "base_url": "https://vizier.u-strasbg.fr/viz-bin/votable",
                "catalog": "I/284/out",
                "ra_col": "RAJ2000",
                "dec_col": "DEJ2000",
                "mag_col": "R2mag",
                "id_col": "USNO-B1.0",
                "b1_mag_col": "B1mag",
                "r1_mag_col": "R1mag",
                "b2_mag_col": "B2mag",
            },
        }

    def query_stars_in_region(
        self, ra: float, dec: float, radius: float
    ) -> list[dict[str, Any]]:
        """Query stars in a circular region around given coordinates.

        Args:
            ra: Right ascension in degrees
            dec: Declination in degrees
            radius: Search radius in arcseconds

        Returns:
            List of star dictionaries with standardized fields

        Raises:
            ValueError: If coordinates or radius are invalid
        """
        try:
            # Validate inputs
            if not (0 <= ra <= 360):
                raise ValueError(f"Invalid RA: {ra}. Must be 0-360 degrees.")
            if not (-90 <= dec <= 90):
                raise ValueError(f"Invalid Dec: {dec}. Must be -90 to 90 degrees.")
            if radius <= 0:
                raise ValueError(f"Invalid radius: {radius}. Must be positive.")

            # Check cache first
            cached_stars = self._query_cache(ra, dec, radius)
            if cached_stars is not None:
                self.logger.debug(f"Retrieved {len(cached_stars)} stars from cache")
                return cached_stars

            # Query each configured catalog
            all_stars = []
            for catalog_name in self.config.catalogs:
                if catalog_name not in self.catalog_configs:
                    self.logger.warning(f"Unknown catalog: {catalog_name}")
                    continue

                try:
                    stars = self._query_catalog(catalog_name, ra, dec, radius)
                    all_stars.extend(stars)
                    self.logger.debug(
                        f"Retrieved {len(stars)} stars from {catalog_name}"
                    )
                except Exception as e:
                    self.logger.error(f"Error querying {catalog_name}: {e}")
                    continue

            # Remove duplicates and apply magnitude limit
            unique_stars = self._deduplicate_stars(all_stars)
            filtered_stars = self._apply_magnitude_filter(unique_stars)

            # Cache results
            self._cache_results(ra, dec, radius, filtered_stars)

            self.logger.info(
                f'Retrieved {len(filtered_stars)} stars in {radius}" around '
                f"RA={ra:.3f}, Dec={dec:.3f}"
            )

            return filtered_stars

        except Exception as e:
            self.logger.error(f"Error querying stars in region: {e}")
            raise

    def get_star_by_id(
        self, star_id: str, catalog: str = "gaia"
    ) -> dict[str, Any] | None:
        """Get specific star by catalog ID.

        Args:
            star_id: Star identifier in the catalog
            catalog: Catalog name ('gaia', 'tycho2', 'usnob1')

        Returns:
            Star data dictionary or None if not found
        """
        try:
            if catalog not in self.catalog_configs:
                raise ValueError(f"Unknown catalog: {catalog}")

            # Check cache first
            cached_star = self._get_cached_star_by_id(star_id, catalog)
            if cached_star:
                return cached_star

            # Query catalog
            star_data = self._query_star_by_id(catalog, star_id)
            if star_data:
                # Cache the result
                self._cache_star(star_data)

            return star_data

        except Exception as e:
            self.logger.error(f"Error getting star {star_id} from {catalog}: {e}")
            return None

    def match_stars_to_image(
        self, stars: list[dict[str, Any]], wcs: WCS, image_shape: tuple[int, int]
    ) -> list[dict[str, Any]]:
        """Match catalog stars to image coordinates.

        Args:
            stars: List of star catalog entries
            wcs: WCS solution for the image
            image_shape: Image shape as (height, width)

        Returns:
            List of stars with pixel coordinates within image bounds
        """
        try:
            matched_stars = []
            height, width = image_shape

            for star in stars:
                try:
                    # Get star coordinates
                    ra = star["ra"]
                    dec = star["dec"]

                    # Convert to pixel coordinates
                    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
                    pixel_x, pixel_y = wcs.world_to_pixel(coord)

                    # Check if star is within image bounds (with small margin)
                    margin = 5  # pixels
                    if (
                        -margin <= pixel_x <= width + margin
                        and -margin <= pixel_y <= height + margin
                    ):
                        # Add pixel coordinates to star data
                        matched_star = star.copy()
                        matched_star.update(
                            {
                                "pixel_x": float(pixel_x),
                                "pixel_y": float(pixel_y),
                                "in_image": (
                                    0 <= pixel_x <= width and 0 <= pixel_y <= height
                                ),
                            }
                        )

                        matched_stars.append(matched_star)

                except Exception as e:
                    self.logger.warning(f"Error matching star to image: {e}")
                    continue

            self.logger.debug(f"Matched {len(matched_stars)} stars to image")
            return matched_stars

        except Exception as e:
            self.logger.error(f"Error matching stars to image: {e}")
            raise

    def calculate_star_magnitudes(
        self, stars: list[dict[str, Any]], filter_band: str
    ) -> list[float]:
        """Calculate magnitudes for stars in specified filter band.

        Args:
            stars: List of star catalog entries
            filter_band: Filter band ('G', 'V', 'R', 'B', etc.)

        Returns:
            List of magnitudes, NaN for missing values
        """
        try:
            magnitudes = []

            for star in stars:
                mag = self._get_magnitude_for_filter(star, filter_band)
                magnitudes.append(mag)

            return magnitudes

        except Exception as e:
            self.logger.error(f"Error calculating magnitudes for {filter_band}: {e}")
            raise

    def cross_match_catalogs(
        self,
        stars1: list[dict[str, Any]],
        stars2: list[dict[str, Any]],
        tolerance: float = 1.0,
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Cross-match stars between two catalogs.

        Args:
            stars1: First catalog star list
            stars2: Second catalog star list
            tolerance: Matching tolerance in arcseconds

        Returns:
            List of matched star pairs
        """
        try:
            matches = []

            # Create SkyCoord objects for efficient matching
            coords1 = SkyCoord(
                ra=[s["ra"] for s in stars1] * u.deg,
                dec=[s["dec"] for s in stars1] * u.deg,
            )
            coords2 = SkyCoord(
                ra=[s["ra"] for s in stars2] * u.deg,
                dec=[s["dec"] for s in stars2] * u.deg,
            )

            # Perform cross-match
            idx, d2d, _ = coords1.match_to_catalog_sky(coords2)

            # Filter by tolerance
            good_matches = d2d.arcsec < tolerance

            for i, (j, distance) in enumerate(zip(idx, d2d, strict=False)):
                if good_matches[i]:
                    match_pair = (stars1[i].copy(), stars2[j].copy())
                    match_pair[0]["match_distance_arcsec"] = float(distance.arcsec)
                    match_pair[1]["match_distance_arcsec"] = float(distance.arcsec)
                    matches.append(match_pair)

            self.logger.debug(f'Found {len(matches)} cross-matches within {tolerance}"')
            return matches

        except Exception as e:
            self.logger.error(f"Error cross-matching catalogs: {e}")
            raise

    # Private methods for catalog queries and caching

    def _init_cache_db(self):
        """Initialize the cache database."""
        try:
            with sqlite3.connect(self.cache_db) as conn:
                cursor = conn.cursor()

                # Create tables if they don't exist
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS star_regions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ra REAL NOT NULL,
                        dec REAL NOT NULL,
                        radius REAL NOT NULL,
                        query_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        star_count INTEGER NOT NULL,
                        data TEXT NOT NULL
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS individual_stars (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        catalog TEXT NOT NULL,
                        star_id TEXT NOT NULL,
                        ra REAL NOT NULL,
                        dec REAL NOT NULL,
                        data TEXT NOT NULL,
                        cache_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(catalog, star_id)
                    )
                """)

                # Create indexes for efficient queries
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_regions_radec
                    ON star_regions (ra, dec, radius)
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_stars_id
                    ON individual_stars (catalog, star_id)
                """)

        except Exception as e:
            self.logger.error(f"Error initializing cache database: {e}")

    def _query_cache(
        self, ra: float, dec: float, radius: float
    ) -> list[dict[str, Any]] | None:
        """Query cached star data for a region."""
        try:
            # Check for existing cached results with some tolerance
            tolerance = 0.001  # degrees

            with sqlite3.connect(self.cache_db) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT data FROM star_regions
                    WHERE ABS(ra - ?) < ? AND ABS(dec - ?) < ? AND radius >= ?
                    ORDER BY query_time DESC LIMIT 1
                """,
                    (ra, tolerance, dec, tolerance, radius),
                )

                result = cursor.fetchone()
                if result:
                    return json.loads(result[0])

        except Exception as e:
            self.logger.debug(f"Cache query failed: {e}")

        return None

    def _cache_results(
        self, ra: float, dec: float, radius: float, stars: list[dict[str, Any]]
    ):
        """Cache star query results."""
        try:
            with sqlite3.connect(self.cache_db) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO star_regions (ra, dec, radius, star_count, data)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (ra, dec, radius, len(stars), json.dumps(stars)),
                )

                # Clean up old cache entries (keep last 1000)
                cursor.execute("""
                    DELETE FROM star_regions
                    WHERE id NOT IN (
                        SELECT id FROM star_regions
                        ORDER BY query_time DESC LIMIT 1000
                    )
                """)

        except Exception as e:
            self.logger.debug(f"Cache storage failed: {e}")

    def _query_catalog(
        self, catalog: str, ra: float, dec: float, radius: float
    ) -> list[dict[str, Any]]:
        """Query a specific catalog for stars in region."""
        # This is a simplified mock implementation
        # In a real implementation, this would make HTTP requests to catalog services

        # Generate mock stars for demonstration
        mock_stars = self._generate_mock_stars(catalog, ra, dec, radius)
        return mock_stars

    def _generate_mock_stars(
        self, catalog: str, ra: float, dec: float, radius: float
    ) -> list[dict[str, Any]]:
        """Generate mock star data for testing purposes."""
        # Number of stars roughly proportional to area and typical star density
        area_sq_deg = np.pi * (radius / 3600.0) ** 2
        n_stars = max(1, int(area_sq_deg * 1000))  # ~1000 stars per sq degree
        n_stars = min(n_stars, self.config.max_stars_per_query)

        # Generate random positions within radius
        np.random.seed(int(ra * 1000 + dec * 1000))  # Reproducible for same coordinates

        # Convert radius to degrees
        radius_deg = radius / 3600.0

        # Generate random positions in circular region
        angles = np.random.uniform(0, 2 * np.pi, n_stars)
        distances = np.sqrt(np.random.uniform(0, 1, n_stars)) * radius_deg

        ra_offsets = distances * np.cos(angles) / np.cos(np.radians(dec))
        dec_offsets = distances * np.sin(angles)

        stars = []
        for i in range(n_stars):
            star_ra = ra + ra_offsets[i]
            star_dec = dec + dec_offsets[i]

            # Generate magnitude
            mag = np.random.uniform(10, self.config.magnitude_limit)

            # Create star entry with catalog-specific fields
            star = {
                "ra": float(star_ra),
                "dec": float(star_dec),
                "catalog": catalog,
                "id": f"{catalog}_{i:06d}",
                "magnitude": float(mag),
            }

            # Add catalog-specific fields
            if catalog == "gaia":
                star.update(
                    {
                        "g_mag": float(mag),
                        "bp_mag": float(mag + np.random.normal(0, 0.1)),
                        "rp_mag": float(mag + np.random.normal(0, 0.1)),
                        "parallax": float(np.random.normal(0, 1)),
                        "pmra": float(np.random.normal(0, 5)),
                        "pmdec": float(np.random.normal(0, 5)),
                    }
                )
            elif catalog == "tycho2":
                star.update(
                    {
                        "v_mag": float(mag),
                        "bt_mag": float(mag + np.random.normal(0, 0.2)),
                        "vt_mag": float(mag + np.random.normal(0, 0.2)),
                    }
                )
            elif catalog == "usnob1":
                star.update(
                    {
                        "b1_mag": float(mag + np.random.normal(0, 0.3)),
                        "r1_mag": float(mag),
                        "b2_mag": float(mag + np.random.normal(0, 0.3)),
                        "r2_mag": float(mag + np.random.normal(0, 0.1)),
                    }
                )

            stars.append(star)

        return stars

    def _deduplicate_stars(self, stars: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove duplicate stars from multiple catalogs."""
        if len(stars) <= 1:
            return stars

        # Group by catalog
        catalog_groups = {}
        for star in stars:
            catalog = star.get("catalog", "unknown")
            if catalog not in catalog_groups:
                catalog_groups[catalog] = []
            catalog_groups[catalog].append(star)

        # If only one catalog, return as-is
        if len(catalog_groups) == 1:
            return stars

        # Cross-match between catalogs to find duplicates
        unique_stars = []
        processed_coords = set()

        for star in stars:
            # Round coordinates to avoid floating point issues
            coord_key = (
                round(star["ra"] / self.config.coordinate_precision)
                * self.config.coordinate_precision,
                round(star["dec"] / self.config.coordinate_precision)
                * self.config.coordinate_precision,
            )

            if coord_key not in processed_coords:
                unique_stars.append(star)
                processed_coords.add(coord_key)

        return unique_stars

    def _apply_magnitude_filter(
        self, stars: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Filter stars by magnitude limit."""
        filtered = []
        for star in stars:
            mag = star.get("magnitude", star.get("g_mag", star.get("v_mag", 99)))
            if mag <= self.config.magnitude_limit:
                filtered.append(star)

        return filtered

    def _get_magnitude_for_filter(
        self, star: dict[str, Any], filter_band: str
    ) -> float:
        """Get magnitude for specific filter band."""
        filter_band = filter_band.upper()

        # Map filter bands to catalog fields
        magnitude_mappings = {
            "G": ["g_mag", "magnitude"],
            "V": ["v_mag", "vt_mag", "magnitude"],
            "R": ["r_mag", "r1_mag", "r2_mag", "rp_mag"],
            "B": ["b_mag", "b1_mag", "b2_mag", "bt_mag", "bp_mag"],
            "I": ["i_mag"],
            "J": ["j_mag"],
            "H": ["h_mag"],
            "K": ["k_mag", "ks_mag"],
        }

        # Try to find magnitude for the requested filter
        if filter_band in magnitude_mappings:
            for mag_field in magnitude_mappings[filter_band]:
                if mag_field in star and star[mag_field] is not None:
                    return float(star[mag_field])

        # Fallback to generic magnitude
        for fallback_field in ["magnitude", "g_mag", "v_mag"]:
            if fallback_field in star and star[fallback_field] is not None:
                return float(star[fallback_field])

        return float("nan")

    def _query_star_by_id(self, catalog: str, star_id: str) -> dict[str, Any] | None:
        """Query specific star by ID (mock implementation)."""
        # This would make actual catalog queries in a real implementation
        return None

    def _get_cached_star_by_id(
        self, star_id: str, catalog: str
    ) -> dict[str, Any] | None:
        """Get cached star by ID."""
        try:
            with sqlite3.connect(self.cache_db) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT data FROM individual_stars
                    WHERE catalog = ? AND star_id = ?
                """,
                    (catalog, star_id),
                )

                result = cursor.fetchone()
                if result:
                    return json.loads(result[0])

        except Exception as e:
            self.logger.debug(f"Cached star query failed: {e}")

        return None

    def _cache_star(self, star: dict[str, Any]):
        """Cache individual star data."""
        try:
            catalog = star.get("catalog")
            star_id = star.get("id")
            ra = star.get("ra")
            dec = star.get("dec")

            if not all([catalog, star_id, ra is not None, dec is not None]):
                return

            with sqlite3.connect(self.cache_db) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO individual_stars
                    (catalog, star_id, ra, dec, data)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (catalog, star_id, ra, dec, json.dumps(star)),
                )

        except Exception as e:
            self.logger.debug(f"Star caching failed: {e}")

    def clear_cache(self):
        """Clear all cached data."""
        try:
            with sqlite3.connect(self.cache_db) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM star_regions")
                cursor.execute("DELETE FROM individual_stars")

            self.logger.info("Cleared star catalog cache")

        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        try:
            with sqlite3.connect(self.cache_db) as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT COUNT(*) FROM star_regions")
                region_count = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM individual_stars")
                star_count = cursor.fetchone()[0]

                cursor.execute("SELECT SUM(star_count) FROM star_regions")
                total_cached_stars = cursor.fetchone()[0] or 0

                return {
                    "cached_regions": region_count,
                    "cached_individual_stars": star_count,
                    "total_cached_stars": total_cached_stars,
                    "cache_size_mb": self.cache_db.stat().st_size / (1024 * 1024),
                }

        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}")
            return {}
