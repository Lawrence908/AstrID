"""Enhanced WCS (World Coordinate System) processing for astronomical observations."""

import logging

import numpy as np
from astropy import units as u
from astropy.coordinates import FK5, ICRS, Galactic, SkyCoord
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel

logger = logging.getLogger(__name__)


class WCSProcessor:
    """Enhanced WCS processor with coordinate transformations and validation."""

    def __init__(self):
        """Initialize WCS processor."""
        self.logger = logging.getLogger(__name__)

        # Supported coordinate systems
        self.coordinate_systems = {
            "icrs": ICRS(),
            "fk5": FK5(),
            "galactic": Galactic(),
        }

    def pixel_to_world_coordinates(
        self, pixel_coords: np.ndarray, wcs: WCS
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert pixel coordinates to world coordinates.

        Args:
            pixel_coords: Array of pixel coordinates [[x1, y1], [x2, y2], ...]
            wcs: WCS object

        Returns:
            Tuple of (ra_array, dec_array) in degrees

        Raises:
            ValueError: If pixel coordinates are invalid or WCS is invalid
        """
        try:
            # Validate inputs
            if pixel_coords.size == 0:
                raise ValueError("Empty pixel coordinates array")

            if not self.validate_wcs_solution(wcs):
                raise ValueError("Invalid WCS solution")

            # Ensure pixel_coords is 2D
            if pixel_coords.ndim == 1:
                pixel_coords = pixel_coords.reshape(1, -1)

            # Convert to world coordinates (astropy returns tuple of arrays)
            x_pixels = pixel_coords[:, 0]
            y_pixels = pixel_coords[:, 1]
            ra, dec = wcs.pixel_to_world_values(x_pixels, y_pixels)

            self.logger.debug(
                f"Converted {len(pixel_coords)} pixel coordinates to world coordinates"
            )

            return ra, dec

        except Exception as e:
            self.logger.error(f"Error converting pixel to world coordinates: {e}")
            raise

    def world_to_pixel_coordinates(
        self, world_coords: tuple[np.ndarray, np.ndarray], wcs: WCS
    ) -> np.ndarray:
        """Convert world coordinates to pixel coordinates.

        Args:
            world_coords: Tuple of (ra_array, dec_array) in degrees
            wcs: WCS object

        Returns:
            Array of pixel coordinates [[x1, y1], [x2, y2], ...]

        Raises:
            ValueError: If world coordinates are invalid or WCS is invalid
        """
        try:
            ra, dec = world_coords

            # Validate inputs
            if len(ra) != len(dec):
                raise ValueError("RA and Dec arrays must have same length")

            if not self.validate_wcs_solution(wcs):
                raise ValueError("Invalid WCS solution")

            # Create SkyCoord object
            coords = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")

            # Convert to pixel coordinates
            pixel_coords = skycoord_to_pixel(coords, wcs)

            # Return as array of [x, y] pairs
            result = np.column_stack([pixel_coords[0], pixel_coords[1]])

            self.logger.debug(
                f"Converted {len(ra)} world coordinates to pixel coordinates"
            )

            return result

        except Exception as e:
            self.logger.error(f"Error converting world to pixel coordinates: {e}")
            raise

    def validate_wcs_solution(self, wcs: WCS) -> bool:
        """Validate WCS solution quality and completeness.

        Args:
            wcs: WCS object to validate

        Returns:
            bool: True if WCS solution is valid and usable
        """
        try:
            # Check if WCS object exists and has necessary attributes
            if wcs is None:
                self.logger.error("WCS object is None")
                return False

            # Check for minimum required dimensions
            if wcs.naxis < 2:
                self.logger.error(
                    f"WCS must have at least 2 dimensions, got {wcs.naxis}"
                )
                return False

            # Check for required WCS arrays/values
            try:
                ctype = getattr(wcs.wcs, "ctype", None)
                crval = getattr(wcs.wcs, "crval", None)
                crpix = getattr(wcs.wcs, "crpix", None)
                cd = getattr(wcs.wcs, "cd", None)
                cdelt = getattr(wcs.wcs, "cdelt", None)

                if ctype is None or len(ctype) < 2 or not ctype[0] or not ctype[1]:
                    self.logger.error("Missing or invalid WCS keyword: CTYPE1/CTYPE2")
                    return False

                if crval is None or len(crval) < 2:
                    self.logger.error("Missing or invalid WCS keyword: CRVAL1/CRVAL2")
                    return False

                if crpix is None or len(crpix) < 2:
                    self.logger.error("Missing or invalid WCS keyword: CRPIX1/CRPIX2")
                    return False

                # Require either CD matrix or CDELT scales
                has_cd = cd is not None and np.array(cd).shape == (2, 2)
                has_cdelt = cdelt is not None and len(cdelt) >= 2
                if not (has_cd or has_cdelt):
                    self.logger.error(
                        "Missing WCS scale keywords: need CD matrix or CDELT1/2"
                    )
                    return False
            except Exception:
                self.logger.error("Error accessing WCS core keywords")
                return False

            # Check coordinate types are celestial
            ctype1 = wcs.wcs.ctype[0]
            ctype2 = wcs.wcs.ctype[1]

            celestial_types = ["RA", "DEC", "GLON", "GLAT", "ELON", "ELAT"]
            if not any(ct in ctype1.upper() for ct in celestial_types) or not any(
                ct in ctype2.upper() for ct in celestial_types
            ):
                self.logger.error(f"Non-celestial coordinate types: {ctype1}, {ctype2}")
                return False

            # Test coordinate transformation
            try:
                # Test transformation at reference pixel
                ref_pixel = [wcs.wcs.crpix[0], wcs.wcs.crpix[1]]
                ra_ref, dec_ref = wcs.pixel_to_world_values(ref_pixel[0], ref_pixel[1])
                x_back, y_back = wcs.world_to_pixel_values(ra_ref, dec_ref)

                # Check round-trip accuracy (should be within 1e-6 pixels)
                pixel_diff = np.abs(np.array([x_back, y_back]) - np.array(ref_pixel))
                if np.any(pixel_diff > 1e-6):
                    self.logger.warning(
                        f"WCS round-trip test failed: pixel difference {pixel_diff}"
                    )
                    return False

            except Exception as e:
                self.logger.error(f"WCS coordinate transformation test failed: {e}")
                return False

            # Check for reasonable coordinate values
            ra_ref = wcs.wcs.crval[0]
            dec_ref = wcs.wcs.crval[1]

            if not (0 <= ra_ref <= 360):
                self.logger.error(f"Invalid reference RA: {ra_ref}")
                return False

            if not (-90 <= dec_ref <= 90):
                self.logger.error(f"Invalid reference Dec: {dec_ref}")
                return False

            self.logger.debug("WCS validation passed")
            return True

        except Exception as e:
            self.logger.error(f"WCS validation error: {e}")
            return False

    def calculate_sky_region_bounds(
        self, wcs: WCS, image_shape: tuple[int, int]
    ) -> dict:
        """Calculate sky region bounds for an image with given WCS.

        Args:
            wcs: WCS object
            image_shape: Image shape as (height, width)

        Returns:
            dict: Sky region bounds and statistics

        Raises:
            ValueError: If WCS or image shape is invalid
        """
        try:
            if not self.validate_wcs_solution(wcs):
                raise ValueError("Invalid WCS solution")

            height, width = image_shape

            # Define corner pixels (0-indexed, but WCS expects 1-indexed)
            corners = np.array(
                [
                    [0.5, 0.5],  # Bottom-left
                    [width - 0.5, 0.5],  # Bottom-right
                    [width - 0.5, height - 0.5],  # Top-right
                    [0.5, height - 0.5],  # Top-left
                ]
            )

            # Convert corners to world coordinates
            ra, dec = self.pixel_to_world_coordinates(corners, wcs)

            # Calculate bounds, handling RA wraparound
            ra_min, ra_max = np.min(ra), np.max(ra)
            dec_min, dec_max = np.min(dec), np.max(dec)

            # Handle RA wraparound at 0/360 degrees
            ra_span = ra_max - ra_min
            if ra_span > 180:  # Likely wraparound
                # Recalculate with shifted coordinates
                ra_shifted = np.where(ra < 180, ra + 360, ra)
                ra_min_shifted = np.min(ra_shifted)
                ra_max_shifted = np.max(ra_shifted)

                if ra_max_shifted - ra_min_shifted < ra_span:
                    ra_min = ra_min_shifted % 360
                    ra_max = ra_max_shifted % 360

            # Calculate center coordinates
            center_pixel = np.array([[width / 2, height / 2]])
            center_ra, center_dec = self.pixel_to_world_coordinates(center_pixel, wcs)

            # Calculate pixel scale (arcsec/pixel)
            pixel_scale = self._calculate_pixel_scale(wcs)

            # Calculate area
            area_sq_deg = (
                np.abs(ra_max - ra_min)
                * np.abs(dec_max - dec_min)
                * np.cos(np.radians(np.mean([dec_min, dec_max])))
            )

            bounds = {
                "ra_min": float(ra_min),
                "ra_max": float(ra_max),
                "dec_min": float(dec_min),
                "dec_max": float(dec_max),
                "center_ra": float(center_ra[0]),
                "center_dec": float(center_dec[0]),
                "width_deg": float(ra_max - ra_min),
                "height_deg": float(dec_max - dec_min),
                "area_sq_deg": float(area_sq_deg),
                "pixel_scale_arcsec": pixel_scale,
                "image_shape": image_shape,
                "corners_ra": ra.tolist(),
                "corners_dec": dec.tolist(),
            }

            self.logger.debug(
                f"Calculated sky region bounds: "
                f"RA {ra_min:.3f}-{ra_max:.3f}, Dec {dec_min:.3f}-{dec_max:.3f}"
            )

            return bounds

        except Exception as e:
            self.logger.error(f"Error calculating sky region bounds: {e}")
            raise

    def transform_coordinates(
        self, coords: tuple[float, float], from_system: str, to_system: str
    ) -> tuple[float, float]:
        """Transform coordinates between different celestial coordinate systems.

        Args:
            coords: Coordinates as (ra, dec) or (lon, lat) in degrees
            from_system: Source coordinate system ('icrs', 'fk5', 'galactic')
            to_system: Target coordinate system ('icrs', 'fk5', 'galactic')

        Returns:
            Tuple of transformed coordinates in degrees

        Raises:
            ValueError: If coordinate systems are not supported
        """
        try:
            from_system = from_system.lower()
            to_system = to_system.lower()

            # Validate coordinate systems
            if from_system not in self.coordinate_systems:
                raise ValueError(f"Unsupported source coordinate system: {from_system}")
            if to_system not in self.coordinate_systems:
                raise ValueError(f"Unsupported target coordinate system: {to_system}")

            # No transformation needed if systems are the same
            if from_system == to_system:
                return coords

            ra, dec = coords

            # Create SkyCoord in source frame
            if from_system == "icrs":
                coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
            elif from_system == "fk5":
                coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="fk5")
            elif from_system == "galactic":
                coord = SkyCoord(l=ra * u.deg, b=dec * u.deg, frame="galactic")

            # Transform to target frame
            if to_system == "icrs":
                transformed = coord.icrs
                result = (float(transformed.ra.deg), float(transformed.dec.deg))
            elif to_system == "fk5":
                transformed = coord.fk5
                result = (float(transformed.ra.deg), float(transformed.dec.deg))
            elif to_system == "galactic":
                transformed = coord.galactic
                result = (float(transformed.l.deg), float(transformed.b.deg))

            self.logger.debug(
                f"Transformed coordinates from {from_system} to {to_system}: "
                f"{coords} -> {result}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Error transforming coordinates: {e}")
            raise

    def assess_wcs_quality(
        self, wcs: WCS, image_shape: tuple[int, int] | None = None
    ) -> dict:
        """Assess WCS solution quality and provide metrics.

        Args:
            wcs: WCS object to assess
            image_shape: Optional image shape for additional checks

        Returns:
            dict: WCS quality assessment metrics
        """
        quality = {
            "is_valid": False,
            "completeness_score": 0.0,
            "accuracy_score": 0.0,
            "reliability_score": 0.0,
            "overall_score": 0.0,
            "issues": [],
            "recommendations": [],
        }

        try:
            # Basic validation
            quality["is_valid"] = self.validate_wcs_solution(wcs)
            if not quality["is_valid"]:
                quality["issues"].append("Failed basic WCS validation")
                return quality

            # Completeness assessment
            completeness_factors = []

            # Check for essential keywords
            essential_keywords = ["ctype", "crval", "crpix", "cdelt", "cd"]
            for keyword in essential_keywords:
                if hasattr(wcs.wcs, keyword) and getattr(wcs.wcs, keyword) is not None:
                    completeness_factors.append(1.0)
                else:
                    completeness_factors.append(0.0)

            quality["completeness_score"] = np.mean(completeness_factors)

            # Accuracy assessment (based on pixel scale consistency)
            try:
                pixel_scale = self._calculate_pixel_scale(wcs)
                if 0.01 < pixel_scale < 100:  # Reasonable range for most instruments
                    quality["accuracy_score"] = 1.0
                else:
                    quality["accuracy_score"] = 0.5
                    quality["issues"].append(
                        f"Unusual pixel scale: {pixel_scale:.3f} arcsec/pixel"
                    )
            except Exception:
                quality["accuracy_score"] = 0.0
                quality["issues"].append("Could not calculate pixel scale")

            # Reliability assessment (coordinate range checks)
            try:
                if image_shape:
                    bounds = self.calculate_sky_region_bounds(wcs, image_shape)

                    # Check for reasonable RA/Dec ranges
                    if (
                        0 <= bounds["ra_min"] <= 360
                        and 0 <= bounds["ra_max"] <= 360
                        and -90 <= bounds["dec_min"] <= 90
                        and -90 <= bounds["dec_max"] <= 90
                    ):
                        quality["reliability_score"] = 1.0
                    else:
                        quality["reliability_score"] = 0.0
                        quality["issues"].append(
                            "Coordinates outside valid celestial ranges"
                        )
                else:
                    quality["reliability_score"] = (
                        0.8  # Partial score without full testing
                    )
            except Exception:
                quality["reliability_score"] = 0.0
                quality["issues"].append("Could not verify coordinate ranges")

            # Overall score
            quality["overall_score"] = np.mean(
                [
                    quality["completeness_score"],
                    quality["accuracy_score"],
                    quality["reliability_score"],
                ]
            )

            # Generate recommendations
            if quality["completeness_score"] < 0.8:
                quality["recommendations"].append("WCS solution is incomplete")
            if quality["accuracy_score"] < 0.8:
                quality["recommendations"].append(
                    "Consider re-calibrating astrometric solution"
                )
            if quality["reliability_score"] < 0.8:
                quality["recommendations"].append(
                    "Verify coordinate system and reference frame"
                )

            self.logger.debug(
                f"WCS quality assessment: overall score {quality['overall_score']:.2f}"
            )

        except Exception as e:
            self.logger.error(f"Error assessing WCS quality: {e}")
            quality["issues"].append(f"Assessment failed: {str(e)}")

        return quality

    def _calculate_pixel_scale(self, wcs: WCS) -> float:
        """Calculate pixel scale in arcseconds per pixel.

        Args:
            wcs: WCS object

        Returns:
            float: Pixel scale in arcseconds per pixel
        """
        try:
            # Use the determinant of the CD matrix or CDELT values
            if hasattr(wcs.wcs, "cd") and wcs.wcs.cd is not None:
                cd_matrix = wcs.wcs.cd
                pixel_scale = (
                    np.sqrt(np.abs(np.linalg.det(cd_matrix))) * 3600
                )  # Convert to arcsec
            elif hasattr(wcs.wcs, "cdelt") and wcs.wcs.cdelt is not None:
                pixel_scale = (
                    np.sqrt(np.abs(wcs.wcs.cdelt[0] * wcs.wcs.cdelt[1])) * 3600
                )
            else:
                # Fallback: calculate from coordinate transformation
                center = [wcs.wcs.crpix[0], wcs.wcs.crpix[1]]
                offset = [center[0] + 1, center[1]]

                ra1, dec1 = wcs.pixel_to_world_values(center[0], center[1])
                ra2, dec2 = wcs.pixel_to_world_values(offset[0], offset[1])

                # Calculate angular separation
                c1 = SkyCoord(ra=ra1, dec=dec1, unit="deg")
                c2 = SkyCoord(ra=ra2, dec=dec2, unit="deg")
                pixel_scale = c1.separation(c2).arcsec

            return float(pixel_scale)

        except Exception as e:
            self.logger.error(f"Error calculating pixel scale: {e}")
            raise
