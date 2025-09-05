"""FITS I/O adapter for astronomical image processing."""

import logging
from typing import Any

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

logger = logging.getLogger(__name__)


class FITSProcessor:
    """FITS file processor for astronomical images."""

    def __init__(self):
        """Initialize FITS processor."""
        self.logger = logging.getLogger(__name__)

    def read_fits(self, file_path: str) -> tuple[np.ndarray, WCS, dict[str, Any]]:
        """Read FITS file and extract image data, WCS, and metadata.

        Args:
            file_path: Path to FITS file

        Returns:
            Tuple of (image_data, wcs, metadata)
        """
        try:
            with fits.open(file_path) as hdul:
                # Extract primary image data
                image_data = hdul[0].data

                # Extract WCS information
                wcs = WCS(hdul[0].header)

                # Extract metadata
                metadata = self._extract_metadata(hdul[0].header)

                self.logger.info(f"Successfully read FITS file: {file_path}")
                return image_data, wcs, metadata

        except Exception as e:
            self.logger.error(f"Error reading FITS file {file_path}: {e}")
            raise

    def _extract_metadata(self, header: fits.Header) -> dict[str, Any]:
        """Extract metadata from FITS header.

        Args:
            header: FITS header object

        Returns:
            Dictionary of metadata
        """
        metadata = {}

        # Standard FITS keywords
        standard_keys = [
            "NAXIS1",
            "NAXIS2",
            "NAXIS3",  # Image dimensions
            "CRPIX1",
            "CRPIX2",  # Reference pixel
            "CRVAL1",
            "CRVAL2",  # Reference coordinates
            "CDELT1",
            "CDELT2",  # Pixel scale
            "CTYPE1",
            "CTYPE2",  # Coordinate types
            "EQUINOX",
            "RADECSYS",  # Coordinate system
            "EXPTIME",
            "EXPOSURE",  # Exposure time
            "FILTER",
            "BAND",  # Filter/band
            "AIRMASS",
            "SEEING",  # Observation conditions
            "OBSERVAT",
            "TELESCOP",  # Instrument info
            "DATE-OBS",
            "TIME-OBS",  # Observation time
            "RA",
            "DEC",  # Target coordinates
        ]

        for key in standard_keys:
            if key in header:
                try:
                    metadata[key] = header[key]
                except Exception:
                    # Skip problematic values
                    continue

        # Extract comment and history
        if "COMMENT" in header:
            metadata["comments"] = [str(comment) for comment in header["COMMENT"]]

        if "HISTORY" in header:
            metadata["history"] = [str(hist) for hist in header["HISTORY"]]

        return metadata

    def get_coordinate_range(self, wcs: WCS) -> dict[str, tuple[float, float]]:
        """Get coordinate range from WCS.

        Args:
            wcs: WCS object

        Returns:
            Dictionary with coordinate ranges
        """
        try:
            # Get image dimensions
            height, width = wcs.pixel_shape

            # Convert corner pixels to world coordinates
            corners = [
                (0, 0),  # Lower left
                (width, 0),  # Lower right
                (0, height),  # Upper left
                (width, height),  # Upper right
            ]

            world_coords = wcs.all_pix2world(corners, 1)

            # Extract RA and Dec ranges
            ra_coords = world_coords[:, 0]
            dec_coords = world_coords[:, 1]

            # Handle RA wrapping
            ra_coords = np.where(ra_coords < 0, ra_coords + 360, ra_coords)

            return {
                "ra": (np.min(ra_coords), np.max(ra_coords)),
                "dec": (np.min(dec_coords), np.max(dec_coords)),
            }

        except Exception as e:
            self.logger.error(f"Error getting coordinate range: {e}")
            raise

    def extract_star_catalog(self, file_path: str) -> np.ndarray | None:
        """Extract star catalog from FITS file if present.

        Args:
            file_path: Path to FITS file

        Returns:
            Star catalog data or None if not found
        """
        try:
            with fits.open(file_path) as hdul:
                # Look for star catalog HDU
                if "STAR_CATALOG" in hdul:
                    catalog_data = hdul["STAR_CATALOG"].data
                    self.logger.info(
                        f"Extracted star catalog with {len(catalog_data)} stars"
                    )
                    return catalog_data
                else:
                    self.logger.info("No star catalog found in FITS file")
                    return None

        except Exception as e:
            self.logger.error(f"Error extracting star catalog: {e}")
            return None

    def extract_pixel_mask(self, file_path: str) -> np.ndarray | None:
        """Extract pixel mask from FITS file if present.

        Args:
            file_path: Path to FITS file

        Returns:
            Pixel mask data or None if not found
        """
        try:
            with fits.open(file_path) as hdul:
                # Look for pixel mask HDU
                if "pixel_mask" in hdul:
                    mask_data = hdul["pixel_mask"].data
                    self.logger.info(
                        f"Extracted pixel mask with shape {mask_data.shape}"
                    )
                    return mask_data
                else:
                    self.logger.info("No pixel mask found in FITS file")
                    return None

        except Exception as e:
            self.logger.error(f"Error extracting pixel mask: {e}")
            return None

    def get_stars_in_image(
        self,
        wcs: WCS,
        catalog_data: np.ndarray,
        coord_range: dict[str, tuple[float, float]],
    ) -> list[dict[str, Any]]:
        """Get stars that fall within the image boundaries.

        Args:
            wcs: WCS object
            catalog_data: Star catalog data
            coord_range: Coordinate range dictionary

        Returns:
            List of stars within image boundaries
        """
        stars_in_image = []

        try:
            ra_min, ra_max = coord_range["ra"]
            dec_min, dec_max = coord_range["dec"]

            for star in catalog_data:
                # Extract coordinates (assuming 2MASS format)
                if "_2MASS" in star.colnames:
                    coords_str = star["_2MASS"]
                    ra, dec = self._parse_2mass_coordinates(coords_str)

                    # Check if star is within image boundaries
                    if (ra_min <= ra <= ra_max) and (dec_min <= dec <= dec_max):
                        # Convert to pixel coordinates
                        pixel_coords = wcs.world_to_pixel(
                            SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
                        )

                        star_info = {
                            "catalog_data": star,
                            "ra": ra,
                            "dec": dec,
                            "pixel_x": pixel_coords[0],
                            "pixel_y": pixel_coords[1],
                        }

                        # Add magnitude if available
                        if "Jmag" in star.colnames:
                            star_info["j_magnitude"] = star["Jmag"]

                        stars_in_image.append(star_info)

            self.logger.info(
                f"Found {len(stars_in_image)} stars within image boundaries"
            )
            return stars_in_image

        except Exception as e:
            self.logger.error(f"Error finding stars in image: {e}")
            return []

    def _parse_2mass_coordinates(self, coords_str: str) -> tuple[float, float]:
        """Parse 2MASS coordinate string.

        Args:
            coords_str: 2MASS coordinate string (e.g., "12345678-1234567")

        Returns:
            Tuple of (ra, dec) in degrees
        """
        try:
            # Handle negative declination
            if "-" in coords_str and coords_str.count("-") == 2:
                # Format: "12345678-1234567"
                ra_str, dec_str = coords_str.split("-", 1)
                dec_str = "-" + dec_str
            elif "+" in coords_str:
                # Format: "12345678+1234567"
                ra_str, dec_str = coords_str.split("+")
                dec_str = "+" + dec_str
            else:
                # Assume positive declination
                ra_str = coords_str[:8]
                dec_str = coords_str[8:]

            # Parse RA (HHMMSS.SS format)
            ra_h = float(ra_str[:2])
            ra_m = float(ra_str[2:4])
            ra_s = float(ra_str[4:])
            ra_deg = (ra_h + ra_m / 60 + ra_s / 3600) * 15  # Convert to degrees

            # Parse Dec (DDMMSS.S format)
            dec_sign = 1 if dec_str[0] == "+" else -1
            dec_d = float(dec_str[1:3])
            dec_m = float(dec_str[3:5])
            dec_s = float(dec_str[5:])
            dec_deg = dec_sign * (dec_d + dec_m / 60 + dec_s / 3600)

            return ra_deg, dec_deg

        except Exception as e:
            self.logger.error(f"Error parsing 2MASS coordinates {coords_str}: {e}")
            raise

    def create_circular_mask(
        self,
        height: int,
        width: int,
        center: tuple[int, int] | None = None,
        radius: int | None = None,
    ) -> np.ndarray:
        """Create a circular mask.

        Args:
            height: Image height
            width: Image width
            center: Center coordinates (x, y), defaults to image center
            radius: Circle radius, defaults to smallest distance to edge

        Returns:
            Boolean mask array
        """
        if center is None:
            center = (width // 2, height // 2)

        if radius is None:
            radius = min(center[0], center[1], width - center[0], height - center[1])

        y, x = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

        return dist_from_center <= radius

    def save_fits(
        self,
        data: np.ndarray,
        file_path: str,
        header: dict[str, Any] | None = None,
        overwrite: bool = True,
    ) -> None:
        """Save data as FITS file.

        Args:
            data: Data array to save
            file_path: Output file path
            header: Optional header information
            overwrite: Whether to overwrite existing file
        """
        try:
            # Create primary HDU
            primary_hdu = fits.PrimaryHDU(data)

            # Add header information if provided
            if header:
                for key, value in header.items():
                    if len(key) <= 8:  # FITS keyword length limit
                        primary_hdu.header[key] = value

            # Save to file
            primary_hdu.writeto(file_path, overwrite=overwrite)
            self.logger.info(f"Saved FITS file: {file_path}")

        except Exception as e:
            self.logger.error(f"Error saving FITS file {file_path}: {e}")
            raise
