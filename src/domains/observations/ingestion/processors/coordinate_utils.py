"""Coordinate processing utilities adapted from proven dataGathering.py functions."""

import random

import astropy.units as u
import numpy as np
from astropy.coordinates import ICRS, SkyCoord
from astropy.wcs import WCS


class CoordinateProcessor:
    """Coordinate processing utilities for astronomical observations."""

    def __init__(self):
        """Initialize coordinate processor."""
        pass

    @staticmethod
    def get_random_coordinates(
        avoid_galactic_plane: bool = True,
    ) -> tuple[float, float]:
        """Generate random sky coordinates, optionally avoiding the galactic plane.

        Adapted from dataGathering.py getRandomCoordinates function.

        Args:
            avoid_galactic_plane: Whether to avoid the galactic plane. Default is True.

        Returns:
            Tuple containing the RA and Dec coordinates in degrees.
        """
        if avoid_galactic_plane:
            while True:
                ra = random.uniform(0, 360)
                # Limit dec upper and lower bound to avoid the "galactic plane"
                dec = random.uniform(-60, 60)
                coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
                galactic_coords = coords.galactic
                if (
                    abs(galactic_coords.b.deg) > 10
                ):  # Avoiding ±10 degrees around the galactic plane
                    break
        else:
            ra = random.uniform(0, 360)
            dec = random.uniform(-90, 90)

        return ra, dec

    @staticmethod
    def clean_dec_value(dec_value: str) -> float:
        """Clean declination value from string format to decimal degrees.

        Adapted from dataGathering.py cleanDecValue function.

        Args:
            dec_value: Declination value as string (e.g., "+45:30:15.5")

        Returns:
            Declination in decimal degrees
        """
        # Remove any leading/trailing whitespace
        dec_value = dec_value.strip()

        # Handle sexagesimal format (e.g., "+45:30:15.5")
        if ":" in dec_value:
            # Extract sign
            sign = 1 if dec_value[0] != "-" else -1
            dec_value = dec_value.lstrip("+-")

            # Split into components
            parts = dec_value.split(":")
            degrees = float(parts[0])
            minutes = float(parts[1]) if len(parts) > 1 else 0.0
            seconds = float(parts[2]) if len(parts) > 2 else 0.0

            # Convert to decimal degrees
            decimal_dec = sign * (degrees + minutes / 60.0 + seconds / 3600.0)
            return decimal_dec
        else:
            # Assume already in decimal format
            return float(dec_value)

    @staticmethod
    def get_coord_range_from_pixels(
        wcs: WCS,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Get the range of ICRS coordinates in the image.

        Adapted from dataGathering.py getCoordRangeFromPixels function.

        Args:
            wcs: WCS object of the image

        Returns:
            Dictionary containing the coordinates of the corners of the image
        """
        x_dim = wcs.pixel_shape[0]  # May need to swap x and y dim
        y_dim = wcs.pixel_shape[1]

        coord_range = {}

        coord_range["lower_left"] = wcs.all_pix2world([0], [0], 1)
        coord_range["lower_right"] = wcs.all_pix2world([x_dim], [0], 1)
        coord_range["upper_left"] = wcs.all_pix2world([0], [y_dim], 1)
        coord_range["upper_right"] = wcs.all_pix2world([x_dim], [y_dim], 1)

        return coord_range

    @staticmethod
    def get_stars_in_image(
        wcs: WCS, catalog_df, coord_range: dict[str, tuple[np.ndarray, np.ndarray]]
    ) -> list:
        """Get all the stars in the image.

        Adapted from dataGathering.py getStarsInImage function.

        Args:
            wcs: WCS object of the image
            catalog_df: DataFrame containing the star catalog
            coord_range: Dictionary containing the coordinates of the corners of the image

        Returns:
            List of stars in the image
        """
        # NOTE: X Max and min are reversed for some reason.. orientation of image in coord system...?
        x_max = coord_range["lower_left"][0]
        x_min = coord_range["lower_right"][0]

        y_min = coord_range["lower_left"][1]
        y_max = coord_range["upper_left"][1]

        stars_in_image = []

        for star in catalog_df.iterrows():
            # Use iloc to avoid deprecation warning
            rej = star[1].iloc[0]
            dej = star[1].iloc[1]

            if rej < x_max and rej > x_min:
                if dej < y_max and dej > y_min:
                    # Star is within bounds of image
                    stars_in_image.append(star)

        return stars_in_image

    @staticmethod
    def parse_star_coords(coords: str) -> str:
        """Parse the star coordinates.

        Adapted from dataGathering.py parseStarCoords function.

        Args:
            coords: The star coordinates

        Returns:
            The parsed star coordinates
        """
        if "-" in coords:
            rej, dej = coords.split("-")
            rej = rej[0:2] + "h" + rej[2:4] + "m" + rej[4:6] + "." + rej[6:] + "s"
            dej = "-" + dej[0:2] + "d" + dej[2:4] + "m" + dej[4:6] + "." + dej[6:] + "s"

        elif "+" in coords:
            rej, dej = coords.split("+")
            rej = rej[0:2] + "h" + rej[2:4] + "m" + rej[4:6] + "." + rej[6:] + "s"
            dej = "+" + dej[0:2] + "d" + dej[2:4] + "m" + dej[4:6] + "." + dej[6:] + "s"

        # Clean the declination value
        dej = CoordinateProcessor.clean_dec_value(dej)

        return rej + dej

    @staticmethod
    def get_pixel_coords_from_star(star, wcs: WCS) -> tuple[float, float]:
        """Get the pixel coordinates of a star from the catalog.

        Adapted from dataGathering.py getPixelCoordsFromStar function.

        Args:
            star: Series containing the star data
            wcs: WCS object of the image

        Returns:
            The pixel coordinates of the star

        Raises:
            ValueError: If coordinates cannot be parsed
        """
        star_coords = star[1]["_2MASS"]

        coords = CoordinateProcessor.parse_star_coords(star_coords)

        try:
            c = SkyCoord(coords, frame=ICRS)
        except ValueError as e:
            raise ValueError(f"Error parsing coordinates: {coords}") from e

        pixel_coords = wcs.world_to_pixel(c)
        return pixel_coords

    @staticmethod
    def create_circular_mask(
        h: int, w: int, center: tuple[int, int] = None, radius: float = None
    ) -> np.ndarray:
        """Create a circular mask.

        Adapted from dataGathering.py createCircularMask function.

        Args:
            h: The height of the mask
            w: The width of the mask
            center: The center of the circle (defaults to center of image)
            radius: The radius of the circle (defaults to smallest distance to edge)

        Returns:
            Boolean array representing the circular mask
        """
        if center is None:  # use the middle of the image
            center = (int(w / 2), int(h / 2))
        if (
            radius is None
        ):  # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w - center[0], h - center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

        mask = dist_from_center <= radius
        return mask

    @staticmethod
    def calculate_star_radius(
        jmag: float,
        min_jmag: float,
        max_jmag: float,
        min_radius: float,
        max_radius: float,
    ) -> float:
        """Calculate star radius based on magnitude.

        Adapted from dataGathering.py calculateRadius function.

        Args:
            jmag: J-band magnitude of the star
            min_jmag: Minimum J-magnitude in the catalog
            max_jmag: Maximum J-magnitude in the catalog
            min_radius: Minimum radius in pixels
            max_radius: Maximum radius in pixels

        Returns:
            Calculated radius for the star
        """
        # Normalize the jmag value (inverted - brighter stars are larger)
        if max_jmag == min_jmag:  # Avoid division by zero
            normalized_jmag = 0.5
        else:
            normalized_jmag = (max_jmag - jmag) / (max_jmag - min_jmag)

        # Scale the normalized value to the desired range of pixel sizes
        radius = min_radius + (normalized_jmag * (max_radius - min_radius))
        return radius

    @staticmethod
    def generate_dynamic_radius_params(wcs: WCS) -> dict[str, float]:
        """Generate dynamic radius parameters based on image dimensions.

        Adapted from dataGathering.py logic.

        Args:
            wcs: WCS object of the image

        Returns:
            Dictionary with min_radius and max_radius
        """
        x_dim = wcs.pixel_shape[0]
        y_dim = wcs.pixel_shape[1]

        min_radius = 1  # Minimum radius in pixels
        max_radius = min(x_dim, y_dim) * 0.005859375  # Fraction of smaller dimension

        return {"min_radius": min_radius, "max_radius": max_radius}

    @staticmethod
    def world_to_pixel_safe(
        ra: float, dec: float, wcs: WCS, frame: str = "icrs"
    ) -> tuple[float, float]:
        """Safely convert world coordinates to pixel coordinates.

        Args:
            ra: Right Ascension in degrees
            dec: Declination in degrees
            wcs: WCS object
            frame: Coordinate frame (default: 'icrs')

        Returns:
            Pixel coordinates (x, y)

        Raises:
            ValueError: If conversion fails
        """
        try:
            coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame=frame)
            x, y = wcs.world_to_pixel(coord)
            return float(x), float(y)
        except Exception as e:
            raise ValueError(
                f"Failed to convert coordinates ({ra}, {dec}) to pixels: {e}"
            ) from e

    @staticmethod
    def pixel_to_world_safe(
        x: float, y: float, wcs: WCS, frame: str = "icrs"
    ) -> tuple[float, float]:
        """Safely convert pixel coordinates to world coordinates.

        Args:
            x: X pixel coordinate
            y: Y pixel coordinate
            wcs: WCS object
            frame: Target coordinate frame (default: 'icrs')

        Returns:
            World coordinates (ra, dec) in degrees

        Raises:
            ValueError: If conversion fails
        """
        try:
            coord = wcs.pixel_to_world(x, y)
            if frame == "icrs":
                coord = coord.icrs
            elif frame == "galactic":
                coord = coord.galactic

            return float(coord.ra.degree), float(coord.dec.degree)
        except Exception as e:
            raise ValueError(
                f"Failed to convert pixels ({x}, {y}) to world coordinates: {e}"
            ) from e
