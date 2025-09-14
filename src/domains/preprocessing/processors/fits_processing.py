"""Advanced FITS processing utilities for preprocessing workflows.

This module contains heavy image processing operations that require
cv2 and matplotlib dependencies. It's separated from basic I/O to
avoid import issues in the main API.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

# Only import heavy dependencies when actually used
try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

try:
    import matplotlib.pyplot as plt
    from astropy.visualization import astropy_mpl_style

    MATPLOTLIB_AVAILABLE = True
    # Set the default plot style
    plt.style.use(astropy_mpl_style)
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None


class AdvancedFITSProcessor:
    """Advanced FITS processing with heavy image processing capabilities.

    This processor handles complex operations that require cv2 and matplotlib.
    It's designed for preprocessing workflows and dataset creation.
    """

    def __init__(self):
        """Initialize the advanced FITS processor."""
        self._check_dependencies()

    def _check_dependencies(self):
        """Check that required dependencies are available."""
        if not CV2_AVAILABLE:
            raise ImportError(
                "opencv-python (cv2) is required for AdvancedFITSProcessor. "
                "Install with: pip install opencv-python"
            )
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "matplotlib is required for AdvancedFITSProcessor. "
                "Install with: pip install matplotlib"
            )

    def create_star_overlay_plot(
        self,
        image_data: np.ndarray,
        wcs: WCS,
        catalog_table: Table,
        output_path: str,
        pixels: int = 512,
    ) -> str:
        """Create a star overlay plot from image and catalog data.

        Args:
            image_data: The image data array
            wcs: WCS object for the image
            catalog_table: Star catalog as Astropy Table
            output_path: Path for the output image file
            pixels: Image size in pixels

        Returns:
            Path to the created overlay image
        """
        if not MATPLOTLIB_AVAILABLE or plt is None:
            raise ImportError("matplotlib is required for creating overlay plots")
            
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection=wcs)

        # Display the image
        ax.imshow(image_data, cmap="gray", origin="lower")

        # Add star positions from catalog
        # This would need coordinate processing logic
        # For now, create a placeholder

        ax.set_xlabel("RA")
        ax.set_ylabel("Dec")
        ax.grid(color="white", ls="dotted")
        ax.set_title("Star Overlay")

        # Save the plot
        plt.savefig(output_path, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        return output_path

    def convert_image_to_fits_hdu(
        self, image_path: str, hdu_name: str, target_size: tuple[int, int] | None = None
    ) -> fits.ImageHDU:
        """Convert an image file to a FITS HDU.

        Args:
            image_path: Path to the image file
            hdu_name: Name for the HDU
            target_size: Optional (width, height) to resize to

        Returns:
            FITS ImageHDU containing the image data
        """
        if not CV2_AVAILABLE or cv2 is None:
            raise ImportError("opencv-python is required for image conversion")
            
        # Read the image
        image_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image_data is None:
            raise ValueError(f"Could not read image file: {image_path}")

        # Resize if requested
        if target_size:
            width, height = target_size
            image_data = cv2.resize(
                image_data, (width, height), interpolation=cv2.INTER_AREA
            )

        # Create HDU
        return fits.ImageHDU(image_data, name=hdu_name)

    def create_circular_mask(
        self,
        height: int,
        width: int,
        center: tuple[int, int] | None = None,
        radius: float | None = None,
    ) -> np.ndarray:
        """Create a circular mask for image processing.

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

    def create_composite_fits_file(
        self,
        image_data: np.ndarray,
        wcs: WCS,
        catalog_table: Table | None = None,
        pixel_mask: np.ndarray | None = None,
        output_path: str = None,
        include_overlays: bool = True,
    ) -> str:
        """Create a composite FITS file with multiple HDUs.

        This recreates the functionality from your dataGathering.py createStarDataset.

        Args:
            image_data: Primary image data
            wcs: WCS information
            catalog_table: Optional star catalog
            pixel_mask: Optional pixel mask
            output_path: Output file path (temp file if None)
            include_overlays: Whether to create overlay visualizations

        Returns:
            Path to the created FITS file
        """
        if output_path is None:
            # Create temporary file
            temp_fd, output_path = tempfile.mkstemp(suffix=".fits")
            os.close(temp_fd)

        # Create primary HDU with image data
        primary_hdu = fits.PrimaryHDU(image_data, header=wcs.to_header())
        hdul = fits.HDUList([primary_hdu])

        # Add star catalog if provided
        if catalog_table is not None:
            # Sanitize catalog metadata
            sanitized_meta = {k[:8]: v for k, v in (catalog_table.meta or {}).items()}
            catalog_copy = Table(catalog_table, meta=sanitized_meta)
            star_hdu = fits.BinTableHDU(catalog_copy, name="STAR_CATALOG")
            hdul.append(star_hdu)

        # Add pixel mask if provided
        if pixel_mask is not None:
            mask_hdu = fits.ImageHDU(pixel_mask, name="pixel_mask")
            hdul.append(mask_hdu)

        # Save the base FITS file
        hdul.writeto(output_path, overwrite=True)

        # Add overlays if requested
        if include_overlays and catalog_table is not None:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create star overlay plot
                overlay_path = os.path.join(temp_dir, "overlay.png")
                self.create_star_overlay_plot(
                    image_data, wcs, catalog_table, overlay_path
                )

                # Convert to FITS HDU and append
                overlay_hdu = self.convert_image_to_fits_hdu(
                    overlay_path, "star_overlay", (512, 512)
                )

                with fits.open(output_path, mode="update") as update_hdul:
                    update_hdul.append(overlay_hdu)
                    update_hdul.flush()

        return output_path

    def batch_process_fits_directory(
        self,
        input_dir: str,
        output_dir: str,
        file_pattern: str = "*.fits",
        include_overlays: bool = True,
    ) -> list[str]:
        """Batch process FITS files in a directory.

        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            file_pattern: File pattern to match
            include_overlays: Whether to create overlay visualizations

        Returns:
            List of processed file paths
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        processed_files = []

        for fits_file in input_path.glob(file_pattern):
            try:
                # Load the FITS file using basic I/O
                with fits.open(fits_file) as hdul:
                    image_data = hdul[0].data
                    wcs = WCS(hdul[0].header)

                    # Extract catalog if present
                    catalog_table = None
                    if "STAR_CATALOG" in hdul:
                        catalog_table = Table(hdul["STAR_CATALOG"].data)

                # Create processed output
                output_file = output_path / fits_file.name
                self.create_composite_fits_file(
                    image_data=image_data,
                    wcs=wcs,
                    catalog_table=catalog_table,
                    output_path=str(output_file),
                    include_overlays=include_overlays,
                )

                processed_files.append(str(output_file))

            except Exception as e:
                print(f"Error processing {fits_file}: {e}")
                continue

        return processed_files
