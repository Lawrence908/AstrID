"""Enhanced FITS file processing for astronomical observations."""

import logging
import time
from pathlib import Path

import numpy as np
from astropy.io import fits

logger = logging.getLogger(__name__)


class FITSProcessor:
    """Enhanced FITS file processor with validation and advanced handling."""

    def __init__(self):
        """Initialize FITS processor."""
        self.logger = logging.getLogger(__name__)

    def read_fits_with_validation(self, file_path: str) -> tuple[np.ndarray, dict]:
        """Read FITS file with comprehensive validation.

        Args:
            file_path: Path to FITS file

        Returns:
            Tuple of (image_data, all_metadata)

        Raises:
            FileNotFoundError: If FITS file doesn't exist
            ValueError: If FITS file is invalid or corrupted
        """
        start_time = time.time()

        try:
            # Validate file exists
            if not Path(file_path).exists():
                raise FileNotFoundError(f"FITS file not found: {file_path}")

            # Validate FITS structure first
            if not self.validate_fits_structure(file_path):
                raise ValueError(f"Invalid FITS file structure: {file_path}")

            with fits.open(file_path, mode="readonly", memmap=False) as hdul:
                # Extract primary image data
                primary_hdu = hdul[0]
                image_data = primary_hdu.data

                if image_data is None:
                    raise ValueError(f"No image data found in primary HDU: {file_path}")

                # Extract comprehensive metadata
                metadata = {
                    "primary_header": dict(primary_hdu.header),
                    "file_info": {
                        "path": file_path,
                        "size_bytes": Path(file_path).stat().st_size,
                        "num_hdus": len(hdul),
                        "image_shape": image_data.shape,
                        "image_dtype": str(image_data.dtype),
                    },
                    "processing_info": {
                        "read_time": time.time() - start_time,
                        "validation_passed": True,
                    },
                }

                # Process multi-extension data if present
                if len(hdul) > 1:
                    metadata["extensions"] = {}
                    for i, hdu in enumerate(hdul[1:], 1):
                        metadata["extensions"][f"ext_{i}"] = {
                            "name": hdu.name,
                            "data_shape": hdu.data.shape
                            if hdu.data is not None
                            else None,
                            "header_keys": list(hdu.header.keys()),
                        }

                self.logger.info(
                    f"Successfully read FITS file: {file_path} "
                    f"({image_data.shape}, {len(hdul)} HDUs, "
                    f"{time.time() - start_time:.3f}s)"
                )

                return image_data, metadata

        except fits.VerifyError as e:
            self.logger.error(f"FITS verification error for {file_path}: {e}")
            raise ValueError(f"FITS verification failed: {e}") from e
        except Exception as e:
            self.logger.error(f"Error reading FITS file {file_path}: {e}")
            raise

    def write_fits_with_metadata(
        self,
        data: np.ndarray,
        headers: dict,
        file_path: str,
        compress: bool = True,
        overwrite: bool = True,
    ) -> None:
        """Write FITS file with comprehensive metadata and optional compression.

        Args:
            data: Image data array
            headers: Header information dictionary
            file_path: Output file path
            compress: Whether to compress the FITS file
            overwrite: Whether to overwrite existing file

        Raises:
            ValueError: If data or headers are invalid
        """
        start_time = time.time()

        try:
            # Validate input data
            if data is None or data.size == 0:
                raise ValueError("Cannot write empty data array")

            # Create primary HDU with data
            primary_hdu = fits.PrimaryHDU(data)

            # Add standard processing headers
            primary_hdu.header["ORIGIN"] = "AstrID-FITS-Pipeline"
            primary_hdu.header["DATE"] = fits.Card(
                "DATE", time.strftime("%Y-%m-%dT%H:%M:%S"), "File creation date"
            )
            primary_hdu.header["CREATOR"] = "AstrID FITS Processor"

            # Add custom headers
            for key, value in headers.items():
                if len(key) <= 8:  # FITS keyword length limit
                    try:
                        primary_hdu.header[key] = value
                    except ValueError as e:
                        self.logger.warning(
                            f"Skipping invalid header {key}={value}: {e}"
                        )

            # Create HDU list
            hdul = fits.HDUList([primary_hdu])

            # Write to file with optional compression
            if compress and not file_path.endswith(".gz"):
                # Use Rice compression for integer data, GZIP for float
                if np.issubdtype(data.dtype, np.integer):
                    compressed_hdu = fits.CompImageHDU(
                        data, header=primary_hdu.header, compression_type="RICE_1"
                    )
                else:
                    compressed_hdu = fits.CompImageHDU(
                        data, header=primary_hdu.header, compression_type="GZIP_1"
                    )
                hdul = fits.HDUList([compressed_hdu])

            hdul.writeto(file_path, overwrite=overwrite)

            write_time = time.time() - start_time
            file_size = Path(file_path).stat().st_size

            self.logger.info(
                f"Successfully wrote FITS file: {file_path} "
                f"({data.shape}, {file_size} bytes, {write_time:.3f}s)"
            )

        except Exception as e:
            self.logger.error(f"Error writing FITS file {file_path}: {e}")
            raise

    def validate_fits_structure(self, file_path: str) -> bool:
        """Validate FITS file structure and integrity.

        Args:
            file_path: Path to FITS file

        Returns:
            bool: True if FITS file is valid, False otherwise
        """
        try:
            # Check file exists and is readable
            if not Path(file_path).exists():
                self.logger.error(f"FITS file does not exist: {file_path}")
                return False

            # Verify FITS format
            with fits.open(
                file_path, mode="readonly", do_not_scale_image_data=True
            ) as hdul:
                # Check primary HDU exists
                if len(hdul) == 0:
                    self.logger.error(f"No HDUs found in FITS file: {file_path}")
                    return False

                # Verify primary HDU
                primary_hdu = hdul[0]
                if not primary_hdu.header.get("SIMPLE", False):
                    self.logger.error(f"Not a standard FITS file: {file_path}")
                    return False

                # Check for required keywords
                required_keywords = ["NAXIS", "BITPIX"]
                for keyword in required_keywords:
                    if keyword not in primary_hdu.header:
                        self.logger.error(
                            f"Missing required keyword {keyword}: {file_path}"
                        )
                        return False

                # Verify data integrity if present
                if primary_hdu.data is not None:
                    # Check data shape matches header
                    naxis = primary_hdu.header.get("NAXIS", 0)
                    if naxis > 0:
                        expected_shape = []
                        for i in range(naxis, 0, -1):
                            expected_shape.append(
                                primary_hdu.header.get(f"NAXIS{i}", 0)
                            )
                        expected_shape = tuple(expected_shape)

                        if primary_hdu.data.shape != expected_shape:
                            self.logger.error(
                                f"Data shape mismatch in {file_path}: "
                                f"expected {expected_shape}, got {primary_hdu.data.shape}"
                            )
                            return False

            self.logger.debug(f"FITS file validation passed: {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"FITS validation error for {file_path}: {e}")
            return False

    def extract_all_headers(self, file_path: str) -> dict:
        """Extract all headers from all HDUs in FITS file.

        Args:
            file_path: Path to FITS file

        Returns:
            dict: Dictionary containing all headers organized by HDU

        Raises:
            ValueError: If FITS file is invalid
        """
        try:
            headers = {}

            with fits.open(file_path, mode="readonly") as hdul:
                for i, hdu in enumerate(hdul):
                    hdu_name = f"HDU_{i}" if hdu.name == "" else hdu.name
                    headers[hdu_name] = {
                        "index": i,
                        "name": hdu.name,
                        "header": dict(hdu.header),
                        "data_info": {
                            "has_data": hdu.data is not None,
                            "shape": hdu.data.shape if hdu.data is not None else None,
                            "dtype": str(hdu.data.dtype)
                            if hdu.data is not None
                            else None,
                        },
                    }

            self.logger.info(
                f"Extracted headers from {len(headers)} HDUs in {file_path}"
            )
            return headers

        except Exception as e:
            self.logger.error(f"Error extracting headers from {file_path}: {e}")
            raise ValueError(f"Failed to extract headers: {e}") from e

    def optimize_fits_file(
        self,
        input_path: str,
        output_path: str,
        compression_type: str = "RICE_1",
        tile_size: tuple[int, int] = (64, 64),
    ) -> dict:
        """Optimize FITS file with compression and tiling.

        Args:
            input_path: Input FITS file path
            output_path: Output optimized FITS file path
            compression_type: Compression algorithm ('RICE_1', 'GZIP_1', 'PLIO_1')
            tile_size: Tile size for compressed images

        Returns:
            dict: Optimization results and statistics
        """
        start_time = time.time()

        try:
            original_size = Path(input_path).stat().st_size

            with fits.open(input_path) as hdul:
                optimized_hdul = fits.HDUList()

                for i, hdu in enumerate(hdul):
                    if i == 0 and hdu.data is not None:
                        # Compress primary image HDU
                        compressed_hdu = fits.CompImageHDU(
                            data=hdu.data,
                            header=hdu.header,
                            compression_type=compression_type,
                        )
                        optimized_hdul.append(compressed_hdu)
                    else:
                        # Keep other HDUs as-is
                        optimized_hdul.append(hdu)

                optimized_hdul.writeto(output_path, overwrite=True)

            optimized_size = Path(output_path).stat().st_size
            processing_time = time.time() - start_time

            results = {
                "original_size": original_size,
                "optimized_size": optimized_size,
                "compression_ratio": original_size / optimized_size,
                "size_reduction_percent": (1 - optimized_size / original_size) * 100,
                "processing_time": processing_time,
                "compression_type": compression_type,
                "tile_size": tile_size,
            }

            self.logger.info(
                f"Optimized FITS file: {input_path} -> {output_path} "
                f"({results['compression_ratio']:.1f}x compression, "
                f"{results['size_reduction_percent']:.1f}% reduction)"
            )

            return results

        except Exception as e:
            self.logger.error(f"Error optimizing FITS file {input_path}: {e}")
            raise

    def verify_fits_integrity(self, file_path: str) -> dict:
        """Perform comprehensive FITS file integrity verification.

        Args:
            file_path: Path to FITS file

        Returns:
            dict: Integrity verification results
        """
        results = {
            "file_exists": False,
            "is_valid_fits": False,
            "structure_valid": False,
            "data_readable": False,
            "header_complete": False,
            "checksum_valid": None,
            "errors": [],
            "warnings": [],
        }

        try:
            # Check file existence
            if not Path(file_path).exists():
                results["errors"].append("File does not exist")
                return results
            results["file_exists"] = True

            # Basic FITS validation
            results["structure_valid"] = self.validate_fits_structure(file_path)

            with fits.open(
                file_path,
                mode="readonly",
                checksum=True,
                disable_image_compression=False,
            ) as hdul:
                results["is_valid_fits"] = True

                # Check each HDU
                for i, hdu in enumerate(hdul):
                    try:
                        # Test data readability
                        if hdu.data is not None:
                            _ = hdu.data.shape  # Trigger data loading
                            results["data_readable"] = True

                        # Check header completeness
                        if "SIMPLE" in hdu.header or "XTENSION" in hdu.header:
                            results["header_complete"] = True

                        # Check checksums if present
                        if "CHECKSUM" in hdu.header:
                            try:
                                hdu.verify_checksum()
                                results["checksum_valid"] = True
                            except Exception:
                                results["checksum_valid"] = False
                                results["warnings"].append(
                                    f"Checksum verification failed for HDU {i}"
                                )

                    except Exception as e:
                        results["errors"].append(f"Error reading HDU {i}: {str(e)}")

            # Overall status
            if not results["errors"] and results["structure_valid"]:
                self.logger.info(f"FITS integrity verification passed: {file_path}")
            else:
                self.logger.warning(
                    f"FITS integrity issues found in {file_path}: {results['errors']}"
                )

        except Exception as e:
            results["errors"].append(f"Verification failed: {str(e)}")
            self.logger.error(f"FITS integrity verification error for {file_path}: {e}")

        return results
