"""Advanced astronomical image processing utilities.

This module contains the core image processing algorithms for astronomical data,
including preprocessing, differencing, and quality assessment.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy import ndimage
from scipy.optimize import minimize
from scipy.ndimage import label
from astropy.stats import sigma_clipped_stats
from astropy.convolution import Gaussian2DKernel, convolve
import cv2
from skimage import morphology

# Only import heavy dependencies when actually used
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    patches = None


class AstronomicalImageProcessor:
    """Core astronomical image processing functionality."""
    
    def __init__(self):
        """Initialize the processor."""
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check that required dependencies are available."""
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: matplotlib not available. Visualization functions will be disabled.")
    
    def enhance_astronomical_image(
        self,
        image: np.ndarray,
        *,
        bias_correction: bool = True,
        flat_correction: bool = True,
        dark_correction: bool = True,
        cosmic_ray_removal: bool = True,
        background_subtraction: bool = True,
        noise_reduction: bool = True
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Enhanced astronomical image preprocessing pipeline.
        
        Args:
            image: Input astronomical image
            bias_correction: Apply bias frame correction
            flat_correction: Apply flat field correction
            dark_correction: Apply dark frame correction
            cosmic_ray_removal: Remove cosmic rays
            background_subtraction: Subtract background
            noise_reduction: Apply noise reduction
            
        Returns:
            Tuple of (processed_image, quality_metrics)
        """
        # Ensure input is 2D for astronomical processing
        if image.ndim == 3:
            # Convert RGB to grayscale using standard weights
            if image.shape[-1] == 3:
                processed = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
            else:
                # For other 3D cases, take mean across last dimension
                processed = np.mean(image, axis=-1)
        elif image.ndim == 2:
            processed = image.copy()
        else:
            # For higher dimensions, flatten to 2D
            processed = np.mean(image, axis=tuple(range(image.ndim - 2)))
        
        processed = processed.astype(float)
        quality_metrics = {}
        
        # 1. Bias correction (simplified - in real implementation would use bias frames)
        if bias_correction:
            bias_level = np.percentile(processed, 1)  # Estimate bias level
            processed = processed - bias_level
            quality_metrics['bias_level'] = float(bias_level)
        
        # 2. Dark correction (simplified)
        if dark_correction:
            dark_current = np.percentile(processed, 0.1)  # Estimate dark current
            processed = processed - dark_current
            quality_metrics['dark_current'] = float(dark_current)
        
        # 3. Flat field correction (simplified)
        if flat_correction:
            # Estimate flat field from image itself (in practice, use separate flat frames)
            flat_field = ndimage.gaussian_filter(processed, sigma=50)
            flat_field = flat_field / np.mean(flat_field)
            processed = processed / (flat_field + 1e-10)
            quality_metrics['flat_variation'] = float(np.std(flat_field))
        
        # 4. Cosmic ray removal
        if cosmic_ray_removal:
            processed = self.remove_cosmic_rays(processed)
            quality_metrics['cosmic_rays_removed'] = True
        
        # 5. Background subtraction
        if background_subtraction:
            background = self.estimate_background(processed)
            processed = processed - background
            quality_metrics['background_level'] = float(np.mean(background))
        
        # 6. Noise reduction
        if noise_reduction:
            processed = self.apply_adaptive_noise_reduction(processed)
            quality_metrics['noise_reduction_applied'] = True
        
        # Calculate final quality metrics
        quality_metrics.update(self.calculate_image_quality(processed))
        
        return processed, quality_metrics

    def remove_cosmic_rays(self, image: np.ndarray, threshold: float = 5.0) -> np.ndarray:
        """Remove cosmic rays using sigma clipping."""
        mean, median, std = sigma_clipped_stats(image, sigma=3.0)
        cosmic_ray_mask = image > (median + threshold * std)
        
        # Replace cosmic rays with median value
        result = image.copy()
        result[cosmic_ray_mask] = median
        
        return result

    def estimate_background(self, image: np.ndarray, method: str = 'gaussian') -> np.ndarray:
        """Estimate background using various methods."""
        if method == 'gaussian':
            # Use large Gaussian kernel to estimate background
            # Ensure kernel size is odd (required for convolution)
            kernel = Gaussian2DKernel(x_stddev=20, y_stddev=20, x_size=41, y_size=41)
            background = convolve(image, kernel)
        elif method == 'morphology':
            # Use morphological operations
            selem = morphology.disk(20)
            background = morphology.white_tophat(image, selem)
        else:
            # Simple median filter
            background = ndimage.median_filter(image, size=20)
        
        return np.array(background)

    def apply_adaptive_noise_reduction(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive noise reduction based on local statistics."""
        # Calculate local noise level
        local_std = ndimage.generic_filter(image, np.std, size=5)
        
        # Apply bilateral filter with adaptive parameters
        result = cv2.bilateralFilter(
            image.astype(np.uint8), 
            d=9, 
            sigmaColor=75, 
            sigmaSpace=75
        ).astype(float)
        
        return result

    def calculate_image_quality(self, image: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive image quality metrics."""
        metrics = {}
        
        # Basic statistics
        metrics['mean'] = float(np.mean(image))
        metrics['std'] = float(np.std(image))
        metrics['min'] = float(np.min(image))
        metrics['max'] = float(np.max(image))
        
        # Signal-to-noise ratio estimation
        signal = np.mean(image)
        noise = np.std(image)
        metrics['snr'] = float(signal / noise) if noise > 0 else 0.0
        
        # Contrast metrics
        metrics['contrast'] = float((np.max(image) - np.min(image)) / (np.max(image) + np.min(image) + 1e-10))
        
        # Sharpness (using Laplacian variance)
        laplacian_var = cv2.Laplacian(image.astype(np.uint8), cv2.CV_64F).var()
        metrics['sharpness'] = float(laplacian_var)
        
        # Dynamic range
        metrics['dynamic_range'] = float(np.max(image) - np.min(image))
        
        return metrics


class ImageDifferencingProcessor:
    """Image differencing algorithms for astronomical data."""
    
    def perform_image_differencing(
        self,
        science_image: np.ndarray,
        reference_image: np.ndarray,
        method: str = 'zogy',
        *,
        psf_science: Optional[Gaussian2DKernel] = None,
        psf_reference: Optional[Gaussian2DKernel] = None,
        noise_science: Optional[np.ndarray] = None,
        noise_reference: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Perform image differencing using various algorithms.
        
        Args:
            science_image: Current observation image
            reference_image: Historical reference image
            method: Differencing method ('zogy', 'classic', 'optimal')
            psf_science: PSF of science image
            psf_reference: PSF of reference image
            noise_science: Noise map of science image
            noise_reference: Noise map of reference image
            
        Returns:
            Tuple of (difference_image, metrics)
        """
        if method == 'zogy':
            return self.zogy_differencing(science_image, reference_image, psf_science, psf_reference, noise_science, noise_reference)
        elif method == 'classic':
            return self.classic_differencing(science_image, reference_image)
        elif method == 'optimal':
            return self.optimal_differencing(science_image, reference_image)
        else:
            raise ValueError(f"Unknown differencing method: {method}")

    def zogy_differencing(
        self,
        science_image: np.ndarray,
        reference_image: np.ndarray,
        psf_science: Optional[Gaussian2DKernel] = None,
        psf_reference: Optional[Gaussian2DKernel] = None,
        noise_science: Optional[np.ndarray] = None,
        noise_reference: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        ZOGY (Zackay, Ofek, and Gal-Yam) optimal image differencing algorithm.
        """
        # Convert to float
        S = science_image.astype(float)
        R = reference_image.astype(float)
        
        # Estimate noise if not provided
        if noise_science is None:
            noise_science = self.estimate_noise_map(S)
        if noise_reference is None:
            noise_reference = self.estimate_noise_map(R)
        
        # Create simple PSFs if not provided
        if psf_science is None:
            psf_science = self.create_gaussian_psf(S.shape, sigma=1.0)
        if psf_reference is None:
            psf_reference = self.create_gaussian_psf(R.shape, sigma=1.0)
        
        # ZOGY algorithm implementation
        # This is a simplified version - full implementation would include FFT operations
        
        # Normalize images
        S_norm = S / np.sqrt(np.sum(S**2))
        R_norm = R / np.sqrt(np.sum(R**2))
        
        # Calculate difference
        D = S_norm - R_norm
        
        # Apply PSF matching (simplified)
        D_matched = convolve(D, psf_science)
        
        # Calculate significance map
        noise_combined = np.sqrt(noise_science**2 + noise_reference**2)
        significance = D_matched / (noise_combined + 1e-10)
        
        metrics = {
            'max_significance': float(np.max(significance)),
            'mean_significance': float(np.mean(significance)),
            'std_significance': float(np.std(significance)),
            'snr_improvement': float(np.max(significance) / np.max(np.abs(D)))
        }
        
        return D_matched, metrics

    def classic_differencing(self, science_image: np.ndarray, reference_image: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Classic image differencing (simple subtraction)."""
        # Align images (simplified - in practice would use WCS alignment)
        diff = science_image.astype(float) - reference_image.astype(float)
        
        metrics = {
            'max_diff': float(np.max(diff)),
            'min_diff': float(np.min(diff)),
            'mean_diff': float(np.mean(diff)),
            'std_diff': float(np.std(diff))
        }
        
        return diff, metrics

    def optimal_differencing(self, science_image: np.ndarray, reference_image: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Optimal image differencing with scaling and offset correction."""
        S = science_image.astype(float)
        R = reference_image.astype(float)
        
        # Find optimal scaling and offset
        def objective(params):
            scale, offset = params
            diff = S - (scale * R + offset)
            return np.sum(diff**2)
        
        result = minimize(objective, [1.0, 0.0], method='BFGS')
        scale, offset = result.x
        
        # Apply optimal transformation
        diff = S - (scale * R + offset)
        
        metrics = {
            'optimal_scale': float(scale),
            'optimal_offset': float(offset),
            'residual_rms': float(np.sqrt(np.mean(diff**2))),
            'correlation': float(np.corrcoef(S.flatten(), R.flatten())[0, 1])
        }
        
        return diff, metrics

    def estimate_noise_map(self, image: np.ndarray, method: str = 'local_std') -> np.ndarray:
        """Estimate noise map for an image."""
        if method == 'local_std':
            # Calculate local standard deviation
            noise_map = ndimage.generic_filter(image, np.std, size=5)
        elif method == 'mad':
            # Median absolute deviation
            median = np.median(image)
            mad = np.median(np.abs(image - median))
            noise_map = np.full_like(image, mad * 1.4826)  # Convert MAD to std
        else:
            # Global standard deviation
            noise_map = np.full_like(image, np.std(image))
        
        return noise_map

    def create_gaussian_psf(self, shape: Tuple[int, int], sigma: float = 1.0) -> Gaussian2DKernel:
        """Create a Gaussian PSF kernel."""
        # Ensure kernel size is odd (required for convolution)
        # Use a reasonable size based on sigma, but ensure it's odd
        kernel_size = max(7, int(6 * sigma) + 1)  # At least 7x7, ensure odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        return Gaussian2DKernel(x_stddev=sigma, y_stddev=sigma, x_size=kernel_size, y_size=kernel_size)


class SourceDetectionProcessor:
    """Source detection and candidate analysis."""
    
    def detect_sources_in_difference(
        self,
        difference_image: np.ndarray,
        threshold: float = 3.0,
        min_area: int = 5,
        *,
        noise_map: Optional[np.ndarray] = None
    ) -> Tuple[List[Dict], np.ndarray]:
        """
        Detect sources in difference image.
        
        Args:
            difference_image: Difference image
            threshold: Detection threshold in sigma
            min_area: Minimum area for detection
            noise_map: Optional noise map for significance calculation
            
        Returns:
            Tuple of (detected_sources, source_mask)
        """
        if noise_map is None:
            differencing_processor = ImageDifferencingProcessor()
            noise_map = differencing_processor.estimate_noise_map(difference_image)
        
        # Calculate significance map
        significance = difference_image / (noise_map + 1e-10)
        
        # Create binary mask
        source_mask = significance > threshold
        
        # Remove small objects
        labeled_mask, num_sources = label(source_mask)
        
        sources = []
        for i in range(1, num_sources + 1):
            source_pixels = labeled_mask == i
            if np.sum(source_pixels) >= min_area:
                # Calculate source properties
                y_coords, x_coords = np.where(source_pixels)
                
                # Center of mass
                center_y = np.mean(y_coords)
                center_x = np.mean(x_coords)
                
                # Flux and significance
                flux = np.sum(difference_image[source_pixels])
                max_significance = np.max(significance[source_pixels])
                mean_significance = np.mean(significance[source_pixels])
                
                # Bounding box
                y_min, y_max = np.min(y_coords), np.max(y_coords)
                x_min, x_max = np.min(x_coords), np.max(x_coords)
                
                source_info = {
                    'id': i,
                    'center_y': float(center_y),
                    'center_x': float(center_x),
                    'flux': float(flux),
                    'max_significance': float(max_significance),
                    'mean_significance': float(mean_significance),
                    'area': int(np.sum(source_pixels)),
                    'bbox': (int(y_min), int(y_max), int(x_min), int(x_max))
                }
                sources.append(source_info)
        
        return sources, source_mask

    def visualize_detections(
        self,
        image: np.ndarray,
        sources: List[Dict],
        *,
        figsize: Tuple[int, int] = (12, 8),
        title: str = "Source Detections"
    ) -> None:
        """Visualize detected sources on the image."""
        if not MATPLOTLIB_AVAILABLE or plt is None or patches is None:
            print("Matplotlib not available. Cannot create visualization.")
            return
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Display image
        im = ax.imshow(image, origin='lower', cmap='gray')
        plt.colorbar(im, ax=ax, label='Intensity')
        
        # Plot detections
        for source in sources:
            center_y, center_x = source['center_y'], source['center_x']
            significance = source['max_significance']
            
            # Color by significance
            color = 'red' if significance > 5 else 'orange' if significance > 3 else 'yellow'
            
            # Plot circle
            circle = patches.Circle((center_x, center_y), radius=3, 
                                  color=color, fill=False, linewidth=2)
            ax.add_patch(circle)
            
            # Add text
            ax.text(center_x + 5, center_y + 5, f"{source['id']}: {significance:.1f}σ", 
                    color=color, fontsize=8, weight='bold')
        
        ax.set_title(title)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        plt.tight_layout()
        plt.show()
