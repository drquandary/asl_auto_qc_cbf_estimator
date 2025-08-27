"""
ASL Data Processing Module for ASL Auto-QC CBF Estimator.

This module handles loading, validation, and preprocessing of Arterial Spin Labeling (ASL) MRI data.
It supports standard ASL acquisition protocols and prepares data for CBF quantification and quality control.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.ndimage import median_filter
from skimage import filters
from skimage.morphology import binary_erosion, binary_dilation

logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class ASLDataError(Exception):
    """Exception raised for ASL data processing errors."""
    pass


class ASLProcessor:
    """
    Processor for ASL MRI data handling and preprocessing.
    
    Supports standard ASL acquisition protocols including pCASL, CASL, and PASL.
    Handles control-label pair processing, motion correction assessment, and data validation.
    """
    
    def __init__(self, 
                 temporal_filter: bool = True,
                 spatial_smoothing: bool = True,
                 motion_threshold: float = 2.0):
        """
        Initialize ASL processor.
        
        Args:
            temporal_filter: Apply temporal filtering to reduce noise
            spatial_smoothing: Apply spatial smoothing for SNR improvement
            motion_threshold: Motion threshold in mm for quality assessment
        """
        self.temporal_filter = temporal_filter
        self.spatial_smoothing = spatial_smoothing
        self.motion_threshold = motion_threshold
        
    def load_asl_data(self, 
                      asl_path: Union[str, Path],
                      m0_path: Optional[Union[str, Path]] = None,
                      mask_path: Optional[Union[str, Path]] = None) -> Dict:
        """
        Load ASL data from NIfTI files.
        
        Args:
            asl_path: Path to ASL 4D NIfTI file (control-label pairs)
            m0_path: Optional path to M0 calibration image
            mask_path: Optional path to brain mask
            
        Returns:
            Dictionary containing loaded ASL data and metadata
            
        Raises:
            ASLDataError: If data loading or validation fails
        """
        try:
            asl_path = Path(asl_path)
            if not asl_path.exists():
                raise ASLDataError(f"ASL file not found: {asl_path}")
            
            # Load ASL data
            asl_img = nib.load(asl_path)
            asl_data = asl_img.get_fdata()
            
            # Validate ASL data dimensions
            if asl_data.ndim != 4:
                raise ASLDataError(f"ASL data must be 4D, got {asl_data.ndim}D")
            
            if asl_data.shape[3] % 2 != 0:
                raise ASLDataError(f"ASL data must have even number of volumes for control-label pairs, got {asl_data.shape[3]}")
            
            logger.info(f"Loaded ASL data: {asl_data.shape}")
            
            # Load M0 calibration if provided
            m0_data = None
            m0_img = None
            if m0_path:
                m0_path = Path(m0_path)
                if m0_path.exists():
                    m0_img = nib.load(m0_path)
                    m0_data = m0_img.get_fdata()
                    logger.info(f"Loaded M0 data: {m0_data.shape}")
                else:
                    logger.warning(f"M0 file not found: {m0_path}")
            
            # Load brain mask if provided
            mask_data = None
            if mask_path:
                mask_path = Path(mask_path)
                if mask_path.exists():
                    mask_img = nib.load(mask_path)
                    mask_data = mask_img.get_fdata() > 0.5
                    logger.info(f"Loaded mask: {mask_data.shape}")
                else:
                    logger.warning(f"Mask file not found: {mask_path}")
            
            # Generate basic brain mask if none provided
            if mask_data is None:
                mask_data = self._generate_brain_mask(asl_data)
                logger.info("Generated brain mask from ASL data")
            
            return {
                'asl_data': asl_data,
                'asl_img': asl_img,
                'm0_data': m0_data,
                'm0_img': m0_img,
                'mask_data': mask_data,
                'n_volumes': asl_data.shape[3],
                'n_pairs': asl_data.shape[3] // 2,
                'voxel_size': asl_img.header.get_zooms()[:3],
                'tr': float(asl_img.header.get_zooms()[3]) if len(asl_img.header.get_zooms()) > 3 else 4.0,
                'shape': asl_data.shape[:3]
            }
            
        except Exception as e:
            logger.error(f"Error loading ASL data: {e}")
            raise ASLDataError(f"Failed to load ASL data: {e}")
    
    def separate_control_label_pairs(self, asl_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Separate ASL data into control and label volumes.
        
        Assumes alternating control-label acquisition pattern.
        
        Args:
            asl_data: 4D ASL data array
            
        Returns:
            Tuple of (control_volumes, label_volumes)
        """
        if asl_data.shape[3] % 2 != 0:
            raise ASLDataError("ASL data must have even number of volumes")
        
        # Standard pattern: control (even indices), label (odd indices)
        control_volumes = asl_data[:, :, :, 0::2]
        label_volumes = asl_data[:, :, :, 1::2]
        
        logger.info(f"Separated {control_volumes.shape[3]} control and {label_volumes.shape[3]} label volumes")
        
        return control_volumes, label_volumes
    
    def calculate_perfusion_weighted_images(self, 
                                           control_volumes: np.ndarray,
                                           label_volumes: np.ndarray) -> np.ndarray:
        """
        Calculate perfusion-weighted difference images.
        
        Args:
            control_volumes: Control volumes
            label_volumes: Label volumes
            
        Returns:
            Perfusion-weighted difference images (control - label)
        """
        if control_volumes.shape != label_volumes.shape:
            raise ASLDataError("Control and label volumes must have same shape")
        
        # Calculate difference: control - label (positive values indicate perfusion)
        diff_images = control_volumes - label_volumes
        
        # Apply temporal filtering if requested
        if self.temporal_filter:
            diff_images = self._apply_temporal_filter(diff_images)
        
        # Apply spatial smoothing if requested
        if self.spatial_smoothing:
            diff_images = self._apply_spatial_smoothing(diff_images)
        
        logger.info(f"Calculated {diff_images.shape[3]} perfusion-weighted images")
        
        return diff_images
    
    def preprocess_asl_data(self, data_dict: Dict) -> Dict:
        """
        Complete preprocessing pipeline for ASL data.
        
        Args:
            data_dict: Dictionary from load_asl_data
            
        Returns:
            Dictionary with preprocessed data and analysis results
        """
        asl_data = data_dict['asl_data']
        mask_data = data_dict['mask_data']
        
        # Separate control and label pairs
        control_volumes, label_volumes = self.separate_control_label_pairs(asl_data)
        
        # Calculate perfusion-weighted images
        perfusion_images = self.calculate_perfusion_weighted_images(control_volumes, label_volumes)
        
        # Calculate mean perfusion-weighted image
        mean_perfusion = np.mean(perfusion_images, axis=3)
        
        # Calculate mean control and label images for M0 estimation if no M0 provided
        mean_control = np.mean(control_volumes, axis=3)
        mean_label = np.mean(label_volumes, axis=3)
        
        # Use M0 if available, otherwise estimate from mean control
        if data_dict['m0_data'] is not None:
            m0_image = data_dict['m0_data']
        else:
            m0_image = mean_control
            logger.info("Using mean control image as M0 estimate")
        
        # Calculate basic statistics within brain mask
        mask_indices = mask_data > 0.5
        
        perfusion_stats = {
            'mean_perfusion_signal': float(np.mean(mean_perfusion[mask_indices])),
            'std_perfusion_signal': float(np.std(mean_perfusion[mask_indices])),
            'snr_perfusion': float(np.mean(mean_perfusion[mask_indices]) / (np.std(mean_perfusion[mask_indices]) + 1e-8))
        }
        
        # Update data dictionary with processed results
        processed_data = data_dict.copy()
        processed_data.update({
            'control_volumes': control_volumes,
            'label_volumes': label_volumes,
            'perfusion_images': perfusion_images,
            'mean_perfusion': mean_perfusion,
            'mean_control': mean_control,
            'mean_label': mean_label,
            'm0_image': m0_image,
            'perfusion_stats': perfusion_stats,
            'preprocessed': True
        })
        
        logger.info("ASL preprocessing completed successfully")
        
        return processed_data
    
    def _generate_brain_mask(self, asl_data: np.ndarray) -> np.ndarray:
        """
        Generate a simple brain mask from ASL data.
        
        Args:
            asl_data: 4D ASL data
            
        Returns:
            3D brain mask
        """
        # Calculate mean across time
        mean_img = np.mean(asl_data, axis=3)
        
        # Apply Otsu thresholding
        threshold = filters.threshold_otsu(mean_img[mean_img > 0])
        mask = mean_img > threshold * 0.3  # More liberal threshold for ASL
        
        # Clean up mask with morphological operations
        mask = binary_erosion(mask, iterations=1)
        mask = binary_dilation(mask, iterations=2)
        
        # Remove small connected components
        labeled_mask, num_labels = ndimage.label(mask)
        if num_labels > 1:
            # Keep largest component
            component_sizes = ndimage.sum(mask, labeled_mask, range(num_labels + 1))
            largest_component = np.argmax(component_sizes[1:]) + 1
            mask = labeled_mask == largest_component
        
        logger.info(f"Generated brain mask: {np.sum(mask)} voxels")
        
        return mask.astype(bool)
    
    def _apply_temporal_filter(self, images: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Apply median temporal filter to reduce noise.
        
        Args:
            images: 4D image data
            kernel_size: Temporal kernel size for median filter
            
        Returns:
            Temporally filtered images
        """
        filtered_images = np.zeros_like(images)
        
        for i in range(images.shape[3]):
            # Create temporal window
            start_idx = max(0, i - kernel_size // 2)
            end_idx = min(images.shape[3], i + kernel_size // 2 + 1)
            
            # Apply median filter across temporal window
            temporal_data = images[:, :, :, start_idx:end_idx]
            filtered_images[:, :, :, i] = np.median(temporal_data, axis=3)
        
        return filtered_images
    
    def _apply_spatial_smoothing(self, images: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """
        Apply Gaussian spatial smoothing.
        
        Args:
            images: 4D image data
            sigma: Gaussian kernel sigma in voxels
            
        Returns:
            Spatially smoothed images
        """
        smoothed_images = np.zeros_like(images)
        
        for i in range(images.shape[3]):
            smoothed_images[:, :, :, i] = ndimage.gaussian_filter(images[:, :, :, i], sigma=sigma)
        
        return smoothed_images

    def estimate_motion_parameters(self, asl_data: np.ndarray) -> Dict:
        """
        Estimate motion parameters from ASL time series.
        
        This is a simplified motion assessment based on volume-to-volume differences.
        In practice, you would use proper motion correction tools like FSL MCFLIRT or SPM realign.
        
        Args:
            asl_data: 4D ASL data
            
        Returns:
            Dictionary with motion estimates
        """
        n_volumes = asl_data.shape[3]
        motion_estimates = []
        
        # Use first volume as reference
        reference = asl_data[:, :, :, 0]
        
        for i in range(1, n_volumes):
            current = asl_data[:, :, :, i]
            
            # Calculate normalized cross-correlation as motion indicator
            diff = np.sum((current - reference) ** 2) / np.sum(reference ** 2)
            motion_estimates.append(diff)
        
        motion_estimates = np.array(motion_estimates)
        
        # Calculate motion metrics
        mean_motion = np.mean(motion_estimates)
        max_motion = np.max(motion_estimates)
        motion_outliers = np.sum(motion_estimates > (np.mean(motion_estimates) + 2 * np.std(motion_estimates)))
        
        motion_dict = {
            'motion_estimates': motion_estimates.tolist(),
            'mean_motion': float(mean_motion),
            'max_motion': float(max_motion),
            'motion_outliers': int(motion_outliers),
            'motion_quality': 'good' if max_motion < 0.1 else 'moderate' if max_motion < 0.2 else 'poor'
        }
        
        logger.info(f"Motion assessment: {motion_dict['motion_quality']} (max motion: {max_motion:.4f})")
        
        return motion_dict