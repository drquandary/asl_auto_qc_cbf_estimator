"""
CBF Calculator Module for ASL Auto-QC CBF Estimator.

This module implements cerebral blood flow (CBF) quantification algorithms
based on BASIL (Bayesian Inference for Arterial Spin Labelling) methodology
and standard ASL quantification models.
"""

import logging
from typing import Dict, Optional, Tuple, Union
import warnings

import numpy as np
from scipy import ndimage
from scipy.optimize import minimize
from scipy.stats import gamma

logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class CBFCalculationError(Exception):
    """Exception raised for CBF calculation errors."""
    pass


class CBFCalculator:
    """
    CBF quantification using BASIL-inspired algorithms and standard ASL models.
    
    Implements both simple and advanced CBF quantification methods including
    kinetic modeling, partial volume correction, and Bayesian inference approaches.
    """
    
    def __init__(self, 
                 labeling_efficiency: float = 0.85,
                 blood_t1: float = 1.65,
                 tissue_t1: float = 1.3,
                 partition_coefficient: float = 0.9,
                 post_labeling_delay: float = 1.8,
                 labeling_duration: float = 1.8):
        """
        Initialize CBF calculator with standard ASL parameters.
        
        Args:
            labeling_efficiency: Labeling efficiency (alpha) [0.85 for pCASL]
            blood_t1: T1 of arterial blood at 3T (seconds)
            tissue_t1: T1 of gray matter at 3T (seconds)
            partition_coefficient: Brain-blood partition coefficient (lambda)
            post_labeling_delay: Post-labeling delay (PLD) in seconds
            labeling_duration: Labeling duration (tau) in seconds
        """
        self.alpha = labeling_efficiency
        self.t1_blood = blood_t1
        self.t1_tissue = tissue_t1
        self.lambda_val = partition_coefficient
        self.pld = post_labeling_delay
        self.tau = labeling_duration
        
        # Calculate T1 decay factors
        self.t1_decay_factor = np.exp(-self.pld / self.t1_blood)
        
        logger.info(f"CBF Calculator initialized with PLD={self.pld}s, tau={self.tau}s")
    
    def calculate_cbf_simple(self, 
                           perfusion_images: np.ndarray,
                           m0_image: np.ndarray,
                           mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate CBF using the simple single-compartment model.
        
        Based on the standard ASL quantification equation:
        CBF = (6000 * lambda * delta_M * exp(PLD/T1_blood)) / (2 * alpha * T1_blood * M0 * (1 - exp(-tau/T1_blood)))
        
        Args:
            perfusion_images: Perfusion-weighted difference images
            m0_image: M0 calibration image or mean control image
            mask: Brain mask (optional)
            
        Returns:
            CBF maps in ml/100g/min
        """
        try:
            # Avoid division by zero
            m0_safe = np.where(m0_image > 0, m0_image, 1.0)
            
            # Calculate the denominator constant
            denom_const = (2 * self.alpha * self.t1_blood * 
                          (1 - np.exp(-self.tau / self.t1_blood)))
            
            # Calculate CBF for each perfusion image
            if perfusion_images.ndim == 4:
                cbf_maps = np.zeros_like(perfusion_images)
                for i in range(perfusion_images.shape[3]):
                    delta_m = perfusion_images[:, :, :, i]
                    cbf_maps[:, :, :, i] = (6000 * self.lambda_val * delta_m * 
                                           np.exp(self.pld / self.t1_blood)) / (denom_const * m0_safe)
            else:
                # Single 3D image
                delta_m = perfusion_images
                cbf_maps = (6000 * self.lambda_val * delta_m * 
                           np.exp(self.pld / self.t1_blood)) / (denom_const * m0_safe)
            
            # Apply brain mask if provided
            if mask is not None:
                if cbf_maps.ndim == 4:
                    for i in range(cbf_maps.shape[3]):
                        cbf_maps[:, :, :, i] = cbf_maps[:, :, :, i] * mask
                else:
                    cbf_maps = cbf_maps * mask
            
            # Remove extreme values
            cbf_maps = np.clip(cbf_maps, -200, 300)  # Physiological range
            
            logger.info(f"Calculated CBF maps with range: {np.min(cbf_maps):.2f} to {np.max(cbf_maps):.2f} ml/100g/min")
            
            return cbf_maps
            
        except Exception as e:
            logger.error(f"Error in CBF calculation: {e}")
            raise CBFCalculationError(f"CBF calculation failed: {e}")
    
    def calculate_cbf_kinetic_model(self, 
                                  perfusion_images: np.ndarray,
                                  m0_image: np.ndarray,
                                  mask: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate CBF using kinetic modeling with multiple PLDs or tau values.
        
        This implements a more sophisticated approach similar to BASIL's kinetic modeling.
        
        Args:
            perfusion_images: Perfusion-weighted difference images (4D with multiple timepoints)
            m0_image: M0 calibration image
            mask: Brain mask
            
        Returns:
            Dictionary with CBF maps and kinetic parameters
        """
        if perfusion_images.ndim != 4:
            logger.warning("Kinetic modeling requires 4D data, falling back to simple model")
            cbf_simple = self.calculate_cbf_simple(perfusion_images, m0_image, mask)
            return {'cbf': cbf_simple, 'model_type': 'simple'}
        
        try:
            # Prepare output arrays
            shape_3d = perfusion_images.shape[:3]
            cbf_map = np.zeros(shape_3d)
            arterial_transit_time = np.zeros(shape_3d)
            model_fit_quality = np.zeros(shape_3d)
            
            # Get mask indices for faster processing
            if mask is not None:
                mask_indices = np.where(mask > 0.5)
            else:
                mask_indices = np.where(np.ones(shape_3d, dtype=bool))
            
            n_voxels = len(mask_indices[0])
            logger.info(f"Processing {n_voxels} voxels for kinetic modeling")
            
            # Process each voxel
            for idx in range(n_voxels):
                i, j, k = mask_indices[0][idx], mask_indices[1][idx], mask_indices[2][idx]
                
                if m0_image[i, j, k] <= 0:
                    continue
                
                # Extract time series for this voxel
                voxel_timeseries = perfusion_images[i, j, k, :]
                
                # Fit kinetic model
                try:
                    cbf_val, att_val, fit_quality = self._fit_kinetic_model_voxel(
                        voxel_timeseries, m0_image[i, j, k])
                    
                    cbf_map[i, j, k] = cbf_val
                    arterial_transit_time[i, j, k] = att_val
                    model_fit_quality[i, j, k] = fit_quality
                    
                except Exception:
                    # Fall back to simple calculation for this voxel
                    mean_perf = np.mean(voxel_timeseries)
                    cbf_fallback = (6000 * self.lambda_val * mean_perf * 
                                   np.exp(self.pld / self.t1_blood)) / (
                                   2 * self.alpha * self.t1_blood * 
                                   (1 - np.exp(-self.tau / self.t1_blood)) * m0_image[i, j, k])
                    cbf_map[i, j, k] = np.clip(cbf_fallback, -200, 300)
            
            # Apply final clipping and masking
            cbf_map = np.clip(cbf_map, -50, 200)
            arterial_transit_time = np.clip(arterial_transit_time, 0, 3.0)
            
            if mask is not None:
                cbf_map *= mask
                arterial_transit_time *= mask
                model_fit_quality *= mask
            
            logger.info(f"Kinetic modeling completed. CBF range: {np.min(cbf_map[cbf_map>0]):.2f} to {np.max(cbf_map):.2f} ml/100g/min")
            
            return {
                'cbf': cbf_map,
                'arterial_transit_time': arterial_transit_time,
                'model_fit_quality': model_fit_quality,
                'model_type': 'kinetic'
            }
            
        except Exception as e:
            logger.error(f"Error in kinetic modeling: {e}")
            raise CBFCalculationError(f"Kinetic modeling failed: {e}")
    
    def calculate_regional_cbf(self, 
                             cbf_map: np.ndarray,
                             atlas: Optional[np.ndarray] = None,
                             region_names: Optional[Dict[int, str]] = None) -> Dict:
        """
        Calculate regional CBF values using anatomical atlas.
        
        Args:
            cbf_map: CBF map
            atlas: Anatomical atlas with integer region labels
            region_names: Dictionary mapping region labels to names
            
        Returns:
            Dictionary with regional CBF statistics
        """
        regional_stats = {}
        
        if atlas is None:
            # Calculate whole-brain statistics
            valid_cbf = cbf_map[cbf_map > 0]
            if len(valid_cbf) > 0:
                regional_stats['whole_brain'] = {
                    'mean_cbf': float(np.mean(valid_cbf)),
                    'std_cbf': float(np.std(valid_cbf)),
                    'median_cbf': float(np.median(valid_cbf)),
                    'percentile_95': float(np.percentile(valid_cbf, 95)),
                    'percentile_05': float(np.percentile(valid_cbf, 5)),
                    'voxel_count': int(len(valid_cbf))
                }
            return regional_stats
        
        # Calculate statistics for each region
        unique_regions = np.unique(atlas[atlas > 0])
        
        for region_id in unique_regions:
            region_mask = atlas == region_id
            region_cbf = cbf_map[region_mask]
            valid_region_cbf = region_cbf[region_cbf > 0]
            
            if len(valid_region_cbf) > 0:
                region_name = region_names.get(int(region_id), f"Region_{int(region_id)}") if region_names else f"Region_{int(region_id)}"
                
                regional_stats[region_name] = {
                    'region_id': int(region_id),
                    'mean_cbf': float(np.mean(valid_region_cbf)),
                    'std_cbf': float(np.std(valid_region_cbf)),
                    'median_cbf': float(np.median(valid_region_cbf)),
                    'percentile_95': float(np.percentile(valid_region_cbf, 95)),
                    'percentile_05': float(np.percentile(valid_region_cbf, 5)),
                    'voxel_count': int(len(valid_region_cbf))
                }
        
        logger.info(f"Calculated regional CBF for {len(regional_stats)} regions")
        
        return regional_stats
    
    def apply_partial_volume_correction(self, 
                                      cbf_map: np.ndarray,
                                      gm_pv: np.ndarray,
                                      wm_pv: np.ndarray,
                                      csf_pv: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply partial volume correction to CBF maps.
        
        Args:
            cbf_map: Uncorrected CBF map
            gm_pv: Gray matter partial volume map
            wm_pv: White matter partial volume map
            csf_pv: CSF partial volume map (optional)
            
        Returns:
            Partial volume corrected CBF map
        """
        try:
            # Simple two-compartment correction
            # Assume CBF in WM is 20% of GM CBF
            wm_cbf_fraction = 0.2
            
            # Calculate effective tissue fraction
            tissue_fraction = gm_pv + wm_cbf_fraction * wm_pv
            
            # Avoid division by zero
            tissue_fraction_safe = np.where(tissue_fraction > 0.1, tissue_fraction, 1.0)
            
            # Apply correction
            cbf_corrected = cbf_map * gm_pv / tissue_fraction_safe
            
            # Clip to reasonable range
            cbf_corrected = np.clip(cbf_corrected, 0, 300)
            
            logger.info("Applied partial volume correction")
            
            return cbf_corrected
            
        except Exception as e:
            logger.error(f"Error in partial volume correction: {e}")
            return cbf_map  # Return uncorrected map on error
    
    def _fit_kinetic_model_voxel(self, 
                               timeseries: np.ndarray, 
                               m0_value: float) -> Tuple[float, float, float]:
        """
        Fit kinetic model to a single voxel time series.
        
        Args:
            timeseries: Perfusion time series for one voxel
            m0_value: M0 value for this voxel
            
        Returns:
            Tuple of (CBF, arterial_transit_time, fit_quality)
        """
        def kinetic_model(params, pld_times):
            cbf, att = params
            
            # Simple kinetic model
            model_signal = np.zeros_like(pld_times)
            for i, pld in enumerate(pld_times):
                if pld > att:
                    # After arrival
                    model_signal[i] = (cbf * 2 * self.alpha * self.t1_blood * 
                                     (1 - np.exp(-self.tau / self.t1_blood)) * 
                                     np.exp(-(pld - att) / self.t1_blood))
                else:
                    # Before arrival
                    model_signal[i] = 0
            
            return model_signal
        
        def objective(params):
            if params[0] < 0 or params[1] < 0 or params[1] > 3.0:
                return 1e6
            
            # Assume PLDs from 1.0 to 3.0 seconds
            n_points = len(timeseries)
            pld_times = np.linspace(1.0, 3.0, n_points)
            
            model_pred = kinetic_model(params, pld_times)
            
            # Scale by M0
            model_pred_scaled = model_pred / m0_value if m0_value > 0 else model_pred
            
            # Calculate residual
            residual = np.sum((timeseries - model_pred_scaled) ** 2)
            
            return residual
        
        # Initial guess: CBF=50, ATT=1.2
        x0 = [50.0, 1.2]
        bounds = [(0, 200), (0.3, 3.0)]
        
        try:
            result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
            cbf_est, att_est = result.x
            fit_quality = 1.0 / (1.0 + result.fun)  # Convert residual to quality metric
            
            return float(cbf_est), float(att_est), float(fit_quality)
            
        except Exception:
            # Return fallback values
            return 0.0, 1.2, 0.0
    
    def calculate_cbf_statistics(self, cbf_map: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate comprehensive CBF statistics.
        
        Args:
            cbf_map: CBF map
            mask: Brain mask (optional)
            
        Returns:
            Dictionary with CBF statistics
        """
        if mask is not None:
            cbf_values = cbf_map[mask > 0.5]
        else:
            cbf_values = cbf_map.flatten()
        
        # Remove zero and negative values
        valid_cbf = cbf_values[cbf_values > 0]
        
        if len(valid_cbf) == 0:
            logger.warning("No valid CBF values found")
            return {}
        
        # Calculate comprehensive statistics
        stats = {
            'mean_cbf': float(np.mean(valid_cbf)),
            'median_cbf': float(np.median(valid_cbf)),
            'std_cbf': float(np.std(valid_cbf)),
            'min_cbf': float(np.min(valid_cbf)),
            'max_cbf': float(np.max(valid_cbf)),
            'percentile_05': float(np.percentile(valid_cbf, 5)),
            'percentile_25': float(np.percentile(valid_cbf, 25)),
            'percentile_75': float(np.percentile(valid_cbf, 75)),
            'percentile_95': float(np.percentile(valid_cbf, 95)),
            'n_voxels': int(len(valid_cbf)),
            'coefficient_of_variation': float(np.std(valid_cbf) / np.mean(valid_cbf)),
            'skewness': float(self._calculate_skewness(valid_cbf)),
            'kurtosis': float(self._calculate_kurtosis(valid_cbf))
        }
        
        # Calculate quality metrics
        stats['abnormal_high_cbf'] = int(np.sum(valid_cbf > 150))  # Abnormally high CBF
        stats['abnormal_low_cbf'] = int(np.sum(valid_cbf < 10))   # Abnormally low CBF
        stats['normal_range_percentage'] = float(np.sum((valid_cbf >= 20) & (valid_cbf <= 100)) / len(valid_cbf) * 100)
        
        logger.info(f"CBF Statistics: Mean={stats['mean_cbf']:.2f}, Std={stats['std_cbf']:.2f} ml/100g/min")
        
        return stats
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0.0
        return np.mean(((data - mean_val) / std_val) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0.0
        return np.mean(((data - mean_val) / std_val) ** 4) - 3.0