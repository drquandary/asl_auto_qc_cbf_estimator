"""
Quality Assessor Module for ASL Auto-QC CBF Estimator.

This module provides comprehensive quality control metrics for ASL data
including signal-to-noise ratio, motion assessment, temporal stability,
perfusion territory analysis, and outlier detection.
"""

import logging
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
from scipy import ndimage, stats
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class QualityAssessmentError(Exception):
    """Exception raised for quality assessment errors."""
    pass


class ASLQualityAssessor:
    """
    Comprehensive quality assessment for ASL data.
    
    Implements multiple QC metrics including SNR analysis, motion assessment,
    temporal stability, spatial coherence, and statistical outlier detection.
    """
    
    def __init__(self, 
                 snr_threshold: float = 3.0,
                 temporal_snr_threshold: float = 5.0,
                 motion_threshold: float = 2.0,
                 outlier_threshold: float = 2.5):
        """
        Initialize quality assessor.
        
        Args:
            snr_threshold: Minimum acceptable SNR
            temporal_snr_threshold: Minimum acceptable temporal SNR
            motion_threshold: Motion threshold in mm
            outlier_threshold: Z-score threshold for outlier detection
        """
        self.snr_threshold = snr_threshold
        self.temporal_snr_threshold = temporal_snr_threshold
        self.motion_threshold = motion_threshold
        self.outlier_threshold = outlier_threshold
        
        logger.info("ASL Quality Assessor initialized")
    
    def assess_asl_quality(self, 
                          processed_data: Dict,
                          cbf_results: Optional[Dict] = None) -> Dict:
        """
        Comprehensive quality assessment of ASL data.
        
        Args:
            processed_data: Dictionary from ASL preprocessing
            cbf_results: Optional CBF calculation results
            
        Returns:
            Dictionary with comprehensive quality metrics
        """
        try:
            quality_metrics = {}
            
            # Basic data quality checks
            quality_metrics.update(self._assess_data_quality(processed_data))
            
            # Signal quality assessment
            quality_metrics.update(self._assess_signal_quality(processed_data))
            
            # Motion assessment
            quality_metrics.update(self._assess_motion_quality(processed_data))
            
            # Temporal stability assessment
            quality_metrics.update(self._assess_temporal_stability(processed_data))
            
            # Spatial coherence assessment
            quality_metrics.update(self._assess_spatial_coherence(processed_data))
            
            # CBF-specific quality metrics if available
            if cbf_results is not None:
                quality_metrics.update(self._assess_cbf_quality(cbf_results))
            
            # Overall quality score
            overall_score = self._calculate_overall_quality_score(quality_metrics)
            quality_metrics['overall_quality_score'] = overall_score
            quality_metrics['overall_quality_grade'] = self._grade_quality(overall_score)
            
            # Quality report summary
            quality_metrics['quality_summary'] = self._generate_quality_summary(quality_metrics)
            
            logger.info(f"Quality assessment completed. Overall grade: {quality_metrics['overall_quality_grade']}")
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error in quality assessment: {e}")
            raise QualityAssessmentError(f"Quality assessment failed: {e}")
    
    def _assess_data_quality(self, processed_data: Dict) -> Dict:
        """Assess basic data quality metrics."""
        metrics = {}
        
        asl_data = processed_data['asl_data']
        mask = processed_data['mask_data']
        
        # Basic data integrity checks
        metrics['data_shape'] = asl_data.shape
        metrics['n_volumes'] = processed_data['n_volumes']
        metrics['n_pairs'] = processed_data['n_pairs']
        metrics['voxel_size'] = list(processed_data['voxel_size'])
        
        # Check for data corruption
        metrics['nan_voxels'] = int(np.sum(np.isnan(asl_data)))
        metrics['inf_voxels'] = int(np.sum(np.isinf(asl_data)))
        metrics['zero_volumes'] = int(np.sum(np.sum(asl_data, axis=(0,1,2)) == 0))
        
        # Brain mask coverage
        total_voxels = np.prod(asl_data.shape[:3])
        brain_voxels = np.sum(mask > 0.5)
        metrics['brain_mask_coverage'] = float(brain_voxels / total_voxels)
        
        # Data range checks
        brain_data = asl_data[mask > 0.5, :]
        if brain_data.size > 0:
            metrics['signal_range'] = {
                'min': float(np.min(brain_data)),
                'max': float(np.max(brain_data)),
                'mean': float(np.mean(brain_data)),
                'std': float(np.std(brain_data))
            }
        
        # Data quality flags
        metrics['data_quality_flags'] = {
            'has_nan': metrics['nan_voxels'] > 0,
            'has_inf': metrics['inf_voxels'] > 0,
            'has_zero_volumes': metrics['zero_volumes'] > 0,
            'low_brain_coverage': metrics['brain_mask_coverage'] < 0.3
        }
        
        return {'data_quality': metrics}
    
    def _assess_signal_quality(self, processed_data: Dict) -> Dict:
        """Assess signal quality metrics including SNR."""
        metrics = {}
        
        control_volumes = processed_data['control_volumes']
        label_volumes = processed_data['label_volumes']
        perfusion_images = processed_data['perfusion_images']
        mask = processed_data['mask_data']
        
        # Calculate SNR for control and label images
        mean_control = np.mean(control_volumes, axis=3)
        std_control = np.std(control_volumes, axis=3)
        
        mean_label = np.mean(label_volumes, axis=3)
        std_label = np.std(label_volumes, axis=3)
        
        # Temporal SNR (mean/std across time)
        brain_indices = mask > 0.5
        
        if np.any(brain_indices):
            control_tsnr = mean_control[brain_indices] / (std_control[brain_indices] + 1e-8)
            label_tsnr = mean_label[brain_indices] / (std_label[brain_indices] + 1e-8)
            
            metrics['temporal_snr'] = {
                'control_mean': float(np.mean(control_tsnr[control_tsnr > 0])),
                'control_std': float(np.std(control_tsnr[control_tsnr > 0])),
                'label_mean': float(np.mean(label_tsnr[label_tsnr > 0])),
                'label_std': float(np.std(label_tsnr[label_tsnr > 0])),
                'control_median': float(np.median(control_tsnr[control_tsnr > 0])),
                'label_median': float(np.median(label_tsnr[label_tsnr > 0]))
            }
        
        # Perfusion-weighted SNR
        mean_perfusion = processed_data['mean_perfusion']
        std_perfusion_temporal = np.std(perfusion_images, axis=3)
        
        if np.any(brain_indices):
            perfusion_snr = np.abs(mean_perfusion[brain_indices]) / (std_perfusion_temporal[brain_indices] + 1e-8)
            valid_snr = perfusion_snr[perfusion_snr > 0]
            
            if len(valid_snr) > 0:
                metrics['perfusion_snr'] = {
                    'mean': float(np.mean(valid_snr)),
                    'std': float(np.std(valid_snr)),
                    'median': float(np.median(valid_snr)),
                    'percentile_95': float(np.percentile(valid_snr, 95)),
                    'percentile_05': float(np.percentile(valid_snr, 5))
                }
            
        # Signal stability assessment
        metrics['signal_stability'] = self._assess_signal_stability(processed_data)
        
        # SNR quality flags
        control_snr_good = metrics.get('temporal_snr', {}).get('control_mean', 0) > self.temporal_snr_threshold
        perfusion_snr_good = metrics.get('perfusion_snr', {}).get('mean', 0) > self.snr_threshold
        
        metrics['snr_quality_flags'] = {
            'control_snr_adequate': control_snr_good,
            'perfusion_snr_adequate': perfusion_snr_good,
            'overall_snr_adequate': control_snr_good and perfusion_snr_good
        }
        
        return {'signal_quality': metrics}
    
    def _assess_motion_quality(self, processed_data: Dict) -> Dict:
        """Assess motion-related quality metrics."""
        metrics = {}
        
        asl_data = processed_data['asl_data']
        
        # Calculate frame-to-frame motion estimates
        motion_estimates = []
        reference_volume = asl_data[:, :, :, 0]
        
        for i in range(1, asl_data.shape[3]):
            current_volume = asl_data[:, :, :, i]
            
            # Simple motion estimate using normalized cross-correlation
            correlation = np.corrcoef(reference_volume.flatten(), current_volume.flatten())[0, 1]
            motion_metric = 1.0 - correlation
            motion_estimates.append(motion_metric)
        
        motion_estimates = np.array(motion_estimates)
        
        metrics['motion_estimates'] = motion_estimates.tolist()
        metrics['mean_motion'] = float(np.mean(motion_estimates))
        metrics['max_motion'] = float(np.max(motion_estimates))
        metrics['std_motion'] = float(np.std(motion_estimates))
        
        # Motion outlier detection
        motion_threshold = np.mean(motion_estimates) + 2 * np.std(motion_estimates)
        motion_outliers = np.where(motion_estimates > motion_threshold)[0]
        metrics['motion_outlier_volumes'] = motion_outliers.tolist()
        metrics['n_motion_outliers'] = len(motion_outliers)
        
        # Control-label specific motion analysis
        control_motion = motion_estimates[::2]  # Even indices
        label_motion = motion_estimates[1::2]   # Odd indices
        
        metrics['control_motion'] = {
            'mean': float(np.mean(control_motion)),
            'std': float(np.std(control_motion)),
            'max': float(np.max(control_motion))
        }
        
        metrics['label_motion'] = {
            'mean': float(np.mean(label_motion)),
            'std': float(np.std(label_motion)),
            'max': float(np.max(label_motion))
        }
        
        # Motion quality assessment
        metrics['motion_quality_score'] = self._calculate_motion_quality_score(motion_estimates)
        metrics['motion_quality_grade'] = self._grade_motion_quality(metrics['motion_quality_score'])
        
        return {'motion_quality': metrics}
    
    def _assess_temporal_stability(self, processed_data: Dict) -> Dict:
        """Assess temporal stability of ASL signal."""
        metrics = {}
        
        perfusion_images = processed_data['perfusion_images']
        mask = processed_data['mask_data']
        
        if perfusion_images.shape[3] < 3:
            metrics['insufficient_timepoints'] = True
            return {'temporal_stability': metrics}
        
        # Calculate temporal statistics within brain mask
        brain_indices = mask > 0.5
        brain_perfusion_timeseries = perfusion_images[brain_indices, :]
        
        # Mean perfusion signal over time
        mean_perfusion_timeseries = np.mean(brain_perfusion_timeseries, axis=0)
        
        # Temporal stability metrics
        metrics['mean_signal_timeseries'] = mean_perfusion_timeseries.tolist()
        metrics['temporal_mean'] = float(np.mean(mean_perfusion_timeseries))
        metrics['temporal_std'] = float(np.std(mean_perfusion_timeseries))
        metrics['temporal_cov'] = float(np.std(mean_perfusion_timeseries) / (abs(np.mean(mean_perfusion_timeseries)) + 1e-8))
        
        # Drift assessment using linear trend
        timepoints = np.arange(len(mean_perfusion_timeseries))
        slope, intercept, r_value, p_value, std_err = stats.linregress(timepoints, mean_perfusion_timeseries)
        
        metrics['temporal_drift'] = {
            'slope': float(slope),
            'r_squared': float(r_value ** 2),
            'p_value': float(p_value),
            'significant_drift': p_value < 0.05
        }
        
        # Temporal outlier detection
        z_scores = np.abs(stats.zscore(mean_perfusion_timeseries))
        temporal_outliers = np.where(z_scores > self.outlier_threshold)[0]
        
        metrics['temporal_outliers'] = temporal_outliers.tolist()
        metrics['n_temporal_outliers'] = len(temporal_outliers)
        
        # Stability quality score
        stability_score = 100 * (1 - min(metrics['temporal_cov'], 1.0))  # Lower COV = higher score
        if metrics['temporal_drift']['significant_drift']:
            stability_score *= 0.8  # Penalize significant drift
        
        metrics['temporal_stability_score'] = float(stability_score)
        metrics['temporal_stability_grade'] = self._grade_temporal_stability(stability_score)
        
        return {'temporal_stability': metrics}
    
    def _assess_spatial_coherence(self, processed_data: Dict) -> Dict:
        """Assess spatial coherence and patterns in perfusion data."""
        metrics = {}
        
        mean_perfusion = processed_data['mean_perfusion']
        mask = processed_data['mask_data']
        
        # Apply mask
        masked_perfusion = mean_perfusion * mask
        
        # Spatial smoothness assessment
        laplacian_map = ndimage.laplace(masked_perfusion)
        spatial_roughness = np.std(laplacian_map[mask > 0.5])
        
        metrics['spatial_roughness'] = float(spatial_roughness)
        
        # Check for expected perfusion patterns
        # Gray matter should have higher perfusion than white matter
        gm_estimate = self._estimate_gray_matter_mask(mean_perfusion, mask)
        wm_estimate = mask & ~gm_estimate
        
        if np.sum(gm_estimate) > 0 and np.sum(wm_estimate) > 0:
            gm_perfusion = np.mean(mean_perfusion[gm_estimate])
            wm_perfusion = np.mean(mean_perfusion[wm_estimate])
            
            metrics['gm_wm_contrast'] = {
                'gm_mean_perfusion': float(gm_perfusion),
                'wm_mean_perfusion': float(wm_perfusion),
                'gm_wm_ratio': float(gm_perfusion / (wm_perfusion + 1e-8)),
                'expected_contrast': gm_perfusion > wm_perfusion
            }
        
        # Spatial clustering analysis
        brain_coords = np.where(mask > 0.5)
        if len(brain_coords[0]) > 100:  # Sufficient points for clustering
            brain_values = mean_perfusion[brain_coords]
            
            # Detect spatial outliers using DBSCAN
            coords_values = np.column_stack([brain_coords[0], brain_coords[1], brain_coords[2], brain_values])
            scaler = StandardScaler()
            coords_values_scaled = scaler.fit_transform(coords_values)
            
            try:
                clustering = DBSCAN(eps=0.5, min_samples=10).fit(coords_values_scaled)
                n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
                n_noise = list(clustering.labels_).count(-1)
                
                metrics['spatial_clustering'] = {
                    'n_clusters': n_clusters,
                    'n_outliers': n_noise,
                    'outlier_percentage': float(n_noise / len(clustering.labels_) * 100)
                }
            except Exception:
                metrics['spatial_clustering'] = {'analysis_failed': True}
        
        # Overall spatial quality score
        spatial_score = 100
        if 'gm_wm_contrast' in metrics and not metrics['gm_wm_contrast']['expected_contrast']:
            spatial_score -= 30
        if 'spatial_clustering' in metrics and metrics['spatial_clustering'].get('outlier_percentage', 0) > 20:
            spatial_score -= 20
        
        metrics['spatial_coherence_score'] = float(max(0, spatial_score))
        metrics['spatial_coherence_grade'] = self._grade_spatial_coherence(spatial_score)
        
        return {'spatial_coherence': metrics}
    
    def _assess_cbf_quality(self, cbf_results: Dict) -> Dict:
        """Assess CBF-specific quality metrics."""
        metrics = {}
        
        cbf_map = cbf_results.get('cbf')
        if cbf_map is None:
            return {'cbf_quality': {'no_cbf_data': True}}
        
        # CBF value distribution analysis
        valid_cbf = cbf_map[cbf_map > 0]
        
        if len(valid_cbf) > 0:
            metrics['cbf_distribution'] = {
                'mean': float(np.mean(valid_cbf)),
                'median': float(np.median(valid_cbf)),
                'std': float(np.std(valid_cbf)),
                'skewness': float(stats.skew(valid_cbf)),
                'kurtosis': float(stats.kurtosis(valid_cbf))
            }
            
            # Physiological range assessment
            normal_range = (valid_cbf >= 20) & (valid_cbf <= 100)
            high_cbf = valid_cbf > 150
            low_cbf = valid_cbf < 10
            
            metrics['physiological_assessment'] = {
                'normal_range_percentage': float(np.sum(normal_range) / len(valid_cbf) * 100),
                'abnormally_high_percentage': float(np.sum(high_cbf) / len(valid_cbf) * 100),
                'abnormally_low_percentage': float(np.sum(low_cbf) / len(valid_cbf) * 100)
            }
            
            # CBF quality score based on physiological plausibility
            cbf_quality_score = metrics['physiological_assessment']['normal_range_percentage']
            
            # Penalize extreme values
            if metrics['physiological_assessment']['abnormally_high_percentage'] > 10:
                cbf_quality_score -= 20
            if metrics['physiological_assessment']['abnormally_low_percentage'] > 10:
                cbf_quality_score -= 20
            
            metrics['cbf_quality_score'] = float(max(0, cbf_quality_score))
            metrics['cbf_quality_grade'] = self._grade_cbf_quality(cbf_quality_score)
        
        return {'cbf_quality': metrics}
    
    def _assess_signal_stability(self, processed_data: Dict) -> Dict:
        """Assess signal stability across control-label pairs."""
        control_volumes = processed_data['control_volumes']
        label_volumes = processed_data['label_volumes']
        mask = processed_data['mask_data']
        
        brain_indices = mask > 0.5
        
        # Calculate signal stability for control and label separately
        control_means = []
        label_means = []
        
        for i in range(control_volumes.shape[3]):
            control_means.append(np.mean(control_volumes[brain_indices, i]))
            label_means.append(np.mean(label_volumes[brain_indices, i]))
        
        control_means = np.array(control_means)
        label_means = np.array(label_means)
        
        stability_metrics = {
            'control_signal_cov': float(np.std(control_means) / (np.mean(control_means) + 1e-8)),
            'label_signal_cov': float(np.std(label_means) / (np.mean(label_means) + 1e-8)),
            'control_signal_drift': float(np.polyfit(range(len(control_means)), control_means, 1)[0]),
            'label_signal_drift': float(np.polyfit(range(len(label_means)), label_means, 1)[0])
        }
        
        return stability_metrics
    
    def _estimate_gray_matter_mask(self, perfusion_image: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
        """Rough estimation of gray matter mask based on perfusion intensity."""
        # Simple threshold-based approach
        # Gray matter typically has higher perfusion
        brain_perfusion = perfusion_image[brain_mask > 0.5]
        if len(brain_perfusion) == 0:
            return np.zeros_like(brain_mask, dtype=bool)
        
        threshold = np.percentile(brain_perfusion, 60)  # Top 40% of perfusion values
        gm_mask = (perfusion_image > threshold) & (brain_mask > 0.5)
        
        return gm_mask
    
    def _calculate_motion_quality_score(self, motion_estimates: np.ndarray) -> float:
        """Calculate motion quality score (0-100)."""
        max_motion = np.max(motion_estimates)
        mean_motion = np.mean(motion_estimates)
        
        # Score based on maximum and mean motion
        score = 100 * (1 - min(max_motion / 0.5, 1.0))  # Normalize to 0.5 threshold
        score *= (1 - min(mean_motion / 0.2, 1.0))      # Penalize high mean motion
        
        return float(max(0, score))
    
    def _calculate_overall_quality_score(self, quality_metrics: Dict) -> float:
        """Calculate overall quality score from individual metrics."""
        scores = []
        weights = []
        
        # Signal quality (30% weight)
        snr_adequate = quality_metrics.get('signal_quality', {}).get('snr_quality_flags', {}).get('overall_snr_adequate', False)
        if snr_adequate:
            scores.append(80.0)
        else:
            scores.append(40.0)
        weights.append(0.3)
        
        # Motion quality (25% weight)
        motion_score = quality_metrics.get('motion_quality', {}).get('motion_quality_score', 50.0)
        scores.append(motion_score)
        weights.append(0.25)
        
        # Temporal stability (25% weight)
        temporal_score = quality_metrics.get('temporal_stability', {}).get('temporal_stability_score', 50.0)
        scores.append(temporal_score)
        weights.append(0.25)
        
        # Spatial coherence (20% weight)
        spatial_score = quality_metrics.get('spatial_coherence', {}).get('spatial_coherence_score', 50.0)
        scores.append(spatial_score)
        weights.append(0.20)
        
        # Calculate weighted average
        overall_score = np.average(scores, weights=weights)
        
        return float(overall_score)
    
    def _grade_quality(self, score: float) -> str:
        """Convert quality score to letter grade."""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def _grade_motion_quality(self, score: float) -> str:
        """Grade motion quality."""
        return self._grade_quality(score)
    
    def _grade_temporal_stability(self, score: float) -> str:
        """Grade temporal stability."""
        return self._grade_quality(score)
    
    def _grade_spatial_coherence(self, score: float) -> str:
        """Grade spatial coherence."""
        return self._grade_quality(score)
    
    def _grade_cbf_quality(self, score: float) -> str:
        """Grade CBF quality."""
        return self._grade_quality(score)
    
    def _generate_quality_summary(self, quality_metrics: Dict) -> Dict:
        """Generate human-readable quality summary."""
        summary = {
            'overall_grade': quality_metrics.get('overall_quality_grade', 'Unknown'),
            'major_issues': [],
            'recommendations': [],
            'data_usability': 'unknown'
        }
        
        # Check for major issues
        data_flags = quality_metrics.get('data_quality', {}).get('data_quality_flags', {})
        if data_flags.get('has_nan') or data_flags.get('has_inf'):
            summary['major_issues'].append("Data corruption detected (NaN/Inf values)")
        
        if data_flags.get('low_brain_coverage'):
            summary['major_issues'].append("Low brain mask coverage")
        
        motion_grade = quality_metrics.get('motion_quality', {}).get('motion_quality_grade', 'Unknown')
        if motion_grade in ['D', 'F']:
            summary['major_issues'].append("Excessive motion detected")
            summary['recommendations'].append("Consider motion correction or subject exclusion")
        
        snr_adequate = quality_metrics.get('signal_quality', {}).get('snr_quality_flags', {}).get('overall_snr_adequate', False)
        if not snr_adequate:
            summary['major_issues'].append("Low signal-to-noise ratio")
            summary['recommendations'].append("Check acquisition parameters and coil performance")
        
        # Determine data usability
        overall_grade = quality_metrics.get('overall_quality_grade', 'F')
        if overall_grade in ['A', 'B']:
            summary['data_usability'] = 'excellent'
        elif overall_grade == 'C':
            summary['data_usability'] = 'acceptable'
        elif overall_grade == 'D':
            summary['data_usability'] = 'marginal'
        else:
            summary['data_usability'] = 'poor'
        
        if not summary['major_issues']:
            summary['major_issues'] = ["No major issues detected"]
        
        if not summary['recommendations']:
            summary['recommendations'] = ["Data appears suitable for analysis"]
        
        return summary