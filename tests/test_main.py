"""
Comprehensive test suite for ASL Auto-QC CBF Estimator.

This module contains tests for the main application functionality,
including API endpoints, ASL processing, CBF calculation, and quality assessment.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.asl_processor import ASLProcessor, ASLDataError
from app.cbf_calculator import CBFCalculator, CBFCalculationError  
from app.quality_assessor import ASLQualityAssessor, QualityAssessmentError
from app.models import ASLProcessingParameters, ProcessingRequest
from app.config import get_settings

client = TestClient(app)


@pytest.fixture
def settings():
    """Get application settings."""
    return get_settings()


@pytest.fixture
def sample_asl_data():
    """Create sample ASL data for testing."""
    # Create synthetic 4D ASL data (64x64x32x20)
    asl_data = np.random.randn(64, 64, 32, 20) * 100 + 1000
    
    # Add some structure to make it more realistic
    # Create control-label pairs with small differences
    for i in range(0, 20, 2):
        # Control volume
        control = asl_data[:, :, :, i]
        # Label volume (slightly lower signal)
        asl_data[:, :, :, i + 1] = control - np.random.randn(64, 64, 32) * 10
    
    return asl_data


@pytest.fixture 
def sample_m0_data():
    """Create sample M0 calibration data."""
    return np.random.randn(64, 64, 32) * 200 + 2000


@pytest.fixture
def sample_mask_data():
    """Create sample brain mask."""
    mask = np.zeros((64, 64, 32))
    # Create a brain-like mask (ellipsoid)
    center = (32, 32, 16)
    for i in range(64):
        for j in range(64):
            for k in range(32):
                dist = ((i - center[0])/30)**2 + ((j - center[1])/30)**2 + ((k - center[2])/15)**2
                if dist <= 1:
                    mask[i, j, k] = 1
    return mask.astype(bool)


@pytest.fixture
def temp_nifti_file(sample_asl_data):
    """Create temporary NIfTI file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as f:
        # In a real implementation, you would use nibabel to create a proper NIfTI file
        # For testing, we'll just create a mock file
        temp_path = Path(f.name)
        
    # Mock the nibabel loading
    mock_img = Mock()
    mock_img.get_fdata.return_value = sample_asl_data
    mock_img.header.get_zooms.return_value = (3.0, 3.0, 3.0, 2.0)
    
    yield temp_path, mock_img
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


class TestMainApplication:
    """Test main application endpoints."""
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get('/')
        assert response.status_code == 200
        assert 'ASL Auto-QC and CBF Estimator' in response.json().get('detail', '')
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = client.get('/api/health')
        assert response.status_code == 200
        data = response.json()
        assert data.get('status') == 'ok'
        assert 'timestamp' in data
        assert 'version' in data
        assert 'dependencies' in data
    
    @patch('app.routes.processing_results', {})
    def test_get_processing_results_not_found(self):
        """Test getting results for non-existent processing ID."""
        response = client.get('/api/results/nonexistent-id')
        assert response.status_code == 404
    
    def test_list_all_results_empty(self):
        """Test listing results when none exist."""
        with patch('app.routes.processing_results', {}):
            response = client.get('/api/results')
            assert response.status_code == 200
            assert response.json() == []


class TestASLProcessor:
    """Test ASL data processing functionality."""
    
    def test_processor_initialization(self):
        """Test ASL processor initialization."""
        processor = ASLProcessor(
            temporal_filter=True,
            spatial_smoothing=True,
            motion_threshold=2.0
        )
        assert processor.temporal_filter is True
        assert processor.spatial_smoothing is True
        assert processor.motion_threshold == 2.0
    
    def test_load_asl_data_file_not_found(self):
        """Test loading ASL data with non-existent file."""
        processor = ASLProcessor()
        
        with pytest.raises(ASLDataError, match="ASL file not found"):
            processor.load_asl_data("/nonexistent/path.nii.gz")
    
    @patch('nibabel.load')
    def test_load_asl_data_success(self, mock_load, sample_asl_data, sample_mask_data):
        """Test successful ASL data loading."""
        # Mock nibabel loading
        mock_img = Mock()
        mock_img.get_fdata.return_value = sample_asl_data
        mock_img.header.get_zooms.return_value = (3.0, 3.0, 3.0, 2.0)
        mock_load.return_value = mock_img
        
        processor = ASLProcessor()
        
        with tempfile.NamedTemporaryFile(suffix='.nii.gz') as f:
            result = processor.load_asl_data(f.name)
            
            assert 'asl_data' in result
            assert 'n_volumes' in result
            assert 'n_pairs' in result
            assert result['n_volumes'] == 20
            assert result['n_pairs'] == 10
            assert result['voxel_size'] == (3.0, 3.0, 3.0)
    
    def test_separate_control_label_pairs(self, sample_asl_data):
        """Test separation of control and label pairs."""
        processor = ASLProcessor()
        
        control, label = processor.separate_control_label_pairs(sample_asl_data)
        
        assert control.shape == (64, 64, 32, 10)
        assert label.shape == (64, 64, 32, 10)
        
        # Test with odd number of volumes (should raise error)
        odd_data = sample_asl_data[:, :, :, :-1]  # Remove one volume
        with pytest.raises(ASLDataError, match="even number of volumes"):
            processor.separate_control_label_pairs(odd_data)
    
    def test_calculate_perfusion_weighted_images(self, sample_asl_data):
        """Test perfusion-weighted image calculation."""
        processor = ASLProcessor()
        
        control, label = processor.separate_control_label_pairs(sample_asl_data)
        perfusion_images = processor.calculate_perfusion_weighted_images(control, label)
        
        assert perfusion_images.shape == (64, 64, 32, 10)
        
        # Test with mismatched shapes
        with pytest.raises(ASLDataError, match="same shape"):
            processor.calculate_perfusion_weighted_images(
                control[:, :, :, :5], label
            )
    
    def test_generate_brain_mask(self, sample_asl_data):
        """Test brain mask generation."""
        processor = ASLProcessor()
        
        mask = processor._generate_brain_mask(sample_asl_data)
        
        assert mask.shape == (64, 64, 32)
        assert mask.dtype == bool
        assert np.sum(mask) > 0  # Should have some brain voxels
    
    def test_estimate_motion_parameters(self, sample_asl_data):
        """Test motion parameter estimation."""
        processor = ASLProcessor()
        
        motion_dict = processor.estimate_motion_parameters(sample_asl_data)
        
        assert 'motion_estimates' in motion_dict
        assert 'mean_motion' in motion_dict
        assert 'max_motion' in motion_dict
        assert 'motion_outliers' in motion_dict
        assert 'motion_quality' in motion_dict
        
        assert len(motion_dict['motion_estimates']) == 19  # n_volumes - 1


class TestCBFCalculator:
    """Test CBF calculation functionality."""
    
    def test_calculator_initialization(self):
        """Test CBF calculator initialization."""
        calculator = CBFCalculator(
            labeling_efficiency=0.85,
            blood_t1=1.65,
            tissue_t1=1.3,
            partition_coefficient=0.9,
            post_labeling_delay=1.8,
            labeling_duration=1.8
        )
        
        assert calculator.alpha == 0.85
        assert calculator.t1_blood == 1.65
        assert calculator.t1_tissue == 1.3
        assert calculator.lambda_val == 0.9
        assert calculator.pld == 1.8
        assert calculator.tau == 1.8
    
    def test_calculate_cbf_simple(self, sample_asl_data, sample_m0_data, sample_mask_data):
        """Test simple CBF calculation."""
        calculator = CBFCalculator()
        processor = ASLProcessor()
        
        # Create perfusion-weighted images
        control, label = processor.separate_control_label_pairs(sample_asl_data)
        perfusion_images = processor.calculate_perfusion_weighted_images(control, label)
        
        # Calculate CBF
        cbf_map = calculator.calculate_cbf_simple(
            perfusion_images, sample_m0_data, sample_mask_data
        )
        
        assert cbf_map.shape == perfusion_images.shape
        
        # Check for reasonable CBF values (should be clipped to physiological range)
        brain_cbf = cbf_map[sample_mask_data]
        assert np.all(brain_cbf >= -200)
        assert np.all(brain_cbf <= 300)
    
    def test_calculate_cbf_statistics(self, sample_mask_data):
        """Test CBF statistics calculation."""
        calculator = CBFCalculator()
        
        # Create mock CBF map
        cbf_map = np.random.randn(64, 64, 32) * 20 + 50  # Mean ~50 ml/100g/min
        cbf_map[~sample_mask_data] = 0  # Zero outside mask
        
        stats = calculator.calculate_cbf_statistics(cbf_map, sample_mask_data)
        
        assert 'mean_cbf' in stats
        assert 'median_cbf' in stats
        assert 'std_cbf' in stats
        assert 'n_voxels' in stats
        assert 'normal_range_percentage' in stats
        assert 'abnormal_high_cbf' in stats
        assert 'abnormal_low_cbf' in stats
        
        assert stats['n_voxels'] > 0
        assert 0 <= stats['normal_range_percentage'] <= 100
    
    def test_calculate_regional_cbf_whole_brain(self, sample_mask_data):
        """Test regional CBF calculation without atlas."""
        calculator = CBFCalculator()
        
        cbf_map = np.random.randn(64, 64, 32) * 20 + 50
        cbf_map[~sample_mask_data] = 0
        
        regional_stats = calculator.calculate_regional_cbf(cbf_map)
        
        assert 'whole_brain' in regional_stats
        wb_stats = regional_stats['whole_brain']
        assert 'mean_cbf' in wb_stats
        assert 'std_cbf' in wb_stats
        assert 'voxel_count' in wb_stats
    
    def test_calculate_regional_cbf_with_atlas(self, sample_mask_data):
        """Test regional CBF calculation with atlas."""
        calculator = CBFCalculator()
        
        cbf_map = np.random.randn(64, 64, 32) * 20 + 50
        cbf_map[~sample_mask_data] = 0
        
        # Create simple atlas with 3 regions
        atlas = np.zeros_like(sample_mask_data, dtype=int)
        atlas[sample_mask_data] = 1  # Most brain voxels in region 1
        atlas[16:48, 16:48, 8:24] = 2  # Central region
        atlas[20:44, 20:44, 10:22] = 3  # Inner region
        
        region_names = {1: "Region1", 2: "Region2", 3: "Region3"}
        
        regional_stats = calculator.calculate_regional_cbf(cbf_map, atlas, region_names)
        
        assert len(regional_stats) == 3
        assert "Region1" in regional_stats
        assert "Region2" in regional_stats
        assert "Region3" in regional_stats
        
        for region_name, stats in regional_stats.items():
            assert 'mean_cbf' in stats
            assert 'region_id' in stats
            assert 'voxel_count' in stats


class TestQualityAssessor:
    """Test quality assessment functionality."""
    
    def test_assessor_initialization(self):
        """Test quality assessor initialization."""
        assessor = ASLQualityAssessor(
            snr_threshold=3.0,
            temporal_snr_threshold=5.0,
            motion_threshold=2.0,
            outlier_threshold=2.5
        )
        
        assert assessor.snr_threshold == 3.0
        assert assessor.temporal_snr_threshold == 5.0
        assert assessor.motion_threshold == 2.0
        assert assessor.outlier_threshold == 2.5
    
    def test_assess_data_quality(self, sample_asl_data, sample_mask_data):
        """Test data quality assessment."""
        assessor = ASLQualityAssessor()
        
        processed_data = {
            'asl_data': sample_asl_data,
            'mask_data': sample_mask_data,
            'n_volumes': 20,
            'n_pairs': 10,
            'voxel_size': (3.0, 3.0, 3.0)
        }
        
        metrics = assessor._assess_data_quality(processed_data)
        
        assert 'data_quality' in metrics
        data_quality = metrics['data_quality']
        
        assert 'data_shape' in data_quality
        assert 'n_volumes' in data_quality
        assert 'brain_mask_coverage' in data_quality
        assert 'data_quality_flags' in data_quality
        
        assert data_quality['data_shape'] == list(sample_asl_data.shape)
        assert data_quality['n_volumes'] == 20
        assert 0 <= data_quality['brain_mask_coverage'] <= 1
    
    def test_assess_signal_quality(self, sample_asl_data, sample_mask_data):
        """Test signal quality assessment."""
        assessor = ASLQualityAssessor()
        processor = ASLProcessor()
        
        # Create processed data
        control, label = processor.separate_control_label_pairs(sample_asl_data)
        perfusion_images = processor.calculate_perfusion_weighted_images(control, label)
        mean_perfusion = np.mean(perfusion_images, axis=3)
        
        processed_data = {
            'asl_data': sample_asl_data,
            'control_volumes': control,
            'label_volumes': label,
            'perfusion_images': perfusion_images,
            'mean_perfusion': mean_perfusion,
            'mask_data': sample_mask_data
        }
        
        metrics = assessor._assess_signal_quality(processed_data)
        
        assert 'signal_quality' in metrics
        signal_quality = metrics['signal_quality']
        
        assert 'temporal_snr' in signal_quality
        assert 'perfusion_snr' in signal_quality
        assert 'snr_quality_flags' in signal_quality
    
    def test_grade_quality(self):
        """Test quality grading function."""
        assessor = ASLQualityAssessor()
        
        assert assessor._grade_quality(95) == 'A'
        assert assessor._grade_quality(85) == 'B' 
        assert assessor._grade_quality(75) == 'C'
        assert assessor._grade_quality(65) == 'D'
        assert assessor._grade_quality(55) == 'F'
    
    def test_overall_quality_score_calculation(self):
        """Test overall quality score calculation."""
        assessor = ASLQualityAssessor()
        
        # Mock quality metrics
        quality_metrics = {
            'signal_quality': {
                'snr_quality_flags': {'overall_snr_adequate': True}
            },
            'motion_quality': {
                'motion_quality_score': 80.0
            },
            'temporal_stability': {
                'temporal_stability_score': 85.0
            },
            'spatial_coherence': {
                'spatial_coherence_score': 75.0
            }
        }
        
        overall_score = assessor._calculate_overall_quality_score(quality_metrics)
        
        assert 0 <= overall_score <= 100
        assert isinstance(overall_score, float)


class TestModels:
    """Test Pydantic models."""
    
    def test_asl_processing_parameters_validation(self):
        """Test ASL processing parameters validation."""
        # Valid parameters
        params = ASLProcessingParameters(
            post_labeling_delay=1.8,
            labeling_duration=1.8,
            labeling_efficiency=0.85
        )
        assert params.post_labeling_delay == 1.8
        assert params.labeling_duration == 1.8
        assert params.labeling_efficiency == 0.85
        
        # Invalid parameters should raise validation error
        with pytest.raises(Exception):  # Pydantic validation error
            ASLProcessingParameters(post_labeling_delay=5.0)  # Too high
            
        with pytest.raises(Exception):
            ASLProcessingParameters(labeling_efficiency=1.5)  # Too high
    
    def test_processing_request_validation(self, temp_nifti_file):
        """Test processing request validation."""
        temp_path, _ = temp_nifti_file
        
        # Valid request
        request = ProcessingRequest(asl_file_path=str(temp_path))
        assert request.asl_file_path == str(temp_path)
        
        # Invalid request - non-existent file
        with pytest.raises(Exception):  # Validation error
            ProcessingRequest(asl_file_path="/nonexistent/file.nii.gz")


class TestIntegration:
    """Integration tests for complete processing pipeline."""
    
    @patch('nibabel.load')
    def test_complete_processing_pipeline(self, mock_load, sample_asl_data, sample_m0_data, sample_mask_data):
        """Test complete ASL processing pipeline."""
        # Mock nibabel loading
        mock_img = Mock()
        mock_img.get_fdata.return_value = sample_asl_data
        mock_img.header.get_zooms.return_value = (3.0, 3.0, 3.0, 2.0)
        mock_load.return_value = mock_img
        
        # Initialize components
        processor = ASLProcessor()
        calculator = CBFCalculator()
        assessor = ASLQualityAssessor()
        
        # Step 1: Load and preprocess data
        with tempfile.NamedTemporaryFile(suffix='.nii.gz') as f:
            asl_data = processor.load_asl_data(f.name)
            asl_data['m0_data'] = sample_m0_data
            asl_data['mask_data'] = sample_mask_data
            
            processed_data = processor.preprocess_asl_data(asl_data)
            
            # Step 2: Calculate CBF
            cbf_map = calculator.calculate_cbf_simple(
                processed_data['perfusion_images'],
                processed_data['m0_image'], 
                processed_data['mask_data']
            )
            
            cbf_results = {
                'cbf': cbf_map,
                'model_type': 'simple'
            }
            
            # Step 3: Quality assessment
            quality_results = assessor.assess_asl_quality(processed_data, cbf_results)
            
            # Verify results
            assert 'preprocessed' in processed_data
            assert processed_data['preprocessed'] is True
            
            assert cbf_map.shape == processed_data['perfusion_images'].shape
            
            assert 'overall_quality_score' in quality_results
            assert 'overall_quality_grade' in quality_results
            assert 'quality_summary' in quality_results
            
            # Check that quality score is reasonable
            assert 0 <= quality_results['overall_quality_score'] <= 100


if __name__ == "__main__":
    pytest.main([__file__])
