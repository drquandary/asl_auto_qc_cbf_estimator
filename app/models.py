"""
Pydantic Models for ASL Auto-QC CBF Estimator API.

This module defines the data models for API requests and responses,
including ASL processing parameters, quality metrics, and CBF results.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
try:
    from pydantic import BaseSettings
except ImportError:
    from pydantic_settings import BaseSettings


class ASLSequenceType(str, Enum):
    """ASL sequence types."""
    PCASL = "pCASL"
    CASL = "CASL"
    PASL = "PASL"
    UNKNOWN = "unknown"


class QualityGrade(str, Enum):
    """Quality assessment grades."""
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    F = "F"


class DataUsability(str, Enum):
    """Data usability categories."""
    EXCELLENT = "excellent"
    ACCEPTABLE = "acceptable"
    MARGINAL = "marginal"
    POOR = "poor"
    UNKNOWN = "unknown"


class ProcessingStatus(str, Enum):
    """Processing status values."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ASLProcessingParameters(BaseModel):
    """Parameters for ASL processing."""
    
    # Sequence parameters
    sequence_type: ASLSequenceType = ASLSequenceType.PCASL
    post_labeling_delay: float = Field(1.8, ge=0.5, le=4.0, description="Post-labeling delay in seconds")
    labeling_duration: float = Field(1.8, ge=0.5, le=3.0, description="Labeling duration in seconds")
    
    # Quantification parameters
    labeling_efficiency: float = Field(0.85, ge=0.5, le=1.0, description="Labeling efficiency (alpha)")
    blood_t1: float = Field(1.65, ge=1.0, le=2.5, description="T1 of arterial blood in seconds")
    tissue_t1: float = Field(1.3, ge=0.8, le=2.0, description="T1 of gray matter in seconds")
    partition_coefficient: float = Field(0.9, ge=0.5, le=1.2, description="Brain-blood partition coefficient")
    
    # Processing options
    temporal_filter: bool = Field(True, description="Apply temporal filtering")
    spatial_smoothing: bool = Field(True, description="Apply spatial smoothing")
    motion_correction: bool = Field(False, description="Apply motion correction")
    partial_volume_correction: bool = Field(False, description="Apply partial volume correction")
    
    # Quality control thresholds
    snr_threshold: float = Field(3.0, ge=1.0, le=10.0, description="Minimum SNR threshold")
    motion_threshold: float = Field(2.0, ge=0.5, le=5.0, description="Motion threshold in mm")
    
    class Config:
        use_enum_values = True


class FileUploadRequest(BaseModel):
    """Request for file upload processing."""
    
    processing_parameters: Optional[ASLProcessingParameters] = None
    calculate_cbf: bool = Field(True, description="Whether to calculate CBF maps")
    perform_qc: bool = Field(True, description="Whether to perform quality control")
    generate_report: bool = Field(True, description="Whether to generate QC report")
    
    class Config:
        use_enum_values = True


class DataQualityMetrics(BaseModel):
    """Data quality metrics."""
    
    data_shape: List[int]
    n_volumes: int
    n_pairs: int
    voxel_size: List[float]
    nan_voxels: int
    inf_voxels: int
    zero_volumes: int
    brain_mask_coverage: float
    signal_range: Dict[str, float]
    data_quality_flags: Dict[str, bool]


class SignalQualityMetrics(BaseModel):
    """Signal quality metrics."""
    
    temporal_snr: Optional[Dict[str, float]] = None
    perfusion_snr: Optional[Dict[str, float]] = None
    signal_stability: Dict[str, float]
    snr_quality_flags: Dict[str, bool]


class MotionQualityMetrics(BaseModel):
    """Motion quality metrics."""
    
    motion_estimates: List[float]
    mean_motion: float
    max_motion: float
    std_motion: float
    motion_outlier_volumes: List[int]
    n_motion_outliers: int
    control_motion: Dict[str, float]
    label_motion: Dict[str, float]
    motion_quality_score: float
    motion_quality_grade: QualityGrade


class TemporalStabilityMetrics(BaseModel):
    """Temporal stability metrics."""
    
    mean_signal_timeseries: List[float]
    temporal_mean: float
    temporal_std: float
    temporal_cov: float
    temporal_drift: Dict[str, Union[float, bool]]
    temporal_outliers: List[int]
    n_temporal_outliers: int
    temporal_stability_score: float
    temporal_stability_grade: QualityGrade


class SpatialCoherenceMetrics(BaseModel):
    """Spatial coherence metrics."""
    
    spatial_roughness: float
    gm_wm_contrast: Optional[Dict[str, Union[float, bool]]] = None
    spatial_clustering: Optional[Dict[str, Union[int, float, bool]]] = None
    spatial_coherence_score: float
    spatial_coherence_grade: QualityGrade


class CBFQualityMetrics(BaseModel):
    """CBF-specific quality metrics."""
    
    cbf_distribution: Optional[Dict[str, float]] = None
    physiological_assessment: Optional[Dict[str, float]] = None
    cbf_quality_score: Optional[float] = None
    cbf_quality_grade: Optional[QualityGrade] = None
    no_cbf_data: Optional[bool] = None


class QualitySummary(BaseModel):
    """Quality assessment summary."""
    
    overall_grade: QualityGrade
    major_issues: List[str]
    recommendations: List[str]
    data_usability: DataUsability


class QualityAssessmentResult(BaseModel):
    """Complete quality assessment results."""
    
    data_quality: DataQualityMetrics
    signal_quality: SignalQualityMetrics
    motion_quality: MotionQualityMetrics
    temporal_stability: TemporalStabilityMetrics
    spatial_coherence: SpatialCoherenceMetrics
    cbf_quality: Optional[CBFQualityMetrics] = None
    overall_quality_score: float = Field(ge=0, le=100)
    overall_quality_grade: QualityGrade
    quality_summary: QualitySummary


class CBFStatistics(BaseModel):
    """CBF statistical measures."""
    
    mean_cbf: float
    median_cbf: float
    std_cbf: float
    min_cbf: float
    max_cbf: float
    percentile_05: float
    percentile_25: float
    percentile_75: float
    percentile_95: float
    n_voxels: int
    coefficient_of_variation: float
    skewness: float
    kurtosis: float
    abnormal_high_cbf: int
    abnormal_low_cbf: int
    normal_range_percentage: float


class RegionalCBF(BaseModel):
    """Regional CBF measurements."""
    
    region_id: Optional[int] = None
    mean_cbf: float
    std_cbf: float
    median_cbf: float
    percentile_95: float
    percentile_05: float
    voxel_count: int


class CBFResults(BaseModel):
    """CBF calculation results."""
    
    model_type: str
    global_statistics: CBFStatistics
    regional_statistics: Optional[Dict[str, RegionalCBF]] = None
    arterial_transit_time_available: bool = False
    partial_volume_corrected: bool = False
    
    class Config:
        use_enum_values = True


class ProcessingProgress(BaseModel):
    """Processing progress information."""
    
    stage: str
    progress_percent: float = Field(ge=0, le=100)
    estimated_time_remaining: Optional[int] = None  # seconds
    current_step: str
    total_steps: int
    completed_steps: int


class ProcessingResult(BaseModel):
    """Complete processing results."""
    
    # Processing metadata
    processing_id: str
    status: ProcessingStatus
    timestamp: datetime
    processing_time_seconds: Optional[float] = None
    
    # Input information
    input_files: Dict[str, str]
    processing_parameters: ASLProcessingParameters
    
    # Results
    quality_assessment: Optional[QualityAssessmentResult] = None
    cbf_results: Optional[CBFResults] = None
    
    # Processing information
    progress: Optional[ProcessingProgress] = None
    error_message: Optional[str] = None
    warnings: List[str] = []
    
    # Output files (paths or URLs)
    output_files: Optional[Dict[str, str]] = None
    
    class Config:
        use_enum_values = True


class HealthCheckResponse(BaseModel):
    """Health check response."""
    
    status: str
    timestamp: datetime
    version: str
    dependencies: Dict[str, str]


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str
    message: str
    details: Optional[Dict] = None
    timestamp: datetime


class ProcessingRequest(BaseModel):
    """Request for ASL processing."""
    
    # Required parameters
    asl_file_path: str = Field(..., description="Path to ASL NIfTI file")
    
    # Optional parameters
    m0_file_path: Optional[str] = Field(None, description="Path to M0 calibration file")
    mask_file_path: Optional[str] = Field(None, description="Path to brain mask file")
    atlas_file_path: Optional[str] = Field(None, description="Path to anatomical atlas file")
    
    # Processing parameters
    processing_parameters: Optional[ASLProcessingParameters] = None
    
    # Output options
    save_intermediate_results: bool = Field(False, description="Save intermediate processing results")
    output_directory: Optional[str] = Field(None, description="Output directory for results")
    
    @validator('asl_file_path', 'm0_file_path', 'mask_file_path', 'atlas_file_path')
    def validate_file_paths(cls, v):
        if v is not None and not Path(v).exists():
            raise ValueError(f"File does not exist: {v}")
        return v


class BatchProcessingRequest(BaseModel):
    """Request for batch processing of multiple ASL datasets."""
    
    processing_requests: List[ProcessingRequest]
    parallel_processing: bool = Field(True, description="Process datasets in parallel")
    max_concurrent_jobs: int = Field(4, ge=1, le=10, description="Maximum concurrent processing jobs")
    
    
class BatchProcessingResult(BaseModel):
    """Results from batch processing."""
    
    total_datasets: int
    completed_successfully: int
    failed_processing: int
    processing_time_seconds: float
    results: List[ProcessingResult]
    summary_statistics: Optional[Dict[str, float]] = None


class VisualizationRequest(BaseModel):
    """Request for generating visualization plots."""
    
    processing_result_id: str
    plot_types: List[str] = Field(["cbf_map", "quality_metrics", "temporal_plot"])
    save_plots: bool = Field(True, description="Save plots to files")
    plot_format: str = Field("png", regex="^(png|jpg|svg|pdf)$")


class ConfigurationUpdate(BaseModel):
    """Update configuration parameters."""
    
    processing_parameters: Optional[ASLProcessingParameters] = None
    quality_thresholds: Optional[Dict[str, float]] = None
    output_settings: Optional[Dict[str, Union[str, bool]]] = None