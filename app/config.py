"""
Configuration for ASL Auto-QC CBF Estimator.

This module contains configuration settings for ASL processing,
quality control thresholds, and application parameters.
"""

import os
from pathlib import Path
from functools import lru_cache
from typing import Optional

from pydantic import Field
try:
    from pydantic import BaseSettings
except ImportError:
    from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with validation."""
    
    # Project paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "data")
    app_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "app")
    upload_directory: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "uploads")
    output_directory: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "outputs")
    
    # Application settings
    app_name: str = "ASL Auto-QC CBF Estimator"
    version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    host: str = Field(default="localhost", env="HOST")
    port: int = Field(default=8000, env="PORT")
    
    # ASL Processing Parameters
    # Sequence parameters
    sequence_type: str = Field(default="pCASL", description="Default ASL sequence type")
    post_labeling_delay: float = Field(default=1.8, ge=0.5, le=4.0, description="Post-labeling delay in seconds")
    labeling_duration: float = Field(default=1.8, ge=0.5, le=3.0, description="Labeling duration in seconds")
    
    # Quantification parameters
    labeling_efficiency: float = Field(default=0.85, ge=0.5, le=1.0, description="Labeling efficiency (alpha)")
    blood_t1: float = Field(default=1.65, ge=1.0, le=2.5, description="T1 of arterial blood at 3T in seconds")
    tissue_t1: float = Field(default=1.3, ge=0.8, le=2.0, description="T1 of gray matter at 3T in seconds") 
    partition_coefficient: float = Field(default=0.9, ge=0.5, le=1.2, description="Brain-blood partition coefficient")
    
    # Processing options
    temporal_filter: bool = Field(default=True, description="Apply temporal filtering by default")
    spatial_smoothing: bool = Field(default=True, description="Apply spatial smoothing by default")
    motion_correction: bool = Field(default=False, description="Apply motion correction by default")
    partial_volume_correction: bool = Field(default=False, description="Apply partial volume correction by default")
    
    # Quality Control Thresholds
    snr_threshold: float = Field(default=3.0, ge=1.0, le=10.0, description="Minimum SNR threshold")
    temporal_snr_threshold: float = Field(default=5.0, ge=2.0, le=20.0, description="Minimum temporal SNR threshold")
    motion_threshold: float = Field(default=2.0, ge=0.5, le=5.0, description="Motion threshold in mm")
    outlier_threshold: float = Field(default=2.5, ge=1.0, le=5.0, description="Z-score threshold for outlier detection")
    
    # CBF Quality Thresholds
    cbf_min_physiological: float = Field(default=10.0, description="Minimum physiological CBF value")
    cbf_max_physiological: float = Field(default=150.0, description="Maximum physiological CBF value")
    cbf_normal_min: float = Field(default=20.0, description="Normal CBF range minimum")
    cbf_normal_max: float = Field(default=100.0, description="Normal CBF range maximum")
    
    # Processing limits
    max_concurrent_jobs: int = Field(default=4, ge=1, le=10, description="Maximum concurrent processing jobs")
    processing_timeout: int = Field(default=3600, ge=300, le=7200, description="Processing timeout in seconds")
    max_file_size_mb: int = Field(default=500, ge=10, le=2000, description="Maximum file size in MB")
    
    # Visualization settings
    default_colormap: str = Field(default="viridis", description="Default colormap for CBF visualizations")
    figure_dpi: int = Field(default=150, ge=72, le=300, description="Figure DPI for saved plots")
    save_format: str = Field(default="png", description="Default format for saved figures")
    
    # Regional analysis settings
    use_default_atlas: bool = Field(default=True, description="Use default brain atlas if none provided")
    atlas_threshold: float = Field(default=0.5, description="Threshold for atlas regions")
    
    # Logging settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    
    # Database settings (for future use)
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    
    # Security settings
    api_key: Optional[str] = Field(default=None, env="API_KEY")
    cors_origins: list = Field(default=["*"], env="CORS_ORIGINS")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.create_directories()
    
    def create_directories(self):
        """Create necessary directories."""
        directories = [
            self.data_dir,
            self.upload_directory,
            self.output_directory,
        ]
        
        for directory in directories:
            directory.mkdir(exist_ok=True, parents=True)


# Legacy configuration for backward compatibility
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
APP_DIR = PROJECT_ROOT / "app"

# Application settings (legacy)
APP_NAME = "ASL Auto-QC CBF Estimator"
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
HOST = os.getenv("HOST", "localhost")
PORT = int(os.getenv("PORT", "8000"))

# Data settings (legacy)
SAMPLE_DATA_FILE = DATA_DIR / "sample_asl_data.json"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# ASL-specific configuration constants
ASL_CONFIG = {
    # Standard ASL sequence parameters
    "SEQUENCES": {
        "pCASL": {
            "labeling_efficiency": 0.85,
            "recommended_pld": [1.8, 2.0],
            "recommended_tau": 1.8
        },
        "CASL": {
            "labeling_efficiency": 0.95,
            "recommended_pld": [1.5, 2.0],
            "recommended_tau": 1.5
        },
        "PASL": {
            "labeling_efficiency": 0.95,
            "recommended_pld": [1.2, 1.8],
            "recommended_tau": 0.7
        }
    },
    
    # Physiological constants
    "PHYSIOLOGY": {
        "blood_t1_3t": 1.65,  # seconds
        "blood_t1_1_5t": 1.35,  # seconds
        "gm_t1_3t": 1.3,  # seconds
        "wm_t1_3t": 0.83,  # seconds
        "partition_coefficient": 0.9,  # ml/g
        "blood_density": 1.04,  # g/ml
    },
    
    # Quality control thresholds
    "QC_THRESHOLDS": {
        "snr_minimum": 3.0,
        "temporal_snr_minimum": 5.0,
        "motion_maximum_mm": 2.0,
        "outlier_z_score": 2.5,
        "cbf_physiological_min": 5.0,  # ml/100g/min
        "cbf_physiological_max": 200.0,  # ml/100g/min
        "cbf_normal_range": [20.0, 100.0],  # ml/100g/min
        "perfusion_signal_threshold": 0.1  # % signal change
    },
    
    # Processing parameters
    "PROCESSING": {
        "spatial_smoothing_fwhm": 4.0,  # mm
        "temporal_filter_kernel": 3,  # volumes
        "brain_mask_threshold": 0.3,
        "erosion_iterations": 1,
        "dilation_iterations": 2
    },
    
    # Regional analysis
    "REGIONS": {
        "default_atlas": "AAL",
        "gm_regions": [1, 2, 3, 4, 5, 6, 7, 8],  # Example region IDs
        "wm_regions": [41, 42, 43, 44, 45, 46, 47, 48]  # Example region IDs
    }
}


def get_sequence_defaults(sequence_type: str) -> dict:
    """Get default parameters for ASL sequence type."""
    return ASL_CONFIG["SEQUENCES"].get(sequence_type.upper(), ASL_CONFIG["SEQUENCES"]["pCASL"])


def get_physiological_constants(field_strength: str = "3T") -> dict:
    """Get physiological constants for field strength."""
    constants = ASL_CONFIG["PHYSIOLOGY"].copy()
    
    if field_strength == "1.5T":
        constants["blood_t1"] = constants["blood_t1_1_5t"]
    else:
        constants["blood_t1"] = constants["blood_t1_3t"]
    
    return constants
