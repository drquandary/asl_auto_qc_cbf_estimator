# ASL Auto-QC and CBF Estimator

## Overview

This service provides comprehensive automated quality control and cerebral blood flow (CBF) quantification for Arterial Spin Labeling (ASL) MRI data, specifically designed for the Detre Lab's perfusion studies. The application implements BASIL-inspired algorithms and follows established ASL processing standards.

## Features

### Core Functionality
- **ASL Data Processing**: Complete preprocessing pipeline for pCASL, CASL, and PASL sequences
- **CBF Quantification**: BASIL-style algorithms with kinetic modeling support
- **Comprehensive Quality Control**: Multi-dimensional QC metrics including SNR, motion, temporal stability
- **Regional Analysis**: Atlas-based perfusion analysis with customizable regions
- **Real-time Processing**: Background processing with progress tracking
- **Export Capabilities**: NIfTI output for CBF maps and JSON reports for QC metrics

### Quality Control Features
- Signal-to-noise ratio assessment
- Motion parameter evaluation and outlier detection
- Temporal stability analysis and drift detection
- Spatial coherence assessment
- Perfusion territory analysis
- Automated quality grading (A-F scale)

### Input/Output Support
- **Input**: ASL MRI data (NIfTI), M0 calibration images, brain masks, anatomical atlases
- **Output**: CBF maps, quality reports, regional statistics, visualization plots

## Quick Start

### Installation

```bash
# Clone and navigate to the project
cd asl_auto_qc_cbf_estimator

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -U pip
pip install -r requirements.txt
```

### Running the Application

```bash
# Start the FastAPI server
uvicorn app.main:app --reload

# The API will be available at http://localhost:8000
# Interactive documentation at http://localhost:8000/docs
```

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/api/health
```

### ASL Processing
```bash
curl -X POST http://localhost:8000/api/process \
  -H "Content-Type: application/json" \
  -d '{
    "asl_file_path": "/path/to/asl_data.nii.gz",
    "m0_file_path": "/path/to/m0_calibration.nii.gz",
    "processing_parameters": {
      "sequence_type": "pCASL",
      "post_labeling_delay": 1.8,
      "labeling_duration": 1.8,
      "temporal_filter": true,
      "spatial_smoothing": true
    }
  }'
```

### File Upload
```bash
curl -X POST http://localhost:8000/api/upload_files \
  -F "asl_file=@asl_data.nii.gz" \
  -F "m0_file=@m0_calibration.nii.gz"
```

### Get Results
```bash
curl http://localhost:8000/api/results/{processing_id}
```

## API Quick Reference

- GET `/api/health`: Service health.
- POST `/api/upload_files`: Upload ASL and M0 files.
- POST `/api/process`: Start ASL QC/CBF processing with parameters.
- GET `/api/results/{processing_id}`: Retrieve processing results.

## Configuration

### ASL Processing Parameters

The application supports comprehensive ASL processing configuration:

```python
from app.models import ASLProcessingParameters

params = ASLProcessingParameters(
    sequence_type="pCASL",           # pCASL, CASL, or PASL
    post_labeling_delay=1.8,         # seconds
    labeling_duration=1.8,           # seconds
    labeling_efficiency=0.85,        # alpha parameter
    blood_t1=1.65,                   # T1 of blood at 3T (seconds)
    tissue_t1=1.3,                   # T1 of GM at 3T (seconds)
    partition_coefficient=0.9,       # brain-blood partition coefficient
    temporal_filter=True,            # apply temporal filtering
    spatial_smoothing=True,          # apply spatial smoothing
    snr_threshold=3.0,              # minimum SNR threshold
    motion_threshold=2.0            # motion threshold (mm)
)
```

### Environment Variables

Create a `.env` file for custom configuration:

```bash
# Application settings
DEBUG=false
HOST=localhost
PORT=8000
LOG_LEVEL=INFO

# Processing settings
MOTION_THRESHOLD=2.0
SNR_THRESHOLD=3.0
MAX_CONCURRENT_JOBS=4

# File paths
UPLOAD_DIRECTORY=./uploads
OUTPUT_DIRECTORY=./outputs
```

## ASL Processing Pipeline

### 1. Data Loading and Validation
- Load ASL 4D NIfTI data
- Validate dimensions and acquisition parameters
- Load optional M0 calibration and brain mask
- Generate brain mask if not provided

### 2. Preprocessing
- Separate control and label volumes
- Calculate perfusion-weighted images (control - label)
- Apply temporal filtering and spatial smoothing
- Motion parameter estimation

### 3. CBF Quantification
Using the standard ASL quantification equation:

```
CBF = (6000 × λ × ΔM × exp(PLD/T1_blood)) / (2 × α × T1_blood × (1 - exp(-τ/T1_blood)) × M0)
```

Where:
- λ = partition coefficient (0.9 ml/g)
- ΔM = perfusion-weighted signal
- PLD = post-labeling delay
- α = labeling efficiency
- τ = labeling duration
- M0 = equilibrium magnetization

### 4. Quality Assessment
- **Data Quality**: Check for NaN/Inf values, brain mask coverage
- **Signal Quality**: Calculate SNR metrics, assess signal stability
- **Motion Quality**: Estimate motion parameters, detect outliers  
- **Temporal Stability**: Assess signal drift and temporal outliers
- **Spatial Coherence**: Analyze spatial patterns and GM/WM contrast

### 5. Regional Analysis
- Apply anatomical atlas for region-based statistics
- Calculate regional CBF means, standard deviations, and percentiles
- Generate regional quality metrics

## Quality Control Metrics

### Overall Quality Grading
- **Grade A (90-100)**: Excellent data quality, suitable for all analyses
- **Grade B (80-89)**: Good quality, suitable for most analyses
- **Grade C (70-79)**: Acceptable quality, caution for sensitive analyses
- **Grade D (60-69)**: Marginal quality, limited usability
- **Grade F (<60)**: Poor quality, not recommended for analysis

### Specific QC Metrics
1. **Signal-to-Noise Ratio**: Temporal and perfusion-weighted SNR
2. **Motion Assessment**: Frame-to-frame motion estimates
3. **Temporal Stability**: Signal drift and coefficient of variation
4. **Spatial Coherence**: Gray matter/white matter contrast
5. **CBF Physiological Range**: Percentage of voxels in normal range (20-100 ml/100g/min)

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=app --cov-report=html

# Run specific test class
pytest tests/test_main.py::TestASLProcessor -v

# Run integration tests
pytest tests/test_main.py::TestIntegration -v
```

### Test Data
The application includes synthetic test data and fixtures for:
- 4D ASL time series (64×64×32×20 volumes)
- M0 calibration images
- Brain masks and anatomical atlases
- Mock processing scenarios

## Development

### Code Structure
```
app/
├── main.py              # FastAPI application
├── routes.py            # API endpoints
├── models.py            # Pydantic data models
├── config.py            # Configuration settings
├── asl_processor.py     # ASL data processing
├── cbf_calculator.py    # CBF quantification algorithms
├── quality_assessor.py  # Quality control metrics
└── common_utils/        # Shared utilities

data/
├── sample_asl_data.json # Sample dataset metadata
└── uploads/             # Uploaded files

tests/
└── test_main.py         # Comprehensive test suite
```

### Contributing

1. **Code Style**: Follow PEP 8, use black formatter
2. **Testing**: Maintain >80% test coverage
3. **Documentation**: Document all public APIs
4. **Error Handling**: Use custom exception classes
5. **Logging**: Use structured logging throughout

```bash
# Development workflow
black app tests
isort app tests  
flake8 app tests
mypy app
pytest tests/ --cov=app
```

### Adding New Features

1. **New Processing Algorithms**: Extend `CBFCalculator` class
2. **Additional QC Metrics**: Add methods to `ASLQualityAssessor`
3. **New API Endpoints**: Add to `routes.py` with proper models
4. **Configuration Options**: Update `Settings` class in `config.py`

## Clinical Usage

### Recommended Workflow
1. **Data Preparation**: Ensure BIDS-compliant ASL data organization
2. **Quality Screening**: Run QC pipeline before analysis
3. **Parameter Optimization**: Adjust processing parameters based on sequence
4. **Batch Processing**: Use batch endpoint for multiple datasets
5. **Results Review**: Examine quality reports and CBF statistics

### Integration with Existing Tools
- **FSL**: Compatible with BASIL outputs
- **SPM**: Can import/export SPM-format images
- **BIDS**: Follows BIDS ASL specification
- **Research Pipelines**: RESTful API for integration

## Performance

### Processing Times
- Typical dataset (64×64×32×60): 2-3 minutes
- High-resolution (128×128×64×60): 5-8 minutes
- Quality assessment only: 30-60 seconds

### System Requirements
- RAM: 4-8 GB for typical datasets
- Storage: 1-2 GB per processed dataset
- CPU: Multi-core recommended for batch processing

## Troubleshooting

### Common Issues

1. **"ASL file not found"**: Check file paths and permissions
2. **"Invalid dimensions"**: Ensure 4D ASL data with even number of volumes
3. **"Low SNR warning"**: Check acquisition parameters and coil performance
4. **"Excessive motion"**: Consider motion correction or subject exclusion
5. **"Unrealistic CBF values"**: Verify sequence parameters and M0 calibration

### Debug Mode
```bash
# Enable debug logging
export DEBUG=true
export LOG_LEVEL=DEBUG

# Run with detailed output
uvicorn app.main:app --reload --log-level debug
```

### Support
For technical issues or questions about ASL processing:
1. Check the comprehensive test suite for usage examples
2. Review the sample data structure in `data/sample_asl_data.json`
3. Consult ASL processing literature and BASIL documentation
4. Contact the Detre Lab for domain-specific questions

## References

1. Alsop DC, et al. Recommended implementation of arterial spin-labeled perfusion MRI for clinical applications. Magn Reson Med. 2015.
2. Chappell MA, et al. Variational Bayesian inference for a nonlinear forward model. IEEE Trans Signal Process. 2009.
3. Groves AR, et al. Combined spatial and non-spatial prior for inference on MRI time-series. NeuroImage. 2009.

---

**Version**: 1.0.0  
**Developed for**: Detre Lab Perfusion Studies  
**License**: See LICENSE file
