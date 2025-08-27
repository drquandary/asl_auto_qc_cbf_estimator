"""
API Routes for ASL Auto-QC CBF Estimator.

This module defines the FastAPI routes for ASL data processing,
quality control, and CBF quantification.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.responses import FileResponse
import numpy as np

from .models import (
    ProcessingRequest,
    ProcessingResult,
    BatchProcessingRequest,
    BatchProcessingResult,
    HealthCheckResponse,
    ErrorResponse,
    VisualizationRequest,
    ConfigurationUpdate,
    ASLProcessingParameters,
    ProcessingStatus
)
from .asl_processor import ASLProcessor, ASLDataError
from .cbf_calculator import CBFCalculator, CBFCalculationError
from .quality_assessor import ASLQualityAssessor, QualityAssessmentError
from .config import get_settings

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Global storage for processing results (in production, use a proper database)
processing_results: Dict[str, ProcessingResult] = {}

# Processing dependencies
settings = get_settings()


def get_asl_processor() -> ASLProcessor:
    """Dependency for ASL processor."""
    return ASLProcessor(
        temporal_filter=True,
        spatial_smoothing=True,
        motion_threshold=settings.motion_threshold
    )


def get_cbf_calculator() -> CBFCalculator:
    """Dependency for CBF calculator."""
    return CBFCalculator(
        labeling_efficiency=settings.labeling_efficiency,
        blood_t1=settings.blood_t1,
        tissue_t1=settings.tissue_t1,
        partition_coefficient=settings.partition_coefficient,
        post_labeling_delay=settings.post_labeling_delay,
        labeling_duration=settings.labeling_duration
    )


def get_quality_assessor() -> ASLQualityAssessor:
    """Dependency for quality assessor."""
    return ASLQualityAssessor(
        snr_threshold=settings.snr_threshold,
        temporal_snr_threshold=settings.temporal_snr_threshold,
        motion_threshold=settings.motion_threshold,
        outlier_threshold=settings.outlier_threshold
    )


@router.get('/health', response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Test basic functionality
        processor = get_asl_processor()
        calculator = get_cbf_calculator()
        assessor = get_quality_assessor()
        
        dependencies = {
            "numpy": np.__version__,
            "nibabel": "5.0.0+",  # Approximate version
            "scipy": "1.9.0+",
            "scikit-image": "0.19.0+",
            "fastapi": "0.100.0+"
        }
        
        return HealthCheckResponse(
            status="ok",
            timestamp=datetime.now(),
            version="1.0.0",
            dependencies=dependencies
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")


@router.post('/process', response_model=ProcessingResult)
async def process_asl_data(
    request: ProcessingRequest,
    background_tasks: BackgroundTasks,
    asl_processor: ASLProcessor = Depends(get_asl_processor),
    cbf_calculator: CBFCalculator = Depends(get_cbf_calculator),
    quality_assessor: ASLQualityAssessor = Depends(get_quality_assessor)
):
    """
    Process ASL data with quality control and CBF quantification.
    
    Args:
        request: Processing request with file paths and parameters
        background_tasks: FastAPI background tasks
        asl_processor: ASL processor instance
        cbf_calculator: CBF calculator instance
        quality_assessor: Quality assessor instance
        
    Returns:
        ProcessingResult with quality metrics and CBF results
    """
    processing_id = str(uuid.uuid4())
    
    # Initialize result
    result = ProcessingResult(
        processing_id=processing_id,
        status=ProcessingStatus.PROCESSING,
        timestamp=datetime.now(),
        input_files={
            "asl_file": request.asl_file_path,
            "m0_file": request.m0_file_path or "",
            "mask_file": request.mask_file_path or "",
            "atlas_file": request.atlas_file_path or ""
        },
        processing_parameters=request.processing_parameters or ASLProcessingParameters()
    )
    
    # Store initial result
    processing_results[processing_id] = result
    
    # Start background processing
    background_tasks.add_task(
        _process_asl_background,
        processing_id,
        request,
        asl_processor,
        cbf_calculator,
        quality_assessor
    )
    
    return result


async def _process_asl_background(
    processing_id: str,
    request: ProcessingRequest,
    asl_processor: ASLProcessor,
    cbf_calculator: CBFCalculator,
    quality_assessor: ASLQualityAssessor
):
    """Background task for ASL processing."""
    start_time = datetime.now()
    
    try:
        result = processing_results[processing_id]
        
        # Step 1: Load and preprocess ASL data
        logger.info(f"Loading ASL data: {request.asl_file_path}")
        asl_data = asl_processor.load_asl_data(
            asl_path=request.asl_file_path,
            m0_path=request.m0_file_path,
            mask_path=request.mask_file_path
        )
        
        # Step 2: Preprocess data
        logger.info("Preprocessing ASL data")
        processed_data = asl_processor.preprocess_asl_data(asl_data)
        
        # Step 3: Calculate CBF
        cbf_results = None
        if request.processing_parameters is None or request.processing_parameters.calculate_cbf:
            logger.info("Calculating CBF")
            try:
                # Update calculator parameters if provided
                if request.processing_parameters:
                    params = request.processing_parameters
                    cbf_calculator.alpha = params.labeling_efficiency
                    cbf_calculator.t1_blood = params.blood_t1
                    cbf_calculator.pld = params.post_labeling_delay
                    cbf_calculator.tau = params.labeling_duration
                
                # Calculate CBF using simple model
                cbf_map = cbf_calculator.calculate_cbf_simple(
                    processed_data['perfusion_images'],
                    processed_data['m0_image'],
                    processed_data['mask_data']
                )
                
                # Calculate CBF statistics
                cbf_stats = cbf_calculator.calculate_cbf_statistics(
                    cbf_map,
                    processed_data['mask_data']
                )
                
                # Regional analysis if atlas provided
                regional_stats = {}
                if request.atlas_file_path:
                    try:
                        import nibabel as nib
                        atlas_img = nib.load(request.atlas_file_path)
                        atlas_data = atlas_img.get_fdata()
                        regional_stats = cbf_calculator.calculate_regional_cbf(
                            cbf_map, atlas_data
                        )
                    except Exception as e:
                        logger.warning(f"Regional analysis failed: {e}")
                
                cbf_results = {
                    'cbf': cbf_map,
                    'model_type': 'simple',
                    'global_statistics': cbf_stats,
                    'regional_statistics': regional_stats
                }
                
                logger.info("CBF calculation completed")
                
            except CBFCalculationError as e:
                logger.error(f"CBF calculation failed: {e}")
                result.warnings.append(f"CBF calculation failed: {e}")
        
        # Step 4: Quality assessment
        quality_results = None
        if request.processing_parameters is None or getattr(request.processing_parameters, 'perform_qc', True):
            logger.info("Performing quality assessment")
            try:
                quality_results = quality_assessor.assess_asl_quality(
                    processed_data,
                    cbf_results
                )
                logger.info("Quality assessment completed")
                
            except QualityAssessmentError as e:
                logger.error(f"Quality assessment failed: {e}")
                result.warnings.append(f"Quality assessment failed: {e}")
        
        # Update result with successful completion
        processing_time = (datetime.now() - start_time).total_seconds()
        result.status = ProcessingStatus.COMPLETED
        result.processing_time_seconds = processing_time
        result.quality_assessment = quality_results
        result.cbf_results = cbf_results
        
        # Save output files if requested
        if request.save_intermediate_results and request.output_directory:
            output_files = _save_processing_results(
                processing_id,
                processed_data,
                cbf_results,
                quality_results,
                Path(request.output_directory)
            )
            result.output_files = output_files
        
        processing_results[processing_id] = result
        logger.info(f"Processing completed successfully in {processing_time:.2f} seconds")
        
    except Exception as e:
        # Handle processing errors
        processing_time = (datetime.now() - start_time).total_seconds()
        error_msg = f"Processing failed: {str(e)}"
        logger.error(error_msg)
        
        result = processing_results[processing_id]
        result.status = ProcessingStatus.FAILED
        result.processing_time_seconds = processing_time
        result.error_message = error_msg
        processing_results[processing_id] = result


@router.get('/results/{processing_id}', response_model=ProcessingResult)
async def get_processing_results(processing_id: str):
    """Get processing results by ID."""
    if processing_id not in processing_results:
        raise HTTPException(status_code=404, detail="Processing ID not found")
    
    return processing_results[processing_id]


@router.get('/results', response_model=List[ProcessingResult])
async def list_all_results(limit: int = 50, offset: int = 0):
    """List all processing results."""
    all_results = list(processing_results.values())
    # Sort by timestamp, most recent first
    all_results.sort(key=lambda x: x.timestamp, reverse=True)
    return all_results[offset:offset + limit]


@router.delete('/results/{processing_id}')
async def delete_processing_results(processing_id: str):
    """Delete processing results by ID."""
    if processing_id not in processing_results:
        raise HTTPException(status_code=404, detail="Processing ID not found")
    
    del processing_results[processing_id]
    return {"message": f"Results for {processing_id} deleted successfully"}


@router.post('/batch_process', response_model=BatchProcessingResult)
async def batch_process_asl_data(
    request: BatchProcessingRequest,
    background_tasks: BackgroundTasks
):
    """Process multiple ASL datasets in batch."""
    batch_id = str(uuid.uuid4())
    logger.info(f"Starting batch processing: {batch_id} with {len(request.processing_requests)} datasets")
    
    # Start background batch processing
    background_tasks.add_task(_process_batch_background, batch_id, request)
    
    return BatchProcessingResult(
        total_datasets=len(request.processing_requests),
        completed_successfully=0,
        failed_processing=0,
        processing_time_seconds=0.0,
        results=[]
    )


async def _process_batch_background(batch_id: str, request: BatchProcessingRequest):
    """Background task for batch processing."""
    # This is a simplified implementation
    # In practice, you'd want more sophisticated batch processing
    logger.info(f"Batch processing {batch_id} started")


@router.post('/upload_files')
async def upload_asl_files(
    asl_file: UploadFile = File(...),
    m0_file: Optional[UploadFile] = File(None),
    mask_file: Optional[UploadFile] = File(None),
    atlas_file: Optional[UploadFile] = File(None)
):
    """
    Upload ASL files for processing.
    
    Returns file paths that can be used in processing requests.
    """
    upload_dir = Path(settings.upload_directory)
    upload_dir.mkdir(exist_ok=True)
    
    uploaded_files = {}
    
    # Save ASL file
    asl_path = upload_dir / f"{uuid.uuid4()}_{asl_file.filename}"
    with open(asl_path, "wb") as f:
        content = await asl_file.read()
        f.write(content)
    uploaded_files["asl_file"] = str(asl_path)
    
    # Save optional files
    if m0_file:
        m0_path = upload_dir / f"{uuid.uuid4()}_{m0_file.filename}"
        with open(m0_path, "wb") as f:
            content = await m0_file.read()
            f.write(content)
        uploaded_files["m0_file"] = str(m0_path)
    
    if mask_file:
        mask_path = upload_dir / f"{uuid.uuid4()}_{mask_file.filename}"
        with open(mask_path, "wb") as f:
            content = await mask_file.read()
            f.write(content)
        uploaded_files["mask_file"] = str(mask_path)
    
    if atlas_file:
        atlas_path = upload_dir / f"{uuid.uuid4()}_{atlas_file.filename}"
        with open(atlas_path, "wb") as f:
            content = await atlas_file.read()
            f.write(content)
        uploaded_files["atlas_file"] = str(atlas_path)
    
    return {"uploaded_files": uploaded_files, "message": "Files uploaded successfully"}


@router.get('/download/{processing_id}/{file_type}')
async def download_results(processing_id: str, file_type: str):
    """Download processing result files."""
    if processing_id not in processing_results:
        raise HTTPException(status_code=404, detail="Processing ID not found")
    
    result = processing_results[processing_id]
    if not result.output_files or file_type not in result.output_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_path = Path(result.output_files[file_type])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")
    
    return FileResponse(
        path=file_path,
        filename=file_path.name,
        media_type='application/octet-stream'
    )


@router.post('/visualize')
async def generate_visualizations(request: VisualizationRequest):
    """Generate visualization plots for processing results."""
    if request.processing_result_id not in processing_results:
        raise HTTPException(status_code=404, detail="Processing ID not found")
    
    # This would generate various plots based on the request
    # Implementation would depend on visualization requirements
    
    return {"message": "Visualization generation not fully implemented"}


@router.post('/configure')
async def update_configuration(config: ConfigurationUpdate):
    """Update processing configuration."""
    # This would update global configuration settings
    # Implementation would depend on configuration management approach
    
    return {"message": "Configuration update not fully implemented"}


def _save_processing_results(
    processing_id: str,
    processed_data: Dict,
    cbf_results: Optional[Dict],
    quality_results: Optional[Dict],
    output_dir: Path
) -> Dict[str, str]:
    """Save processing results to files."""
    output_dir.mkdir(exist_ok=True, parents=True)
    output_files = {}
    
    try:
        import nibabel as nib
        
        # Save CBF map if available
        if cbf_results and 'cbf' in cbf_results:
            cbf_path = output_dir / f"{processing_id}_cbf.nii.gz"
            # Create NIfTI image (simplified - would need proper affine matrix)
            cbf_img = nib.Nifti1Image(
                cbf_results['cbf'].astype(np.float32),
                affine=np.eye(4)
            )
            nib.save(cbf_img, cbf_path)
            output_files["cbf_map"] = str(cbf_path)
        
        # Save quality report as JSON
        if quality_results:
            import json
            quality_path = output_dir / f"{processing_id}_quality_report.json"
            with open(quality_path, 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                quality_json = _convert_numpy_types(quality_results)
                json.dump(quality_json, f, indent=2, default=str)
            output_files["quality_report"] = str(quality_path)
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")
    
    return output_files


def _convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj
