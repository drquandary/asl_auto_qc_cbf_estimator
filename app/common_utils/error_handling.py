"""Common error handling utilities for BSC AI Apps."""

from typing import Dict, Any, Optional, Union
from fastapi import HTTPException
import logging
from .logging_config import get_logger

logger = get_logger(__name__)

class BSCAppError(Exception):
    """Base exception class for BSC AI Apps."""
    def __init__(self, message: str, status_code: int = 500, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}

class ValidationError(BSCAppError):
    """Error for invalid input data."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=400, details=details)

class ConfigurationError(BSCAppError):
    """Error for configuration issues."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=500, details=details)

class ProcessingError(BSCAppError):
    """Error for processing failures."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=422, details=details)

def handle_errors(func):
    """Decorator to handle errors in API endpoints."""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except BSCAppError as e:
            logger.error(f"BSCAppError in {func.__name__}: {e.message}", extra=e.details)
            raise HTTPException(status_code=e.status_code, detail=e.message)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")
    return wrapper

def handle_sync_errors(func):
    """Decorator to handle errors in synchronous functions."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except BSCAppError as e:
            logger.error(f"BSCAppError in {func.__name__}: {e.message}", extra=e.details)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            raise
    return wrapper

def create_error_response(message: str, status_code: int = 500, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a standardized error response."""
    response = {
        "error": {
            "message": message,
            "status_code": status_code
        }
    }
    if details:
        response["error"]["details"] = details
    return response

def log_error(error: Exception, context: Optional[Dict[str, Any]] = None):
    """Log an error with context."""
    context = context or {}
    logger.error(f"Error: {str(error)}", extra=context)