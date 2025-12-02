"""
Detector factory for creating appropriate detector instances.

This module provides factory functions to create detector instances with
automatic fallback from ONNX to PyTorch when needed.
"""

import logging
import os
from typing import Union, List, Optional
from .detector import PersonDetector
from .onnx_detector import ONNXPersonDetector


logger = logging.getLogger(__name__)


def create_detector(
    onnx_model_path: Optional[str] = None,
    pytorch_model_path: Optional[str] = None,
    confidence_threshold: float = 0.5,
    use_onnx: bool = True,
    onnx_providers: Optional[List[str]] = None,
    inter_op_threads: int = 4,
    intra_op_threads: int = 4
) -> Union[ONNXPersonDetector, PersonDetector]:
    """Create appropriate detector with automatic fallback.
    
    Attempts to create ONNX detector first if enabled and model exists.
    Falls back to PyTorch detector if ONNX fails or is disabled.
    
    Args:
        onnx_model_path: Path to ONNX model file
        pytorch_model_path: Path to PyTorch model file (fallback)
        confidence_threshold: Minimum confidence for detections
        use_onnx: Whether to attempt using ONNX
        onnx_providers: List of ONNX Runtime providers
        inter_op_threads: Inter-op thread count for ONNX
        intra_op_threads: Intra-op thread count for ONNX
        
    Returns:
        ONNXPersonDetector or PersonDetector instance
        
    Raises:
        Exception: If no valid detector can be created
    """
    detector = None
    
    # Try ONNX first if enabled
    if use_onnx and onnx_model_path:
        logger.info("Attempting to create ONNX detector")
        
        # Validate ONNX model exists
        if not os.path.exists(onnx_model_path):
            logger.warning(f"ONNX model not found: {onnx_model_path}")
            logger.info("Will attempt PyTorch fallback")
        else:
            try:
                # Set default providers if not specified
                if onnx_providers is None:
                    onnx_providers = ['CPUExecutionProvider']
                
                logger.info(f"Loading ONNX model: {onnx_model_path}")
                logger.info(f"Providers: {onnx_providers}")
                
                detector = ONNXPersonDetector(
                    model_path=onnx_model_path,
                    confidence_threshold=confidence_threshold,
                    providers=onnx_providers,
                    inter_op_num_threads=inter_op_threads,
                    intra_op_num_threads=intra_op_threads
                )
                
                logger.info("✓ ONNX detector created successfully")
                return detector
                
            except ImportError as e:
                logger.error(f"ONNX Runtime not available: {e}")
                logger.info("Install with: pip install onnxruntime")
                logger.info("Falling back to PyTorch detector")
                
            except Exception as e:
                logger.error(f"Failed to create ONNX detector: {e}")
                logger.info("Falling back to PyTorch detector")
    
    # Fallback to PyTorch detector
    if pytorch_model_path:
        logger.info("Creating PyTorch detector")
        
        # Validate PyTorch model exists
        if not os.path.exists(pytorch_model_path):
            logger.error(f"PyTorch model not found: {pytorch_model_path}")
            raise FileNotFoundError(f"PyTorch model not found: {pytorch_model_path}")
        
        try:
            logger.info(f"Loading PyTorch model: {pytorch_model_path}")
            
            detector = PersonDetector(
                model_path=pytorch_model_path,
                confidence_threshold=confidence_threshold
            )
            
            logger.info("✓ PyTorch detector created successfully")
            return detector
            
        except Exception as e:
            logger.error(f"Failed to create PyTorch detector: {e}")
            raise Exception(f"Failed to create any detector: {e}")
    
    # No valid model path provided
    logger.error("No valid model path provided for detector creation")
    raise ValueError("Either onnx_model_path or pytorch_model_path must be provided")


def validate_model_file(model_path: str, model_type: str = "model") -> bool:
    """Validate that model file exists and is accessible.
    
    Args:
        model_path: Path to model file
        model_type: Type of model for logging (e.g., "ONNX", "PyTorch")
        
    Returns:
        True if model file is valid, False otherwise
    """
    if not model_path:
        logger.error(f"{model_type} model path is empty")
        return False
    
    if not os.path.exists(model_path):
        logger.error(f"{model_type} model file not found: {model_path}")
        return False
    
    if not os.path.isfile(model_path):
        logger.error(f"{model_type} model path is not a file: {model_path}")
        return False
    
    # Check file size (should be > 0)
    file_size = os.path.getsize(model_path)
    if file_size == 0:
        logger.error(f"{model_type} model file is empty: {model_path}")
        return False
    
    logger.info(f"✓ {model_type} model file validated: {model_path} ({file_size / 1024 / 1024:.2f} MB)")
    return True


def get_recommended_providers() -> List[str]:
    """Get recommended ONNX Runtime providers for current platform.
    
    Returns:
        List of recommended provider names
    """
    import platform
    
    system = platform.system()
    machine = platform.machine()
    
    providers = []
    
    # Check for CUDA availability
    try:
        import onnxruntime as ort
        available_providers = ort.get_available_providers()
        
        # Prefer CUDA if available
        if 'CUDAExecutionProvider' in available_providers:
            providers.append('CUDAExecutionProvider')
            logger.info("CUDA provider available")
        
        # Check for other accelerators
        if 'TensorrtExecutionProvider' in available_providers:
            providers.append('TensorrtExecutionProvider')
            logger.info("TensorRT provider available")
        
    except ImportError:
        logger.warning("onnxruntime not installed, cannot check available providers")
    
    # Always include CPU as fallback
    providers.append('CPUExecutionProvider')
    
    logger.info(f"Recommended providers for {system}/{machine}: {providers}")
    
    return providers


def log_detector_info(detector: Union[ONNXPersonDetector, PersonDetector]) -> None:
    """Log information about the created detector.
    
    Args:
        detector: Detector instance to log info about
    """
    if isinstance(detector, ONNXPersonDetector):
        logger.info("=== ONNX Detector Info ===")
        logger.info(f"Model: {detector.model_path}")
        logger.info(f"Input size: {detector.input_height}x{detector.input_width}")
        logger.info(f"Confidence threshold: {detector.confidence_threshold}")
        logger.info(f"Providers: {detector.session.get_providers()}")
        logger.info("========================")
    elif isinstance(detector, PersonDetector):
        logger.info("=== PyTorch Detector Info ===")
        logger.info(f"Confidence threshold: {detector.confidence_threshold}")
        logger.info("===========================")
    else:
        logger.warning(f"Unknown detector type: {type(detector)}")
