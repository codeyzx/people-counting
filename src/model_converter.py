"""
Model converter utility for converting YOLOv8 PyTorch models to ONNX format.

This module provides utilities to convert YOLOv8 models to ONNX format for
optimized inference on various platforms including Orange Pi 5.
"""

import logging
import os
from pathlib import Path
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class ModelConverter:
    """Converts YOLOv8 PyTorch models to ONNX format."""
    
    @staticmethod
    def convert_to_onnx(
        pytorch_model_path: str,
        onnx_output_path: str,
        input_size: Tuple[int, int] = (640, 640),
        opset_version: int = 12,
        simplify: bool = True
    ) -> bool:
        """Convert YOLOv8 PyTorch model to ONNX format.
        
        Args:
            pytorch_model_path: Path to YOLOv8 PyTorch model (.pt file)
            onnx_output_path: Path where ONNX model will be saved
            input_size: Input image size as (height, width)
            opset_version: ONNX opset version (12 recommended for compatibility)
            simplify: Whether to simplify the ONNX model
            
        Returns:
            True if conversion successful, False otherwise
            
        Raises:
            FileNotFoundError: If pytorch_model_path doesn't exist
            ValueError: If input parameters are invalid
        """
        try:
            # Validate input path
            if not os.path.exists(pytorch_model_path):
                raise FileNotFoundError(f"PyTorch model not found: {pytorch_model_path}")
            
            # Validate input size
            if len(input_size) != 2 or any(s <= 0 for s in input_size):
                raise ValueError(f"Invalid input size: {input_size}")
            
            logger.info(f"Converting model from {pytorch_model_path} to {onnx_output_path}")
            logger.info(f"Input size: {input_size}, Opset version: {opset_version}")
            
            # Import ultralytics here to avoid dependency if not needed
            from ultralytics import YOLO
            
            # Load the YOLOv8 model
            model = YOLO(pytorch_model_path)
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(onnx_output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Export to ONNX
            # The export method returns the path to the exported model
            export_path = model.export(
                format='onnx',
                imgsz=input_size,
                opset=opset_version,
                simplify=simplify,
                dynamic=False  # Static shape for better optimization
            )
            
            # Move/rename if needed
            if export_path != onnx_output_path:
                import shutil
                shutil.move(export_path, onnx_output_path)
            
            logger.info(f"Model successfully converted to {onnx_output_path}")
            
            # Verify the converted model
            if ModelConverter.verify_onnx_model(onnx_output_path):
                logger.info("ONNX model verification passed")
                return True
            else:
                logger.error("ONNX model verification failed")
                return False
                
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise
        except ValueError as e:
            logger.error(f"Invalid parameter: {e}")
            raise
        except ImportError as e:
            logger.error(f"Failed to import required library: {e}")
            logger.error("Make sure ultralytics is installed: pip install ultralytics")
            return False
        except Exception as e:
            logger.error(f"Failed to convert model: {e}")
            return False
    
    @staticmethod
    def verify_onnx_model(onnx_path: str) -> bool:
        """Verify that ONNX model can be loaded and is valid.
        
        Args:
            onnx_path: Path to ONNX model file
            
        Returns:
            True if model is valid and can be loaded, False otherwise
        """
        try:
            if not os.path.exists(onnx_path):
                logger.error(f"ONNX model not found: {onnx_path}")
                return False
            
            logger.info(f"Verifying ONNX model: {onnx_path}")
            
            # Try to load with onnx library for structure validation
            try:
                import onnx
                model = onnx.load(onnx_path)
                onnx.checker.check_model(model)
                logger.info("ONNX model structure is valid")
            except ImportError:
                logger.warning("onnx library not available, skipping structure check")
            except Exception as e:
                logger.error(f"ONNX structure validation failed: {e}")
                return False
            
            # Try to load with ONNX Runtime
            try:
                import onnxruntime as ort
                
                # Create inference session
                session = ort.InferenceSession(
                    onnx_path,
                    providers=['CPUExecutionProvider']
                )
                
                # Get model info
                inputs = session.get_inputs()
                outputs = session.get_outputs()
                
                logger.info(f"ONNX Runtime loaded model successfully")
                logger.info(f"Input: {inputs[0].name}, shape: {inputs[0].shape}")
                logger.info(f"Outputs: {len(outputs)} tensors")
                
                return True
                
            except ImportError:
                logger.error("onnxruntime not installed. Install with: pip install onnxruntime")
                return False
            except Exception as e:
                logger.error(f"Failed to load model with ONNX Runtime: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Model verification failed: {e}")
            return False


def main():
    """CLI interface for model conversion."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert YOLOv8 PyTorch model to ONNX format"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to input PyTorch model (.pt file)"
    )
    parser.add_argument(
        "output",
        type=str,
        help="Path for output ONNX model (.onnx file)"
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[640, 640],
        metavar=("HEIGHT", "WIDTH"),
        help="Input image size (default: 640 640)"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset version (default: 12)"
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Disable ONNX model simplification"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing ONNX model without conversion"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        if args.verify_only:
            # Verify existing model
            if ModelConverter.verify_onnx_model(args.output):
                print(f"✓ Model verification successful: {args.output}")
                return 0
            else:
                print(f"✗ Model verification failed: {args.output}")
                return 1
        else:
            # Convert model
            success = ModelConverter.convert_to_onnx(
                pytorch_model_path=args.input,
                onnx_output_path=args.output,
                input_size=tuple(args.input_size),
                opset_version=args.opset,
                simplify=not args.no_simplify
            )
            
            if success:
                print(f"✓ Conversion successful: {args.output}")
                return 0
            else:
                print(f"✗ Conversion failed")
                return 1
                
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
