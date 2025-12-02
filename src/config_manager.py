"""
Configuration management for ONNX and WebSocket integration.

This module provides centralized configuration loading from environment
variables and configuration files.
"""

import logging
import os
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path
from .models import Config


logger = logging.getLogger(__name__)


@dataclass
class ONNXConfig:
    """ONNX Runtime configuration.
    
    Attributes:
        model_path: Path to ONNX model file
        providers: List of execution providers (e.g., ['CPUExecutionProvider'])
        inter_op_num_threads: Number of threads for inter-op parallelism
        intra_op_num_threads: Number of threads for intra-op parallelism
        enable: Whether to use ONNX (if False, fallback to PyTorch)
    """
    model_path: str = "models/yolov8s.onnx"
    providers: List[str] = field(default_factory=lambda: ['CPUExecutionProvider'])
    inter_op_num_threads: int = 4
    intra_op_num_threads: int = 4
    enable: bool = True


@dataclass
class WebSocketConfig:
    """WebSocket connection configuration.
    
    Attributes:
        url: WebSocket server URL (detection devices connect to /device)
        reconnect_interval: Initial reconnection interval in seconds
        max_reconnect_interval: Maximum reconnection interval in seconds
        buffer_size: Maximum number of events to buffer when disconnected
        enable: Whether WebSocket is enabled
    """
    url: str = "ws://localhost:8001/device"  # Detection devices use /device endpoint
    reconnect_interval: float = 1.0
    max_reconnect_interval: float = 60.0
    buffer_size: int = 1000
    enable: bool = True


@dataclass
class SystemConfig:
    """Complete system configuration.
    
    Attributes:
        onnx: ONNX Runtime configuration
        websocket: WebSocket configuration
        video: Video processing configuration (existing Config)
        source_id: Unique identifier for this camera/source
    """
    onnx: ONNXConfig = field(default_factory=ONNXConfig)
    websocket: WebSocketConfig = field(default_factory=WebSocketConfig)
    video: Config = field(default_factory=Config)
    source_id: str = "camera_01"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            "onnx": asdict(self.onnx),
            "websocket": asdict(self.websocket),
            "video": asdict(self.video),
            "source_id": self.source_id
        }


class ConfigManager:
    """Manages configuration loading from various sources."""
    
    @staticmethod
    def load_from_env() -> SystemConfig:
        """Load configuration from environment variables.
        
        Environment variables:
            - ONNX_MODEL_PATH: Path to ONNX model
            - ONNX_ENABLE: Enable ONNX (true/false)
            - ONNX_PROVIDERS: Comma-separated list of providers
            - ONNX_THREADS: Number of threads (sets both inter and intra)
            - WS_URL: WebSocket server URL
            - WS_ENABLE: Enable WebSocket (true/false)
            - WS_BUFFER_SIZE: Buffer size for events
            - SOURCE_ID: Camera/source identifier
            - MODEL_PATH: PyTorch model path (fallback)
            - CONFIDENCE: Confidence threshold
            - TRACKING_DISTANCE: Tracking distance threshold
            
        Returns:
            SystemConfig with values from environment or defaults
        """
        logger.info("Loading configuration from environment variables")
        
        config = SystemConfig()
        
        # ONNX configuration
        if os.getenv('ONNX_MODEL_PATH'):
            config.onnx.model_path = os.getenv('ONNX_MODEL_PATH')
            logger.info(f"ONNX model path from env: {config.onnx.model_path}")
        
        if os.getenv('ONNX_ENABLE'):
            config.onnx.enable = os.getenv('ONNX_ENABLE').lower() in ('true', '1', 'yes')
            logger.info(f"ONNX enable from env: {config.onnx.enable}")
        
        if os.getenv('ONNX_PROVIDERS'):
            providers_str = os.getenv('ONNX_PROVIDERS')
            config.onnx.providers = [p.strip() for p in providers_str.split(',')]
            logger.info(f"ONNX providers from env: {config.onnx.providers}")
        
        if os.getenv('ONNX_THREADS'):
            try:
                threads = int(os.getenv('ONNX_THREADS'))
                config.onnx.inter_op_num_threads = threads
                config.onnx.intra_op_num_threads = threads
                logger.info(f"ONNX threads from env: {threads}")
            except ValueError:
                logger.warning(f"Invalid ONNX_THREADS value, using default")
        
        # WebSocket configuration
        if os.getenv('WS_URL'):
            config.websocket.url = os.getenv('WS_URL')
            logger.info(f"WebSocket URL from env: {config.websocket.url}")
        
        if os.getenv('WS_ENABLE'):
            config.websocket.enable = os.getenv('WS_ENABLE').lower() in ('true', '1', 'yes')
            logger.info(f"WebSocket enable from env: {config.websocket.enable}")
        
        if os.getenv('WS_BUFFER_SIZE'):
            try:
                config.websocket.buffer_size = int(os.getenv('WS_BUFFER_SIZE'))
                logger.info(f"WebSocket buffer size from env: {config.websocket.buffer_size}")
            except ValueError:
                logger.warning(f"Invalid WS_BUFFER_SIZE value, using default")
        
        # Source ID
        if os.getenv('SOURCE_ID'):
            config.source_id = os.getenv('SOURCE_ID')
            logger.info(f"Source ID from env: {config.source_id}")
        
        # Video configuration
        if os.getenv('MODEL_PATH'):
            config.video.model_path = os.getenv('MODEL_PATH')
            logger.info(f"PyTorch model path from env: {config.video.model_path}")
        
        if os.getenv('CONFIDENCE'):
            try:
                config.video.confidence_threshold = float(os.getenv('CONFIDENCE'))
                logger.info(f"Confidence threshold from env: {config.video.confidence_threshold}")
            except ValueError:
                logger.warning(f"Invalid CONFIDENCE value, using default")
        
        if os.getenv('TRACKING_DISTANCE'):
            try:
                config.video.tracking_distance = float(os.getenv('TRACKING_DISTANCE'))
                logger.info(f"Tracking distance from env: {config.video.tracking_distance}")
            except ValueError:
                logger.warning(f"Invalid TRACKING_DISTANCE value, using default")
        
        # Validate configuration
        ConfigManager._validate_config(config)
        
        return config
    
    @staticmethod
    def load_from_file(path: str) -> SystemConfig:
        """Load configuration from JSON file.
        
        Args:
            path: Path to JSON configuration file
            
        Returns:
            SystemConfig loaded from file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        logger.info(f"Loading configuration from file: {path}")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            config = SystemConfig()
            
            # Load ONNX config
            if 'onnx' in data:
                onnx_data = data['onnx']
                if 'model_path' in onnx_data:
                    config.onnx.model_path = onnx_data['model_path']
                if 'providers' in onnx_data:
                    config.onnx.providers = onnx_data['providers']
                if 'inter_op_num_threads' in onnx_data:
                    config.onnx.inter_op_num_threads = onnx_data['inter_op_num_threads']
                if 'intra_op_num_threads' in onnx_data:
                    config.onnx.intra_op_num_threads = onnx_data['intra_op_num_threads']
                if 'enable' in onnx_data:
                    config.onnx.enable = onnx_data['enable']
            
            # Load WebSocket config
            if 'websocket' in data:
                ws_data = data['websocket']
                if 'url' in ws_data:
                    config.websocket.url = ws_data['url']
                if 'reconnect_interval' in ws_data:
                    config.websocket.reconnect_interval = ws_data['reconnect_interval']
                if 'max_reconnect_interval' in ws_data:
                    config.websocket.max_reconnect_interval = ws_data['max_reconnect_interval']
                if 'buffer_size' in ws_data:
                    config.websocket.buffer_size = ws_data['buffer_size']
                if 'enable' in ws_data:
                    config.websocket.enable = ws_data['enable']
            
            # Load video config
            if 'video' in data:
                video_data = data['video']
                if 'model_path' in video_data:
                    config.video.model_path = video_data['model_path']
                if 'confidence_threshold' in video_data:
                    config.video.confidence_threshold = video_data['confidence_threshold']
                if 'tracking_distance' in video_data:
                    config.video.tracking_distance = video_data['tracking_distance']
                if 'roi_file' in video_data:
                    config.video.roi_file = video_data['roi_file']
            
            # Load source ID
            if 'source_id' in data:
                config.source_id = data['source_id']
            
            logger.info("Configuration loaded successfully from file")
            
            # Validate configuration
            ConfigManager._validate_config(config)
            
            return config
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            raise ValueError(f"Invalid JSON in config file: {e}")
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            raise
    
    @staticmethod
    def _validate_config(config: SystemConfig) -> None:
        """Validate configuration and apply fallbacks for invalid values.
        
        Args:
            config: SystemConfig to validate
        """
        # Validate confidence threshold
        if not (0.0 <= config.video.confidence_threshold <= 1.0):
            logger.warning(
                f"Invalid confidence threshold {config.video.confidence_threshold}, "
                f"using default 0.5"
            )
            config.video.confidence_threshold = 0.5
        
        # Validate tracking distance
        if config.video.tracking_distance <= 0:
            logger.warning(
                f"Invalid tracking distance {config.video.tracking_distance}, "
                f"using default 50.0"
            )
            config.video.tracking_distance = 50.0
        
        # Validate thread counts
        if config.onnx.inter_op_num_threads <= 0:
            logger.warning(
                f"Invalid inter_op_num_threads {config.onnx.inter_op_num_threads}, "
                f"using default 4"
            )
            config.onnx.inter_op_num_threads = 4
        
        if config.onnx.intra_op_num_threads <= 0:
            logger.warning(
                f"Invalid intra_op_num_threads {config.onnx.intra_op_num_threads}, "
                f"using default 4"
            )
            config.onnx.intra_op_num_threads = 4
        
        # Validate buffer size
        if config.websocket.buffer_size <= 0:
            logger.warning(
                f"Invalid buffer size {config.websocket.buffer_size}, "
                f"using default 1000"
            )
            config.websocket.buffer_size = 1000
        
        # Validate reconnect intervals
        if config.websocket.reconnect_interval <= 0:
            logger.warning(
                f"Invalid reconnect_interval {config.websocket.reconnect_interval}, "
                f"using default 1.0"
            )
            config.websocket.reconnect_interval = 1.0
        
        if config.websocket.max_reconnect_interval < config.websocket.reconnect_interval:
            logger.warning(
                f"max_reconnect_interval must be >= reconnect_interval, "
                f"using default 60.0"
            )
            config.websocket.max_reconnect_interval = 60.0
        
        # Validate WebSocket URL format
        if config.websocket.enable:
            if not config.websocket.url.startswith(('ws://', 'wss://')):
                logger.warning(
                    f"Invalid WebSocket URL format: {config.websocket.url}, "
                    f"disabling WebSocket"
                )
                config.websocket.enable = False
        
        logger.info("Configuration validation completed")
    
    @staticmethod
    def save_to_file(config: SystemConfig, path: str) -> None:
        """Save configuration to JSON file.
        
        Args:
            config: SystemConfig to save
            path: Path where to save the configuration
        """
        logger.info(f"Saving configuration to file: {path}")
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            with open(path, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)
            
            logger.info("Configuration saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save config file: {e}")
            raise
