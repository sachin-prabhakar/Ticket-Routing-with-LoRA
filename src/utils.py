"""
Utility functions for ticket routing project.
Includes configuration loading, logging, seeding, and timing utilities.
"""

import os
import yaml
import json
import random
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import time
from contextlib import contextmanager

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Make deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@contextmanager
def Timer():
    """Context manager for timing code execution."""
    start_time = time.time()
    timer = type('Timer', (), {'elapsed_time': 0})()
    
    try:
        yield timer
    finally:
        timer.elapsed_time = time.time() - start_time

def get_device_info() -> Dict[str, Any]:
    """
    Get information about available devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        'cpu_count': os.cpu_count(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'mps_available': torch.backends.mps.is_available(),
    }
    
    if torch.cuda.is_available():
        info.update({
            'cuda_device_count': torch.cuda.device_count(),
            'cuda_current_device': torch.cuda.current_device(),
            'cuda_device_name': torch.cuda.get_device_name(),
            'cuda_memory_allocated': torch.cuda.memory_allocated(),
            'cuda_memory_reserved': torch.cuda.memory_reserved(),
        })
    
    return info

def print_device_info():
    """Print device information to console."""
    info = get_device_info()
    
    print("=" * 50)
    print("DEVICE INFORMATION")
    print("=" * 50)
    print(f"PyTorch version: {info['torch_version']}")
    print(f"CPU count: {info['cpu_count']}")
    print(f"CUDA available: {info['cuda_available']}")
    print(f"MPS available: {info['mps_available']}")
    
    if info['cuda_available']:
        print(f"CUDA devices: {info['cuda_device_count']}")
        print(f"Current CUDA device: {info['cuda_current_device']}")
        print(f"CUDA device name: {info['cuda_device_name']}")
        print(f"CUDA memory allocated: {info['cuda_memory_allocated'] / 1024**2:.1f} MB")
        print(f"CUDA memory reserved: {info['cuda_memory_reserved'] / 1024**2:.1f} MB")
    
    print("=" * 50)

def ensure_dir(path: str) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj

def save_json(data: Any, file_path: str, indent: int = 2) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Output file path
        indent: JSON indentation
    """
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent)

def load_json(file_path: str) -> Any:
    """
    Load data from JSON file.
    
    Args:
        file_path: Input file path
        
    Returns:
        Loaded data
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def format_time(seconds: float) -> str:
    """
    Format time duration in human-readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"

def get_model_size(model: torch.nn.Module) -> Dict[str, int]:
    """
    Get model size information.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with size information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': total_params - trainable_params
    }

def print_model_summary(model: torch.nn.Module, input_size: tuple = None):
    """
    Print model summary.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size for parameter counting
    """
    size_info = get_model_size(model)
    
    print("=" * 50)
    print("MODEL SUMMARY")
    print("=" * 50)
    print(f"Total parameters: {size_info['total_params']:,}")
    print(f"Trainable parameters: {size_info['trainable_params']:,}")
    print(f"Frozen parameters: {size_info['frozen_params']:,}")
    print(f"Trainable ratio: {size_info['trainable_params']/size_info['total_params']:.2%}")
    print("=" * 50)

def create_experiment_dir(base_dir: str, experiment_name: str) -> Path:
    """
    Create experiment directory with timestamp.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Name of the experiment
        
    Returns:
        Path to experiment directory
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    return exp_dir

def log_experiment_info(config: Dict[str, Any], output_dir: str):
    """
    Log experiment information to file.
    
    Args:
        config: Experiment configuration
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    
    # Save config
    save_config(config, output_path / "config.yaml")
    
    # Save device info
    device_info = get_device_info()
    save_json(device_info, output_path / "device_info.json")
    
    # Save git info (if available)
    try:
        import subprocess
        git_info = {
            'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
            'branch': subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode().strip(),
            'status': subprocess.check_output(['git', 'status', '--porcelain']).decode().strip()
        }
        save_json(git_info, output_path / "git_info.json")
    except:
        pass  # Git not available or not in git repo
