"""
Ticket Routing LoRA Package

A demonstration of efficient ticket routing using LoRA-tuned language models.
Supports both Apple Silicon (MPS) and NVIDIA CUDA hardware.
"""

__version__ = "1.0.0"
__author__ = "Ticket Routing Team"

from .data import load_synthetic, load_ag_news, get_data_loader, validate_dataframe
from .preprocess import preprocess_text, split_data, create_label_mapping, preprocess_dataset
from .model import TicketRoutingModel
from .train import TicketTrainer, load_config
from .eval import TicketEvaluator, evaluate_model_predictions
from .infer import TicketInference
from .serve import TicketServer
from .utils import (
    load_config, save_config, setup_logging, set_seed, Timer,
    get_device_info, print_device_info, ensure_dir, save_json, load_json
)

__all__ = [
    # Data utilities
    'load_synthetic', 'load_ag_news', 'get_data_loader', 'validate_dataframe',
    
    # Preprocessing
    'preprocess_text', 'split_data', 'create_label_mapping', 'preprocess_dataset',
    
    # Model
    'TicketRoutingModel',
    
    # Training
    'TicketTrainer', 'load_config',
    
    # Evaluation
    'TicketEvaluator', 'evaluate_model_predictions',
    
    # Inference
    'TicketInference',
    
    # Serving
    'TicketServer',
    
    # Utilities
    'save_config', 'setup_logging', 'set_seed', 'Timer',
    'get_device_info', 'print_device_info', 'ensure_dir', 'save_json', 'load_json'
]
