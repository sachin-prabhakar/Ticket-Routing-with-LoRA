"""
Export script for Triton Inference Server deployment.
Converts LoRA model to TorchScript/ONNX format for production serving.
"""

import argparse
import torch
import os
from pathlib import Path
import logging

from src.model import TicketRoutingModel
from src.utils import setup_logging, load_json

logger = logging.getLogger(__name__)

def export_for_triton(model_path: str, output_dir: str, format: str = "torchscript"):
    """
    Export LoRA model for Triton Inference Server.
    
    Args:
        model_path: Path to saved LoRA model
        output_dir: Output directory for exported model
        format: Export format ("torchscript" or "onnx")
    """
    logger.info(f"Exporting model from {model_path} to {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = TicketRoutingModel(
        model_name="EleutherAI/pythia-410m-deduped",
        device="cpu"  # Export on CPU for compatibility
    )
    model.load_model(model_path)
    
    # Load label mapping
    mapping_path = Path(model_path) / "label_mapping.json"
    if mapping_path.exists():
        label_mapping = load_json(str(mapping_path))
        logger.info(f"Loaded label mapping: {label_mapping}")
    else:
        logger.warning("Label mapping not found, using default")
        label_mapping = {"sales": 0, "tech_support": 1, "general": 2, "billing": 3}
    
    # Merge LoRA weights for export
    logger.info("Merging LoRA weights...")
    merged_model = model.model.merge_and_unload()
    
    # Create dummy input for tracing
    dummy_text = "This is a test ticket for tracing"
    dummy_inputs = model.tokenize([dummy_text])
    
    # Export based on format
    if format.lower() == "torchscript":
        export_torchscript(merged_model, dummy_inputs, output_dir)
    elif format.lower() == "onnx":
        export_onnx(merged_model, dummy_inputs, output_dir)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    # Save label mapping and metadata
    save_metadata(output_dir, label_mapping, model_path)
    
    logger.info(f"Model exported successfully to {output_dir}")

def export_torchscript(model, dummy_inputs, output_dir):
    """Export model to TorchScript format."""
    logger.info("Exporting to TorchScript...")
    
    # Trace the model
    traced_model = torch.jit.trace(
        model,
        (dummy_inputs["input_ids"], dummy_inputs["attention_mask"])
    )
    
    # Save traced model
    torchscript_path = os.path.join(output_dir, "model.pt")
    traced_model.save(torchscript_path)
    
    logger.info(f"TorchScript model saved to {torchscript_path}")

def export_onnx(model, dummy_inputs, output_dir):
    """Export model to ONNX format."""
    logger.info("Exporting to ONNX...")
    
    # Create dummy inputs for ONNX export
    input_ids = dummy_inputs["input_ids"]
    attention_mask = dummy_inputs["attention_mask"]
    
    # Export to ONNX
    onnx_path = os.path.join(output_dir, "model.onnx")
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"}
        },
        opset_version=11
    )
    
    logger.info(f"ONNX model saved to {onnx_path}")

def save_metadata(output_dir, label_mapping, original_model_path):
    """Save metadata for Triton deployment."""
    metadata = {
        "model_type": "ticket_routing_lora",
        "base_model": "EleutherAI/pythia-410m-deduped",
        "num_labels": len(label_mapping),
        "label_mapping": label_mapping,
        "max_length": 256,
        "original_model_path": original_model_path,
        "export_timestamp": torch.datetime.now().isoformat()
    }
    
    import json
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Metadata saved to {metadata_path}")

def create_triton_config(output_dir, model_name="ticket_routing"):
    """Create Triton model configuration."""
    config = {
        "name": model_name,
        "platform": "pytorch_libtorch",
        "max_batch_size": 32,
        "input": [
            {
                "name": "input_ids",
                "data_type": "TYPE_INT64",
                "dims": [256]
            },
            {
                "name": "attention_mask", 
                "data_type": "TYPE_INT64",
                "dims": [256]
            }
        ],
        "output": [
            {
                "name": "logits",
                "data_type": "TYPE_FP32",
                "dims": [4]
            }
        ],
        "instance_group": [
            {
                "count": 1,
                "kind": "KIND_GPU"
            }
        ],
        "dynamic_batching": {
            "max_queue_delay_microseconds": 100
        }
    }
    
    import json
    config_path = os.path.join(output_dir, "config.pbtxt")
    
    # Convert to protobuf text format (simplified)
    config_text = f"""name: "{model_name}"
platform: "pytorch_libtorch"
max_batch_size: 32

input {{
  name: "input_ids"
  data_type: TYPE_INT64
  dims: [256]
}}

input {{
  name: "attention_mask"
  data_type: TYPE_INT64
  dims: [256]
}}

output {{
  name: "logits"
  data_type: TYPE_FP32
  dims: [4]
}}

instance_group {{
  count: 1
  kind: KIND_GPU
}}

dynamic_batching {{
  max_queue_delay_microseconds: 100
}}
"""
    
    with open(config_path, 'w') as f:
        f.write(config_text)
    
    logger.info(f"Triton config saved to {config_path}")

def main():
    """Main export function."""
    parser = argparse.ArgumentParser(description='Export LoRA model for Triton')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to saved LoRA model')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for exported model')
    parser.add_argument('--format', type=str, default='torchscript',
                       choices=['torchscript', 'onnx'],
                       help='Export format')
    parser.add_argument('--model-name', type=str, default='ticket_routing',
                       help='Model name for Triton')
    parser.add_argument('--create-triton-config', action='store_true',
                       help='Create Triton configuration file')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging('INFO')
    
    # Export model
    export_for_triton(args.model_path, args.output_dir, args.format)
    
    # Create Triton config if requested
    if args.create_triton_config:
        create_triton_config(args.output_dir, args.model_name)
    
    logger.info("Export completed successfully!")
    logger.info(f"Deploy to Triton by copying {args.output_dir} to your Triton model repository")

if __name__ == "__main__":
    main()
