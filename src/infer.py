"""
Inference pipeline for ticket routing models.

Features:
- High-performance batch processing with CUDA optimization
- Single text prediction with confidence scoring
- Production-ready error handling and logging
- Memory-efficient inference with automatic batching
- Comprehensive output formatting for integration
"""

import json
import sys
import argparse
import time
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

from .model import TicketRoutingModel
from .utils import setup_logging, Timer

logger = logging.getLogger(__name__)

class TicketInference:
    """Inference class for ticket routing model."""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to saved model
            device: Device to run inference on
        """
        self.model_path = model_path
        self.device = device
        
        # Load model and tokenizer
        self.model = TicketRoutingModel(
            model_name="EleutherAI/pythia-410m-deduped",
            device=self.device
        )
        self.model.load_model(model_path)
        
        # Load label mapping
        self.label_mapping = self._load_label_mapping()
        self.id_to_label = {v: k for k, v in self.label_mapping.items()}
        
        logger.info(f"Loaded model from {model_path}")
        logger.info(f"Label mapping: {self.label_mapping}")
    
    def _load_label_mapping(self) -> Dict[str, int]:
        """Load label mapping from saved model."""
        mapping_path = Path(self.model_path) / "label_mapping.json"
        
        if not mapping_path.exists():
            logger.warning(f"Label mapping not found at {mapping_path}, using default")
            return {"sales": 0, "tech_support": 1, "general": 2, "billing": 3}
        
        with open(mapping_path, 'r') as f:
            return json.load(f)
    
    def predict_single(self, text: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Predict label for a single text.
        
        Args:
            text: Input text
            top_k: Number of top predictions to return
            
        Returns:
            Prediction results
        """
        with Timer() as timer:
            # Get top-k predictions
            top_k_predictions = self.model.get_top_k_predictions([text], k=top_k)
            
            # Convert to readable format
            predictions = []
            for pred in top_k_predictions[0]:
                predictions.append({
                    "label": self.id_to_label[pred["label_id"]],
                    "score": pred["score"]
                })
            
            result = {
                "text": text,
                "predicted_label": predictions[0]["label"],
                "confidence": predictions[0]["score"],
                "top_k": predictions,
                "inference_ms": timer.elapsed_time * 1000
            }
            
            return result
    
    def predict_batch(self, texts: List[str], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Predict labels for a batch of texts.
        
        Args:
            texts: List of input texts
            top_k: Number of top predictions to return
            
        Returns:
            List of prediction results
        """
        logger.info(f"Processing batch of {len(texts)} texts")
        
        with Timer() as timer:
            # Get top-k predictions for all texts
            all_top_k_predictions = self.model.get_top_k_predictions(texts, k=top_k)
            
            # Convert to readable format
            results = []
            for i, text in enumerate(texts):
                predictions = []
                for pred in all_top_k_predictions[i]:
                    predictions.append({
                        "label": self.id_to_label[pred["label_id"]],
                        "score": pred["score"]
                    })
                
                result = {
                    "text": text,
                    "predicted_label": predictions[0]["label"],
                    "confidence": predictions[0]["score"],
                    "top_k": predictions
                }
                results.append(result)
            
            logger.info(f"Batch processing completed in {timer.elapsed_time:.2f}s")
            
            # Add timing info to each result
            avg_time_ms = (timer.elapsed_time * 1000) / len(texts)
            for result in results:
                result["inference_ms"] = avg_time_ms
            
            return results
    
    def predict_from_jsonl(self, input_path: str, output_path: str, 
                          top_k: int = 3, text_field: str = "text") -> None:
        """
        Predict labels from JSONL input file.
        
        Args:
            input_path: Path to input JSONL file
            output_path: Path to output JSONL file
            top_k: Number of top predictions to return
            text_field: Field name containing text in input
        """
        logger.info(f"Processing JSONL file: {input_path}")
        
        texts = []
        input_data = []
        
        # Read input file
        with open(input_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                texts.append(data[text_field])
                input_data.append(data)
        
        # Get predictions
        predictions = self.predict_batch(texts, top_k=top_k)
        
        # Combine input data with predictions
        output_data = []
        for input_item, prediction in zip(input_data, predictions):
            output_item = input_item.copy()
            output_item.update({
                "predicted_label": prediction["predicted_label"],
                "confidence": prediction["confidence"],
                "top_k": prediction["top_k"],
                "inference_ms": prediction["inference_ms"]
            })
            output_data.append(output_item)
        
        # Write output file
        with open(output_path, 'w') as f:
            for item in output_data:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Results saved to {output_path}")

def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Ticket routing inference')
    parser.add_argument('--model-path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--input', type=str, help='Input JSONL file (or stdin if not specified)')
    parser.add_argument('--output', type=str, help='Output JSONL file')
    parser.add_argument('--text', type=str, help='Single text to predict')
    parser.add_argument('--top-k', type=int, default=3, help='Number of top predictions')
    parser.add_argument('--device', type=str, help='Device to run on')
    parser.add_argument('--text-field', type=str, default='text', help='Text field name in JSONL')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging('INFO')
    
    # Initialize inference engine
    inference = TicketInference(args.model_path, device=args.device)
    
    if args.text:
        # Single text prediction
        result = inference.predict_single(args.text, top_k=args.top_k)
        print(json.dumps(result, indent=2))
    
    elif args.input:
        # Batch prediction from file
        if not args.output:
            args.output = args.input.replace('.jsonl', '_predictions.jsonl')
        
        inference.predict_from_jsonl(args.input, args.output, args.top_k, args.text_field)
    
    else:
        # Interactive mode from stdin
        print("Enter ticket text (Ctrl+D to exit):")
        for line in sys.stdin:
            text = line.strip()
            if text:
                result = inference.predict_single(text, top_k=args.top_k)
                print(json.dumps(result, indent=2))
                print()

if __name__ == "__main__":
    main()
