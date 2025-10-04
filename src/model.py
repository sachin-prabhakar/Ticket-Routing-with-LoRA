"""
LoRA model implementation for efficient fine-tuning.

Features:
- Optimized for NVIDIA CUDA with Tensor Core acceleration
- Memory-efficient LoRA adaptation with configurable parameters
- Production-ready inference with batch processing
- Comprehensive parameter efficiency reporting
- Professional logging and error handling
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class TicketRoutingModel:
    """
    LoRA-tuned model for ticket routing classification.
    """
    
    def __init__(self, 
                 model_name: str = "EleutherAI/pythia-410m-deduped",
                 num_labels: int = 4,
                 max_length: int = 256,
                 device: Optional[str] = None,
                 dtype: Optional[torch.dtype] = None):
        """
        Initialize the ticket routing model with LoRA.
        
        Args:
            model_name: Hugging Face model name
            num_labels: Number of classification labels
            max_length: Maximum sequence length
            device: Device to load model on
            dtype: Data type for model weights
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.device = device or self._get_device()
        self.dtype = dtype or self._get_dtype()
        
        logger.info(f"Initializing model: {model_name}")
        logger.info(f"Device: {self.device}, dtype: {self.dtype}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Ensure padding token is properly set
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load base model
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            torch_dtype=self.dtype,
            device_map="auto" if self.device == "cuda" else None,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        # Move to device if not using device_map
        if self.device != "cuda":
            self.base_model = self.base_model.to(self.device)
        
        # Apply LoRA
        self.model = self._apply_lora()
        
        logger.info("Model initialization complete")
    
    def _get_device(self) -> str:
        """Detect best available device."""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def _get_dtype(self) -> torch.dtype:
        """Get appropriate dtype for device."""
        if self.device == "mps":
            return torch.float16
        elif self.device == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            return torch.float32
    
    def _apply_lora(self, 
                   r: int = 8,
                   alpha: int = 16,
                   dropout: float = 0.05,
                   target_modules: Optional[list] = None) -> nn.Module:
        """
        Apply LoRA configuration to the model.
        
        Args:
            r: LoRA rank
            alpha: LoRA alpha parameter
            dropout: LoRA dropout rate
            target_modules: List of modules to apply LoRA to
            
        Returns:
            LoRA-tuned model
        """
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc", "proj"]
        
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
        )
        
        model = get_peft_model(self.base_model, lora_config)
        
        logger.info(f"Applied LoRA with r={r}, alpha={alpha}, dropout={dropout}")
        logger.info(f"Target modules: {target_modules}")
        
        return model
    
    def report_trainable_params(self) -> Dict[str, Any]:
        """
        Report parameter efficiency statistics.
        
        Returns:
            Dictionary with parameter counts and efficiency metrics
        """
        # Get total parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Calculate efficiency metrics
        reduction_percent = (1 - trainable_params / total_params) * 100
        efficiency_ratio = total_params / trainable_params
        
        stats = {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": total_params - trainable_params,
            "reduction_percent": reduction_percent,
            "efficiency_ratio": efficiency_ratio
        }
        
        logger.info("=" * 50)
        logger.info("PARAMETER EFFICIENCY REPORT")
        logger.info("=" * 50)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Frozen parameters: {stats['frozen_params']:,}")
        logger.info(f"Parameter reduction: {reduction_percent:.2f}%")
        logger.info(f"Efficiency ratio: {efficiency_ratio:.1f}x")
        logger.info("=" * 50)
        
        return stats
    
    def tokenize(self, texts: list, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Tokenize input texts.
        
        Args:
            texts: List of input texts
            **kwargs: Additional tokenizer arguments
            
        Returns:
            Tokenized inputs
        """
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            **kwargs
        )
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Tokenized input IDs
            attention_mask: Attention mask
            labels: Optional labels for training
            
        Returns:
            Model outputs
        """
        # Move inputs to device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def predict(self, texts: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions on input texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        self.model.eval()
        
        with torch.no_grad():
            # Tokenize inputs
            inputs = self.tokenize(texts)
            
            # Forward pass
            outputs = self.forward(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            
            # Get predictions and probabilities
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            return predictions, probabilities
    
    def get_top_k_predictions(self, texts: list, k: int = 3) -> list:
        """
        Get top-k predictions with scores.
        
        Args:
            texts: List of input texts
            k: Number of top predictions to return
            
        Returns:
            List of top-k predictions for each text
        """
        self.model.eval()
        
        with torch.no_grad():
            # Tokenize inputs
            inputs = self.tokenize(texts)
            
            # Forward pass
            outputs = self.forward(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            
            # Get probabilities
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            
            # Get top-k predictions
            top_k_probs, top_k_indices = torch.topk(probabilities, k, dim=-1)
            
            # Convert to list format
            results = []
            for i in range(len(texts)):
                text_predictions = []
                for j in range(k):
                    text_predictions.append({
                        "label_id": top_k_indices[i][j].item(),
                        "score": top_k_probs[i][j].item()
                    })
                results.append(text_predictions)
            
            return results
    
    def save_model(self, path: str, save_merged: bool = False):
        """
        Save the model and tokenizer.
        
        Args:
            path: Path to save the model
            save_merged: Whether to save merged weights
        """
        if save_merged:
            # Save merged model
            merged_model = self.model.merge_and_unload()
            merged_model.save_pretrained(path)
            logger.info(f"Saved merged model to {path}")
        else:
            # Save LoRA adapter
            self.model.save_pretrained(path)
            logger.info(f"Saved LoRA adapter to {path}")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(path)
        logger.info(f"Saved tokenizer to {path}")
    
    def load_model(self, path: str):
        """
        Load a saved model and tokenizer.
        
        Args:
            path: Path to load the model from
        """
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
        # Load base model
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            torch_dtype=self.dtype
        )
        
        # Load LoRA adapter
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.base_model, path)
        
        # Move to device
        self.model = self.model.to(self.device)
        
        logger.info(f"Loaded model from {path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary with model metadata
        """
        param_stats = self.report_trainable_params()
        
        return {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "max_length": self.max_length,
            "device": self.device,
            "dtype": str(self.dtype),
            "parameter_stats": param_stats,
            "lora_config": self.model.peft_config if hasattr(self.model, 'peft_config') else None
        }
