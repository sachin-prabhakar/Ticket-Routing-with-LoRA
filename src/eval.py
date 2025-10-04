"""
Evaluation pipeline for ticket routing models.

Features:
- Comprehensive metrics computation (accuracy, F1, precision, recall)
- Professional confusion matrix visualization
- Detailed classification reports with per-class analysis
- Top-k accuracy evaluation for production deployment
- Professional logging and result persistence
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from typing import Dict, Any, List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class TicketEvaluator:
    """Comprehensive evaluator for ticket routing model."""
    
    def __init__(self, label_mapping: Dict[str, int], output_dir: str = "reports"):
        """
        Initialize evaluator.
        
        Args:
            label_mapping: Mapping from label names to IDs
            output_dir: Directory to save evaluation results
        """
        self.label_mapping = label_mapping
        self.id_to_label = {v: k for k, v in label_mapping.items()}
        self.label_names = list(label_mapping.keys())
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized evaluator with {len(self.label_names)} classes")
    
    def compute_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        weighted_f1 = f1_score(y_true, y_pred, average='weighted')
        macro_precision = precision_score(y_true, y_pred, average='macro')
        macro_recall = recall_score(y_true, y_pred, average='macro')
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        # Create per-class metrics dictionary
        per_class_metrics = {}
        for i, label_name in enumerate(self.label_names):
            per_class_metrics[label_name] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1': float(f1_per_class[i])
            }
        
        metrics = {
            'accuracy': float(accuracy),
            'macro_f1': float(macro_f1),
            'weighted_f1': float(weighted_f1),
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'per_class': per_class_metrics
        }
        
        logger.info(f"Computed metrics: accuracy={accuracy:.4f}, macro_f1={macro_f1:.4f}")
        return metrics
    
    def compute_top_k_accuracy(self, y_true: List[int], y_pred_probs: np.ndarray, k: int = 3) -> Dict[str, float]:
        """
        Compute top-k accuracy metrics.
        
        Args:
            y_true: True labels
            y_pred_probs: Prediction probabilities (n_samples, n_classes)
            k: Number of top predictions to consider
            
        Returns:
            Dictionary of top-k accuracy metrics
        """
        top_k_accuracies = {}
        
        for k_val in range(1, k + 1):
            # Get top-k predictions
            top_k_preds = np.argsort(y_pred_probs, axis=1)[:, -k_val:]
            
            # Check if true label is in top-k
            correct = 0
            for i, true_label in enumerate(y_true):
                if true_label in top_k_preds[i]:
                    correct += 1
            
            top_k_accuracies[f'top_{k_val}_accuracy'] = correct / len(y_true)
        
        logger.info(f"Top-k accuracies: {top_k_accuracies}")
        return top_k_accuracies
    
    def create_confusion_matrix(self, y_true: List[int], y_pred: List[int], 
                              save_path: Optional[str] = None) -> np.ndarray:
        """
        Create and save confusion matrix visualization.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot
            
        Returns:
            Confusion matrix array
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_names,
                   yticklabels=self.label_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / 'confusion_matrix.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {save_path}")
        return cm
    
    def create_classification_report(self, y_true: List[int], y_pred: List[int],
                                   save_path: Optional[str] = None) -> str:
        """
        Create detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the report
            
        Returns:
            Classification report string
        """
        report = classification_report(
            y_true, y_pred,
            target_names=self.label_names,
            digits=4
        )
        
        # Save report
        if save_path is None:
            save_path = self.output_dir / 'classification_report.txt'
        
        with open(save_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Classification report saved to {save_path}")
        return report
    
    def evaluate_predictions(self, y_true: List[int], y_pred: List[int], 
                           y_pred_probs: Optional[np.ndarray] = None,
                           top_k: int = 3) -> Dict[str, Any]:
        """
        Comprehensive evaluation of predictions.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_probs: Prediction probabilities (optional)
            top_k: Number of top predictions for top-k metrics
            
        Returns:
            Complete evaluation results
        """
        logger.info(f"Evaluating {len(y_true)} predictions")
        
        # Basic metrics
        metrics = self.compute_metrics(y_true, y_pred)
        
        # Top-k accuracy (if probabilities provided)
        if y_pred_probs is not None:
            top_k_metrics = self.compute_top_k_accuracy(y_true, y_pred_probs, top_k)
            metrics.update(top_k_metrics)
        
        # Confusion matrix
        cm = self.create_confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        report = self.create_classification_report(y_true, y_pred)
        metrics['classification_report'] = report
        
        # Label distribution
        true_counts = pd.Series(y_true).value_counts().sort_index()
        pred_counts = pd.Series(y_pred).value_counts().sort_index()
        
        metrics['label_distribution'] = {
            'true_counts': {self.id_to_label[i]: int(count) for i, count in true_counts.items()},
            'pred_counts': {self.id_to_label[i]: int(count) for i, count in pred_counts.items()}
        }
        
        return metrics
    
    def save_evaluation_results(self, results: Dict[str, Any], 
                              filename: str = "evaluation_results.json"):
        """
        Save evaluation results to JSON file.
        
        Args:
            results: Evaluation results dictionary
            filename: Output filename
        """
        save_path = self.output_dir / filename
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = self._prepare_for_json(results)
        
        with open(save_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {save_path}")
    
    def _prepare_for_json(self, obj: Any) -> Any:
        """Prepare object for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        else:
            return obj
    
    def print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary."""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Macro F1: {results['macro_f1']:.4f}")
        print(f"Weighted F1: {results['weighted_f1']:.4f}")
        print(f"Macro Precision: {results['macro_precision']:.4f}")
        print(f"Macro Recall: {results['macro_recall']:.4f}")
        
        # Top-k accuracies
        for key, value in results.items():
            if key.startswith('top_') and key.endswith('_accuracy'):
                print(f"{key.replace('_', ' ').title()}: {value:.4f}")
        
        print("\nPer-Class Metrics:")
        print("-" * 40)
        for label, metrics in results['per_class'].items():
            print(f"{label:15} | P: {metrics['precision']:.3f} | R: {metrics['recall']:.3f} | F1: {metrics['f1']:.3f}")
        
        print("="*60)

def evaluate_model_predictions(model, test_texts: List[str], test_labels: List[str],
                             label_mapping: Dict[str, int], output_dir: str = "reports") -> Dict[str, Any]:
    """
    Evaluate model predictions on test data.
    
    Args:
        model: Trained model
        test_texts: Test text data
        test_labels: Test labels (string names)
        label_mapping: Label name to ID mapping
        output_dir: Output directory for results
        
    Returns:
        Evaluation results
    """
    logger.info(f"Evaluating model on {len(test_texts)} test examples")
    
    # Convert string labels to IDs
    test_label_ids = [label_mapping[label] for label in test_labels]
    
    # Get predictions
    predictions, probabilities = model.predict(test_texts)
    predictions = predictions.cpu().numpy()
    probabilities = probabilities.cpu().numpy()
    
    # Initialize evaluator
    evaluator = TicketEvaluator(label_mapping, output_dir)
    
    # Evaluate
    results = evaluator.evaluate_predictions(
        test_label_ids, predictions, probabilities, top_k=3
    )
    
    # Save results
    evaluator.save_evaluation_results(results)
    
    # Print summary
    evaluator.print_summary(results)
    
    return results

def load_evaluation_results(results_path: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results

def compare_models(results_paths: List[str], model_names: List[str]) -> pd.DataFrame:
    """
    Compare multiple model evaluation results.
    
    Args:
        results_paths: Paths to evaluation result files
        model_names: Names of models
        
    Returns:
        DataFrame with comparison metrics
    """
    comparison_data = []
    
    for path, name in zip(results_paths, model_names):
        results = load_evaluation_results(path)
        comparison_data.append({
            'model': name,
            'accuracy': results['accuracy'],
            'macro_f1': results['macro_f1'],
            'weighted_f1': results['weighted_f1'],
            'macro_precision': results['macro_precision'],
            'macro_recall': results['macro_recall']
        })
    
    df = pd.DataFrame(comparison_data)
    df = df.set_index('model')
    
    return df
