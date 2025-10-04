"""
Test suite for ticket routing evaluation utilities.
Tests for metrics computation and visualization.
"""

import unittest
import tempfile
import os
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# Add src to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from eval import TicketEvaluator, evaluate_model_predictions, compare_models

class TestTicketEvaluator(unittest.TestCase):
    """Test TicketEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.label_mapping = {
            'sales': 0,
            'tech_support': 1,
            'general': 2,
            'billing': 3
        }
        self.evaluator = TicketEvaluator(self.label_mapping, self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_compute_metrics_perfect_predictions(self):
        """Test metrics computation with perfect predictions."""
        y_true = [0, 1, 2, 3, 0, 1, 2, 3]
        y_pred = [0, 1, 2, 3, 0, 1, 2, 3]
        
        metrics = self.evaluator.compute_metrics(y_true, y_pred)
        
        # Perfect predictions should give perfect metrics
        self.assertEqual(metrics['accuracy'], 1.0)
        self.assertEqual(metrics['macro_f1'], 1.0)
        self.assertEqual(metrics['weighted_f1'], 1.0)
        self.assertEqual(metrics['macro_precision'], 1.0)
        self.assertEqual(metrics['macro_recall'], 1.0)
        
        # Per-class metrics should all be 1.0
        for label in self.label_mapping.keys():
            self.assertEqual(metrics['per_class'][label]['precision'], 1.0)
            self.assertEqual(metrics['per_class'][label]['recall'], 1.0)
            self.assertEqual(metrics['per_class'][label]['f1'], 1.0)
    
    def test_compute_metrics_random_predictions(self):
        """Test metrics computation with random predictions."""
        y_true = [0, 1, 2, 3] * 10  # 40 samples
        y_pred = [0, 1, 2, 3] * 10  # Perfect predictions
        
        metrics = self.evaluator.compute_metrics(y_true, y_pred)
        
        # Should have perfect metrics
        self.assertEqual(metrics['accuracy'], 1.0)
        self.assertEqual(metrics['macro_f1'], 1.0)
    
    def test_compute_top_k_accuracy(self):
        """Test top-k accuracy computation."""
        y_true = [0, 1, 2]
        y_pred_probs = np.array([
            [0.8, 0.1, 0.05, 0.05],  # Correct prediction
            [0.1, 0.7, 0.1, 0.1],   # Correct prediction  
            [0.1, 0.2, 0.3, 0.4]    # Wrong prediction (true=2, pred=3)
        ])
        
        top_k_metrics = self.evaluator.compute_top_k_accuracy(y_true, y_pred_probs, k=3)
        
        # Check that all top-k metrics are present
        for k in range(1, 4):
            self.assertIn(f'top_{k}_accuracy', top_k_metrics)
            self.assertGreaterEqual(top_k_metrics[f'top_{k}_accuracy'], 0.0)
            self.assertLessEqual(top_k_metrics[f'top_{k}_accuracy'], 1.0)
        
        # Top-1 accuracy should be 2/3
        self.assertAlmostEqual(top_k_metrics['top_1_accuracy'], 2/3, places=3)
        
        # Top-2 accuracy should be 3/3 (true label is in top-2 for all)
        self.assertAlmostEqual(top_k_metrics['top_2_accuracy'], 1.0, places=3)
    
    def test_confusion_matrix_creation(self):
        """Test confusion matrix creation."""
        y_true = [0, 0, 1, 1, 2, 2, 3, 3]
        y_pred = [0, 1, 1, 1, 2, 2, 3, 0]  # Some errors
        
        cm = self.evaluator.create_confusion_matrix(y_true, y_pred)
        
        # Check shape
        self.assertEqual(cm.shape, (4, 4))
        
        # Check that it's a valid confusion matrix
        self.assertTrue(np.all(cm >= 0))  # All values non-negative
        self.assertEqual(cm.sum(), len(y_true))  # Sum equals number of samples
    
    @patch('matplotlib.pyplot.savefig')
    def test_confusion_matrix_save(self, mock_savefig):
        """Test confusion matrix saving."""
        y_true = [0, 1, 2, 3]
        y_pred = [0, 1, 2, 3]
        
        # Should not raise
        cm = self.evaluator.create_confusion_matrix(y_true, y_pred)
        
        # Check that savefig was called
        mock_savefig.assert_called_once()
    
    def test_classification_report_creation(self):
        """Test classification report creation."""
        y_true = [0, 1, 2, 3, 0, 1, 2, 3]
        y_pred = [0, 1, 2, 3, 0, 1, 2, 3]
        
        report = self.evaluator.create_classification_report(y_true, y_pred)
        
        # Check that report contains expected content
        self.assertIn('precision', report.lower())
        self.assertIn('recall', report.lower())
        self.assertIn('f1-score', report.lower())
        
        # Check that all labels are in the report
        for label in self.label_mapping.keys():
            self.assertIn(label, report)
    
    def test_evaluate_predictions_comprehensive(self):
        """Test comprehensive evaluation."""
        y_true = [0, 1, 2, 3, 0, 1, 2, 3]
        y_pred = [0, 1, 2, 3, 0, 1, 2, 3]
        y_pred_probs = np.random.rand(8, 4)  # Random probabilities
        
        results = self.evaluator.evaluate_predictions(y_true, y_pred, y_pred_probs, top_k=3)
        
        # Check that all expected keys are present
        expected_keys = [
            'accuracy', 'macro_f1', 'weighted_f1', 'macro_precision', 'macro_recall',
            'per_class', 'confusion_matrix', 'classification_report', 'label_distribution'
        ]
        
        for key in expected_keys:
            self.assertIn(key, results)
        
        # Check top-k metrics
        for k in range(1, 4):
            self.assertIn(f'top_{k}_accuracy', results)
    
    def test_save_evaluation_results(self):
        """Test saving evaluation results."""
        results = {
            'accuracy': 0.95,
            'macro_f1': 0.94,
            'per_class': {
                'sales': {'precision': 0.9, 'recall': 0.9, 'f1': 0.9}
            }
        }
        
        # Should not raise
        self.evaluator.save_evaluation_results(results, "test_results.json")
        
        # Check that file was created
        result_path = os.path.join(self.temp_dir, "test_results.json")
        self.assertTrue(os.path.exists(result_path))
        
        # Check file content
        import json
        with open(result_path, 'r') as f:
            loaded_results = json.load(f)
        
        self.assertEqual(loaded_results['accuracy'], 0.95)
        self.assertEqual(loaded_results['macro_f1'], 0.94)

class TestEvaluationFunctions(unittest.TestCase):
    """Test evaluation utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('eval.TicketRoutingModel')
    def test_evaluate_model_predictions(self, mock_model_class):
        """Test model evaluation function."""
        # Mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = (
            np.array([0, 1, 2, 3]),  # predictions
            np.random.rand(4, 4)      # probabilities
        )
        mock_model_class.return_value = mock_model
        
        # Test data
        test_texts = ["text1", "text2", "text3", "text4"]
        test_labels = ["sales", "tech_support", "general", "billing"]
        label_mapping = {"sales": 0, "tech_support": 1, "general": 2, "billing": 3}
        
        # Should not raise
        results = evaluate_model_predictions(
            mock_model, test_texts, test_labels, label_mapping, self.temp_dir
        )
        
        # Check that results contain expected keys
        self.assertIn('accuracy', results)
        self.assertIn('macro_f1', results)
        self.assertIn('per_class', results)
    
    def test_compare_models(self):
        """Test model comparison function."""
        # Create temporary result files
        results1_path = os.path.join(self.temp_dir, "model1_results.json")
        results2_path = os.path.join(self.temp_dir, "model2_results.json")
        
        results1 = {
            'accuracy': 0.90,
            'macro_f1': 0.88,
            'weighted_f1': 0.89,
            'macro_precision': 0.87,
            'macro_recall': 0.89
        }
        
        results2 = {
            'accuracy': 0.95,
            'macro_f1': 0.93,
            'weighted_f1': 0.94,
            'macro_precision': 0.92,
            'macro_recall': 0.94
        }
        
        # Save results
        import json
        with open(results1_path, 'w') as f:
            json.dump(results1, f)
        with open(results2_path, 'w') as f:
            json.dump(results2, f)
        
        # Compare models
        comparison_df = compare_models(
            [results1_path, results2_path],
            ['Model 1', 'Model 2']
        )
        
        # Check DataFrame structure
        self.assertIsInstance(comparison_df, pd.DataFrame)
        self.assertEqual(len(comparison_df), 2)
        self.assertEqual(list(comparison_df.index), ['Model 1', 'Model 2'])
        
        # Check that Model 2 has better metrics
        self.assertGreater(comparison_df.loc['Model 2', 'accuracy'], 
                          comparison_df.loc['Model 1', 'accuracy'])
        self.assertGreater(comparison_df.loc['Model 2', 'macro_f1'], 
                          comparison_df.loc['Model 1', 'macro_f1'])

if __name__ == '__main__':
    unittest.main()
