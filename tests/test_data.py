"""
Test suite for ticket routing data utilities.
Minimal smoke tests for data loading and validation.
"""

import unittest
import tempfile
import json
import os
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Add src to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data import load_synthetic, load_ag_news, validate_dataframe
from preprocess import preprocess_text, split_data, create_label_mapping
from eval import TicketEvaluator

class TestDataLoading(unittest.TestCase):
    """Test data loading utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_path = os.path.join(self.temp_dir, "test_tickets.jsonl")
        
        # Create test synthetic data
        test_tickets = [
            {
                "id": "TKT-001",
                "subject": "Login issues",
                "body": "I cannot log into my account",
                "label": "tech_support",
                "created_at": "2024-01-01T10:00:00",
                "meta": {"priority": "high"}
            },
            {
                "id": "TKT-002", 
                "subject": "Pricing inquiry",
                "body": "What are your pricing plans?",
                "label": "sales",
                "created_at": "2024-01-01T11:00:00",
                "meta": {"priority": "medium"}
            }
        ]
        
        with open(self.test_data_path, 'w') as f:
            for ticket in test_tickets:
                f.write(json.dumps(ticket) + '\n')
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_synthetic(self):
        """Test loading synthetic ticket data."""
        df = load_synthetic(self.test_data_path)
        
        # Check DataFrame structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        
        # Check required columns
        required_columns = ['id', 'text', 'label', 'created_at', 'meta']
        for col in required_columns:
            self.assertIn(col, df.columns)
        
        # Check text combination
        self.assertIn("Login issues I cannot log into my account", df['text'].values)
        self.assertIn("Pricing inquiry What are your pricing plans?", df['text'].values)
        
        # Check labels
        self.assertIn("tech_support", df['label'].values)
        self.assertIn("sales", df['label'].values)
    
    def test_load_synthetic_file_not_found(self):
        """Test handling of missing synthetic data file."""
        with self.assertRaises(FileNotFoundError):
            load_synthetic("nonexistent_file.jsonl")
    
    @patch('data.load_dataset')
    def test_load_ag_news(self, mock_load_dataset):
        """Test loading AG News dataset."""
        # Mock the dataset
        mock_dataset = MagicMock()
        mock_dataset.select.return_value = mock_dataset
        mock_dataset.__len__ = lambda x: 4
        
        # Mock dataset data
        mock_dataset.__iter__ = lambda x: iter([
            {"text": "World news article", "label": 0},
            {"text": "Sports news article", "label": 1},
            {"text": "Business news article", "label": 2},
            {"text": "Tech news article", "label": 3}
        ])
        
        mock_load_dataset.return_value = mock_dataset
        
        df = load_ag_news(limit=4)
        
        # Check DataFrame structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 4)
        
        # Check required columns
        required_columns = ['id', 'text', 'label', 'created_at', 'meta']
        for col in required_columns:
            self.assertIn(col, df.columns)
        
        # Check label mapping
        expected_labels = ["sales", "tech_support", "general", "billing"]
        for label in expected_labels:
            self.assertIn(label, df['label'].values)
    
    def test_validate_dataframe(self):
        """Test DataFrame validation."""
        # Valid DataFrame
        valid_df = pd.DataFrame({
            'id': ['1', '2'],
            'text': ['Test text 1', 'Test text 2'],
            'label': ['sales', 'tech_support'],
            'created_at': ['2024-01-01', '2024-01-02'],
            'meta': [{}, {}]
        })
        
        # Should not raise
        self.assertTrue(validate_dataframe(valid_df))
        
        # Invalid DataFrame - missing column
        invalid_df = valid_df.drop('label', axis=1)
        with self.assertRaises(ValueError):
            validate_dataframe(invalid_df)
        
        # Invalid DataFrame - empty text
        invalid_df = valid_df.copy()
        invalid_df.loc[0, 'text'] = ''
        with self.assertRaises(ValueError):
            validate_dataframe(invalid_df)
        
        # Invalid DataFrame - invalid label
        invalid_df = valid_df.copy()
        invalid_df.loc[0, 'label'] = 'invalid_label'
        with self.assertRaises(ValueError):
            validate_dataframe(invalid_df)

class TestPreprocessing(unittest.TestCase):
    """Test preprocessing utilities."""
    
    def test_preprocess_text(self):
        """Test text preprocessing pipeline."""
        # Test HTML stripping
        html_text = "<p>Hello <b>world</b></p>"
        processed = preprocess_text(html_text, strip_html_tags=True)
        self.assertEqual(processed, "Hello world")
        
        # Test whitespace collapse
        messy_text = "Hello    world\n\n\nTest"
        processed = preprocess_text(messy_text, strip_html_tags=False)
        self.assertEqual(processed, "Hello world Test")
        
        # Test PII redaction
        pii_text = "Contact me at john@example.com or call 555-123-4567"
        processed = preprocess_text(pii_text, redact_pii_data=True)
        self.assertIn("[EMAIL_REDACTED]", processed)
        self.assertIn("[PHONE_REDACTED]", processed)
        
        # Test lowercase
        text = "Hello WORLD"
        processed = preprocess_text(text, lowercase=True)
        self.assertEqual(processed, "hello world")
    
    def test_split_data(self):
        """Test data splitting functionality."""
        # Create test DataFrame
        df = pd.DataFrame({
            'id': range(100),
            'text': [f'Text {i}' for i in range(100)],
            'label': ['sales'] * 25 + ['tech_support'] * 25 + ['general'] * 25 + ['billing'] * 25,
            'created_at': ['2024-01-01'] * 100,
            'meta': [{}] * 100
        })
        
        train_df, val_df, test_df = split_data(df, test_size=0.2, val_size=0.2)
        
        # Check split sizes
        self.assertEqual(len(train_df), 60)  # 100 * 0.6
        self.assertEqual(len(val_df), 20)    # 100 * 0.2
        self.assertEqual(len(test_df), 20)   # 100 * 0.2
        
        # Check stratification (each split should have all labels)
        for split_df in [train_df, val_df, test_df]:
            unique_labels = set(split_df['label'].unique())
            self.assertEqual(len(unique_labels), 4)  # All 4 labels present
    
    def test_create_label_mapping(self):
        """Test label mapping creation."""
        df = pd.DataFrame({
            'label': ['sales', 'tech_support', 'general', 'billing']
        })
        
        mapping = create_label_mapping(df)
        
        expected_mapping = {
            'billing': 0,
            'general': 1, 
            'sales': 2,
            'tech_support': 3
        }
        
        self.assertEqual(mapping, expected_mapping)

class TestEvaluation(unittest.TestCase):
    """Test evaluation utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.label_mapping = {
            'sales': 0,
            'tech_support': 1,
            'general': 2,
            'billing': 3
        }
        
        self.evaluator = TicketEvaluator(self.label_mapping)
    
    def test_compute_metrics(self):
        """Test metrics computation."""
        y_true = [0, 1, 2, 3, 0, 1]
        y_pred = [0, 1, 2, 3, 1, 0]  # One error
        
        metrics = self.evaluator.compute_metrics(y_true, y_pred)
        
        # Check metric structure
        self.assertIn('accuracy', metrics)
        self.assertIn('macro_f1', metrics)
        self.assertIn('weighted_f1', metrics)
        self.assertIn('macro_precision', metrics)
        self.assertIn('macro_recall', metrics)
        self.assertIn('per_class', metrics)
        
        # Check per-class metrics
        self.assertEqual(len(metrics['per_class']), 4)
        for label in self.label_mapping.keys():
            self.assertIn(label, metrics['per_class'])
            self.assertIn('precision', metrics['per_class'][label])
            self.assertIn('recall', metrics['per_class'][label])
            self.assertIn('f1', metrics['per_class'][label])
    
    def test_compute_top_k_accuracy(self):
        """Test top-k accuracy computation."""
        y_true = [0, 1, 2]
        y_pred_probs = np.array([
            [0.7, 0.2, 0.1, 0.0],  # Correct prediction
            [0.3, 0.4, 0.2, 0.1],  # Correct prediction
            [0.1, 0.2, 0.3, 0.4]   # Wrong prediction
        ])
        
        top_k_metrics = self.evaluator.compute_top_k_accuracy(y_true, y_pred_probs, k=2)
        
        self.assertIn('top_1_accuracy', top_k_metrics)
        self.assertIn('top_2_accuracy', top_k_metrics)
        
        # Top-1 accuracy should be 2/3 = 0.667
        self.assertAlmostEqual(top_k_metrics['top_1_accuracy'], 2/3, places=3)
        
        # Top-2 accuracy should be 3/3 = 1.0
        self.assertAlmostEqual(top_k_metrics['top_2_accuracy'], 1.0, places=3)
    
    def test_confusion_matrix_shape(self):
        """Test confusion matrix creation."""
        y_true = [0, 1, 2, 3, 0, 1, 2, 3]
        y_pred = [0, 1, 2, 3, 0, 1, 2, 3]
        
        cm = self.evaluator.create_confusion_matrix(y_true, y_pred)
        
        # Check shape
        self.assertEqual(cm.shape, (4, 4))  # 4 classes
        
        # Check diagonal (perfect predictions)
        np.testing.assert_array_equal(np.diag(cm), [2, 2, 2, 2])

if __name__ == '__main__':
    unittest.main()
