"""
Text preprocessing utilities for ticket routing.
Includes PII redaction, HTML stripping, and train/val/test splitting.
"""

import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# PII patterns for redaction
EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
PHONE_PATTERN = r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'
SSN_PATTERN = r'\b\d{3}-?\d{2}-?\d{4}\b'
CREDIT_CARD_PATTERN = r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'

def strip_html(text: str) -> str:
    """Remove HTML tags from text."""
    if not isinstance(text, str):
        return text
    return re.sub(r'<[^>]+>', '', text)

def collapse_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into single spaces."""
    if not isinstance(text, str):
        return text
    return re.sub(r'\s+', ' ', text).strip()

def redact_pii(text: str, redaction_map: Optional[Dict[str, str]] = None) -> str:
    """
    Redact PII from text using regex patterns.
    
    Args:
        text: Input text to redact
        redaction_map: Custom redaction patterns {pattern: replacement}
        
    Returns:
        Text with PII redacted
    """
    if not isinstance(text, str):
        return text
    
    # Default redaction patterns
    default_patterns = {
        EMAIL_PATTERN: '[EMAIL_REDACTED]',
        PHONE_PATTERN: '[PHONE_REDACTED]',
        SSN_PATTERN: '[SSN_REDACTED]',
        CREDIT_CARD_PATTERN: '[CARD_REDACTED]'
    }
    
    # Merge with custom patterns
    patterns = {**default_patterns}
    if redaction_map:
        patterns.update(redaction_map)
    
    # Apply redaction
    redacted_text = text
    for pattern, replacement in patterns.items():
        redacted_text = re.sub(pattern, replacement, redacted_text, flags=re.IGNORECASE)
    
    return redacted_text

def preprocess_text(text: str, 
                   lowercase: bool = False,
                   strip_html_tags: bool = True,
                   redact_pii_data: bool = True) -> str:
    """
    Comprehensive text preprocessing pipeline.
    
    Args:
        text: Input text to preprocess
        lowercase: Whether to convert to lowercase
        strip_html_tags: Whether to remove HTML tags
        redact_pii_data: Whether to redact PII
        
    Returns:
        Preprocessed text
    """
    if not isinstance(text, str):
        return text
    
    # Strip HTML tags
    if strip_html_tags:
        text = strip_html(text)
    
    # Collapse whitespace
    text = collapse_whitespace(text)
    
    # Redact PII
    if redact_pii_data:
        text = redact_pii(text)
    
    # Convert to lowercase
    if lowercase:
        text = text.lower()
    
    return text

def split_data(df: pd.DataFrame, 
               test_size: float = 0.1,
               val_size: float = 0.1,
               stratify_column: str = 'label',
               date_column: Optional[str] = None,
               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/validation/test sets.
    
    Args:
        df: Input DataFrame
        test_size: Fraction for test set
        val_size: Fraction for validation set (from remaining data)
        stratify_column: Column to stratify on
        date_column: Optional date column for time-based splitting
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info(f"Splitting {len(df)} examples into train/val/test sets")
    
    if date_column and date_column in df.columns:
        # Time-based splitting (chronological)
        logger.info("Using chronological splitting based on date column")
        df_sorted = df.sort_values(date_column)
        
        n_total = len(df_sorted)
        n_test = int(n_total * test_size)
        n_val = int(n_total * val_size)
        n_train = n_total - n_test - n_val
        
        train_df = df_sorted.iloc[:n_train]
        val_df = df_sorted.iloc[n_train:n_train + n_val]
        test_df = df_sorted.iloc[n_train + n_val:]
        
    else:
        # Stratified random splitting
        logger.info("Using stratified random splitting")
        
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df, 
            test_size=test_size,
            stratify=df[stratify_column],
            random_state=random_state
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            stratify=train_val_df[stratify_column],
            random_state=random_state
        )
    
    # Log split statistics
    logger.info(f"Train: {len(train_df)} examples")
    logger.info(f"Validation: {len(val_df)} examples")
    logger.info(f"Test: {len(test_df)} examples")
    
    # Log label distribution
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        label_counts = split_df[stratify_column].value_counts()
        logger.info(f"{split_name} label distribution: {label_counts.to_dict()}")
    
    return train_df, val_df, test_df

def create_label_mapping(df: pd.DataFrame, label_column: str = 'label') -> Dict[str, int]:
    """
    Create mapping from label names to integers.
    
    Args:
        df: DataFrame containing labels
        label_column: Name of label column
        
    Returns:
        Dictionary mapping label names to integers
    """
    unique_labels = sorted(df[label_column].unique())
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    
    logger.info(f"Label mapping: {label_to_id}")
    return label_to_id

def preprocess_dataset(df: pd.DataFrame,
                      text_column: str = 'text',
                      label_column: str = 'label',
                      lowercase: bool = False,
                      strip_html_tags: bool = True,
                      redact_pii_data: bool = True) -> pd.DataFrame:
    """
    Apply preprocessing to entire dataset.
    
    Args:
        df: Input DataFrame
        text_column: Name of text column to preprocess
        label_column: Name of label column
        lowercase: Whether to convert text to lowercase
        strip_html_tags: Whether to remove HTML tags
        redact_pii_data: Whether to redact PII
        
    Returns:
        Preprocessed DataFrame
    """
    logger.info(f"Preprocessing {len(df)} examples")
    
    # Create a copy to avoid modifying original
    df_processed = df.copy()
    
    # Apply text preprocessing
    df_processed[text_column] = df_processed[text_column].apply(
        lambda x: preprocess_text(
            x, 
            lowercase=lowercase,
            strip_html_tags=strip_html_tags,
            redact_pii_data=redact_pii_data
        )
    )
    
    # Remove any empty texts after preprocessing
    initial_count = len(df_processed)
    df_processed = df_processed[df_processed[text_column].str.strip() != '']
    final_count = len(df_processed)
    
    if initial_count != final_count:
        logger.warning(f"Removed {initial_count - final_count} empty texts after preprocessing")
    
    logger.info(f"Preprocessing complete: {final_count} examples remaining")
    return df_processed
