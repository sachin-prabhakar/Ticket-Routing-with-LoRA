"""
Data loading utilities for ticket routing.
Supports synthetic tickets and AG News dataset.
"""

import json
import pandas as pd
from datasets import load_dataset
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Queue mapping for AG News labels
AG_NEWS_QUEUE_MAP = {
    0: "sales",        # World
    1: "tech_support", # Sports  
    2: "general",      # Business
    3: "billing"       # Sci/Tech
}

def load_synthetic(path: str = "data/synthetic_tickets.jsonl") -> pd.DataFrame:
    """
    Load synthetic ticket data from JSONL file.
    
    Args:
        path: Path to synthetic tickets JSONL file
        
    Returns:
        DataFrame with columns: id, text, label, created_at, meta
    """
    try:
        tickets = []
        with open(path, 'r') as f:
            for line in f:
                ticket = json.loads(line.strip())
                # Combine subject and body into text
                text = f"{ticket['subject']} {ticket['body']}"
                tickets.append({
                    'id': ticket['id'],
                    'text': text,
                    'label': ticket['label'],
                    'created_at': ticket['created_at'],
                    'meta': ticket.get('meta', {})
                })
        
        df = pd.DataFrame(tickets)
        logger.info(f"Loaded {len(df)} synthetic tickets from {path}")
        return df
        
    except FileNotFoundError:
        logger.error(f"Synthetic tickets file not found: {path}")
        logger.info("Run 'python scripts/make_synth.py' to generate synthetic data")
        raise
    except Exception as e:
        logger.error(f"Error loading synthetic tickets: {e}")
        raise

def load_ag_news(limit: int = 2000) -> pd.DataFrame:
    """
    Load AG News dataset and map to ticket queues.
    
    Args:
        limit: Maximum number of examples to load
        
    Returns:
        DataFrame with columns: id, text, label, created_at, meta
    """
    try:
        # Load AG News dataset
        dataset = load_dataset("ag_news", split="train")
        
        # Limit dataset size for quick runs
        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))
        
        # Convert to DataFrame
        df = pd.DataFrame(dataset)
        
        # Map labels to queue names
        df['label'] = df['label'].map(AG_NEWS_QUEUE_MAP)
        
        # Rename columns to match expected format
        df = df.rename(columns={
            'text': 'text',
            'label': 'label'
        })
        
        # Add required columns
        df['id'] = range(len(df))
        df['created_at'] = pd.Timestamp.now().isoformat()
        df['meta'] = [{'source': 'ag_news', 'original_label': int(label)} 
                     for label in dataset['label']]
        
        # Reorder columns
        df = df[['id', 'text', 'label', 'created_at', 'meta']]
        
        logger.info(f"Loaded {len(df)} AG News examples")
        logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading AG News dataset: {e}")
        raise

def get_data_loader(data_source: str = "synthetic", **kwargs) -> pd.DataFrame:
    """
    Get data loader based on source type.
    
    Args:
        data_source: Either "synthetic" or "ag_news"
        **kwargs: Additional arguments passed to loader
        
    Returns:
        DataFrame with ticket data
    """
    if data_source == "synthetic":
        return load_synthetic(**kwargs)
    elif data_source == "ag_news":
        return load_ag_news(**kwargs)
    else:
        raise ValueError(f"Unknown data source: {data_source}")

def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate that DataFrame has required columns and data types.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if valid, raises ValueError if not
    """
    required_columns = ['id', 'text', 'label', 'created_at', 'meta']
    
    # Check required columns exist
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for empty text
    if df['text'].isna().any() or (df['text'] == '').any():
        raise ValueError("Found empty or null text fields")
    
    # Check label values
    valid_labels = {"sales", "tech_support", "general", "billing"}
    invalid_labels = set(df['label'].unique()) - valid_labels
    if invalid_labels:
        raise ValueError(f"Invalid labels found: {invalid_labels}")
    
    logger.info(f"DataFrame validation passed: {len(df)} rows, {len(df.columns)} columns")
    return True
