"""
Utility functions for the job posting classification system.

This module provides common utility functions used across the project,
including data loading, preprocessing, and evaluation helpers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from gensim.models import Word2Vec, Doc2Vec
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load data from various file formats.
    
    Args:
        file_path: Path to the data file (supports .parquet, .csv, .json)
        
    Returns:
        Loaded DataFrame
        
    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    suffix = file_path.suffix.lower()
    
    if suffix == ".parquet":
        return pd.read_parquet(file_path)
    elif suffix == ".csv":
        df = pd.read_csv(file_path)
        # Preserve object dtype for id column if it exists
        if 'id' in df.columns:
            df['id'] = df['id'].astype(str)
        return df
    elif suffix == ".json":
        df = pd.read_json(file_path)
        # Preserve object dtype for id column if it exists
        if 'id' in df.columns:
            df['id'] = df['id'].astype(str)
        return df
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def save_data(df: pd.DataFrame, file_path: Union[str, Path]) -> None:
    """Save DataFrame to various file formats.
    
    Args:
        df: DataFrame to save
        file_path: Path to save the file (supports .parquet, .csv, .json)
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    suffix = file_path.suffix.lower()
    
    if suffix == ".parquet":
        df.to_parquet(file_path, index=False)
    elif suffix == ".csv":
        # Convert id to string before saving to preserve dtype
        df_to_save = df.copy()
        if 'id' in df_to_save.columns:
            df_to_save['id'] = df_to_save['id'].astype(str)
        df_to_save.to_csv(file_path, index=False)
    elif suffix == ".json":
        # Convert id to string before saving to preserve dtype
        df_to_save = df.copy()
        if 'id' in df_to_save.columns:
            df_to_save['id'] = df_to_save['id'].astype(str)
        df_to_save.to_json(file_path, orient="records", indent=2)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def create_train_test_split(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create stratified train/test split.
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column for stratification
        test_size: Proportion of data to use for testing
        random_state: Random state for reproducibility
        stratify: Whether to stratify the split
        
    Returns:
        Tuple of (train_df, test_df)
    """
    stratify_col = None
    if stratify:
        # Check if we have enough samples per class for stratification
        class_counts = df[target_column].value_counts()
        min_samples_per_class = int(1 / test_size) + 1  # Need at least this many per class
        if all(count >= min_samples_per_class for count in class_counts):
            stratify_col = df[target_column]
        else:
            # Not enough samples for stratification, disable it
            logger.warning(
                f"Insufficient samples for stratification. "
                f"Min required: {min_samples_per_class} per class, "
                f"but found: {class_counts.to_dict()}. Disabling stratification."
            )
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col,
    )
    
    logger.info(f"Created train/test split: {len(train_df)} train, {len(test_df)} test")
    return train_df, test_df


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    text_columns: Optional[List[str]] = None,
) -> List[str]:
    """Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        text_columns: List of text columns to check for non-empty values
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check required columns
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
    
    # Check for empty DataFrame
    if len(df) == 0:
        errors.append("DataFrame is empty")
    
    # Check text columns
    if text_columns:
        for col in text_columns:
            if col in df.columns:
                empty_count = (df[col].fillna("").str.strip() == "").sum()
                if empty_count > 0:
                    errors.append(f"Column '{col}' has {empty_count} empty values")
    
    return errors


def get_class_distribution(y: Union[np.ndarray, pd.Series]) -> Dict[str, int]:
    """Get class distribution from target labels.
    
    Args:
        y: Target labels
        
    Returns:
        Dictionary mapping class names to counts
    """
    if isinstance(y, pd.Series):
        return y.value_counts().to_dict()
    else:
        unique, counts = np.unique(y, return_counts=True)
        return dict(zip(unique, counts))


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler(),
        ],
    )


def create_output_directory(output_path: Union[str, Path]) -> Path:
    """Create output directory if it doesn't exist.
    
    Args:
        output_path: Path to create
        
    Returns:
        Path object for the created directory
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def log_model_performance(
    model_name: str,
    metrics: Dict[str, float],
    logger: Optional[logging.Logger] = None,
) -> None:
    """Log model performance metrics.
    
    Args:
        model_name: Name of the model
        metrics: Dictionary of metric names and values
        logger: Logger instance (uses module logger if None)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Model: {model_name}")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")


def format_metrics_for_display(metrics: Dict[str, float]) -> str:
    """Format metrics dictionary for display.
    
    Args:
        metrics: Dictionary of metric names and values
        
    Returns:
        Formatted string for display
    """
    lines = []
    for metric, value in metrics.items():
        lines.append(f"{metric}: {value:.4f}")
    return "\n".join(lines)


def load_word2vec_model(model_path: Union[str, Path]) -> Word2Vec:
    """Load a trained Word2Vec model.
    
    Args:
        model_path: Path to the Word2Vec model file
        
    Returns:
        Loaded Word2Vec model
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Word2Vec model not found: {model_path}")
    
    return Word2Vec.load(str(model_path))


def load_doc2vec_model(model_path: Union[str, Path]) -> Doc2Vec:
    """Load a trained Doc2Vec model.
    
    Args:
        model_path: Path to the Doc2Vec model file
        
    Returns:
        Loaded Doc2Vec model
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Doc2Vec model not found: {model_path}")
    
    return Doc2Vec.load(str(model_path))


def load_transformer_embeddings(embeddings_path: Union[str, Path]) -> np.ndarray:
    """Load pre-computed transformer embeddings.
    
    Args:
        embeddings_path: Path to the embeddings .npy file
        
    Returns:
        Numpy array of embeddings
    """
    embeddings_path = Path(embeddings_path)
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Transformer embeddings not found: {embeddings_path}")
    
    return np.load(embeddings_path)


def load_embeddings_metadata(metadata_path: Union[str, Path]) -> Dict[str, Any]:
    """Load embeddings metadata from JSON file.
    
    Args:
        metadata_path: Path to the metadata JSON file
        
    Returns:
        Dictionary containing metadata
    """
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        return json.load(f)


def get_document_embeddings_word2vec(
    model: Word2Vec, 
    documents: List[List[str]], 
    vector_size: int = 300
) -> np.ndarray:
    """Get document embeddings by averaging Word2Vec word vectors.
    
    Args:
        model: Trained Word2Vec model
        documents: List of tokenized documents
        vector_size: Size of the embedding vectors
        
    Returns:
        Numpy array of document embeddings
    """
    embeddings = []
    
    for doc in documents:
        # Get word vectors for words in vocabulary
        word_vectors = []
        for word in doc:
            if word in model.wv:
                word_vectors.append(model.wv[word])
        
        if word_vectors:
            # Average the word vectors
            doc_embedding = np.mean(word_vectors, axis=0)
        else:
            # Use zero vector if no words in vocabulary
            doc_embedding = np.zeros(vector_size)
        
        embeddings.append(doc_embedding)
    
    return np.array(embeddings)


def get_document_embeddings_doc2vec(
    model: Doc2Vec, 
    documents: List[List[str]]
) -> np.ndarray:
    """Get document embeddings using Doc2Vec model.
    
    Args:
        model: Trained Doc2Vec model
        documents: List of tokenized documents
        
    Returns:
        Numpy array of document embeddings
    """
    embeddings = []
    
    for doc in documents:
        # Infer vector for the document
        doc_embedding = model.infer_vector(doc)
        embeddings.append(doc_embedding)
    
    return np.array(embeddings)


def load_clustering_results(results_path: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load clustering results from parquet and JSON files.
    
    Args:
        results_path: Path to clustering results directory
        
    Returns:
        Tuple of (cluster_labels, metrics_dict)
    """
    results_path = Path(results_path)
    
    # Load cluster labels
    labels_file = results_path / "cluster_labels.parquet"
    if not labels_file.exists():
        raise FileNotFoundError(f"Cluster labels not found: {labels_file}")
    
    labels_df = pd.read_parquet(labels_file)
    cluster_labels = labels_df['cluster'].values
    
    # Load metrics
    metrics_file = results_path / "clustering_metrics.json"
    metrics = {}
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
    
    return cluster_labels, metrics


def load_lda_model(lda_path: Union[str, Path]) -> Tuple[Any, Any, Dict[str, Any]]:
    """Load LDA model and related files.
    
    Args:
        lda_path: Path to LDA model directory
        
    Returns:
        Tuple of (lda_model, dictionary, metadata)
    """
    lda_path = Path(lda_path)
    
    # Load LDA model
    model_file = lda_path / "lda_model.pkl"
    if not model_file.exists():
        raise FileNotFoundError(f"LDA model not found: {model_file}")
    
    import pickle
    with open(model_file, 'rb') as f:
        lda_model = pickle.load(f)
    
    # Load dictionary
    dict_file = lda_path / "dictionary.pkl"
    if not dict_file.exists():
        raise FileNotFoundError(f"Dictionary not found: {dict_file}")
    
    with open(dict_file, 'rb') as f:
        dictionary = pickle.load(f)
    
    # Load metadata
    metadata = {}
    topics_file = lda_path / "topics_top_terms.json"
    if topics_file.exists():
        with open(topics_file, 'r') as f:
            metadata['topics'] = json.load(f)
    
    return lda_model, dictionary, metadata

