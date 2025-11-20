"""
TF-IDF vectorization utilities for preprocessed text data.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import logging
from dataclasses import dataclass
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TfidfConfig:
    """Configuration for TF-IDF vectorization."""
    max_features: int = 10000
    min_df: Union[int, float] = 2
    max_df: Union[float] = 0.95
    ngram_range: Tuple[int, int] = (1, 2)
    use_idf: bool = True
    smooth_idf: bool = True
    sublinear_tf: bool = True
    norm: str = 'l2'
    lowercase: bool = True
    strip_accents: str = 'unicode'
    stop_words: Optional[Union[str, List[str]]] = None
    vocabulary: Optional[Dict[str, int]] = None
    binary: bool = False
    use_hashing: bool = False
    n_features: int = 2**20  # For hashing vectorizer


class TfidfVectorizerWrapper:
    """Enhanced TF-IDF vectorizer with additional features."""
    
    def __init__(self, config: TfidfConfig = None):
        self.config = config or TfidfConfig()
        self.vectorizer = None
        self.feature_names = []
        self.tfidf_matrix = None
        self.vocabulary = None
        self.idf_scores = None
        
    def _prepare_text_data(self, df: pd.DataFrame, text_columns: List[str]) -> List[str]:
        """
        Prepare text data for vectorization.
        
        Args:
            df: Input DataFrame
            text_columns: List of column names containing text data
            
        Returns:
            List of processed text strings
        """
        processed_texts = []
        
        for _, row in df.iterrows():
            text_parts = []
            
            for col in text_columns:
                if col in df.columns:
                    try:
                        cell_value = row[col]
                        if pd.notna(cell_value):
                            text = str(cell_value)
                            
                            # If it's a list of tokens, join them
                            if isinstance(cell_value, list):
                                text = ' '.join(str(token) for token in cell_value)
                            
                            if text.strip():
                                text_parts.append(text)
                    except (ValueError, TypeError):
                        # Skip problematic cells
                        continue
            
            # Combine all text parts
            combined_text = ' '.join(text_parts)
            processed_texts.append(combined_text)
        
        return processed_texts
    
    def fit_transform(self, df: pd.DataFrame, text_columns: List[str]) -> Tuple[np.ndarray, 'TfidfVectorizerWrapper']:
        """
        Fit TF-IDF vectorizer and transform data.
        
        Args:
            df: Input DataFrame
            text_columns: List of column names containing text data
            
        Returns:
            Tuple of (TF-IDF matrix, fitted vectorizer)
        """
        logger.info("Preparing text data for TF-IDF vectorization")
        texts = self._prepare_text_data(df, text_columns)
        
        logger.info(f"Vectorizing {len(texts)} documents")
        
        # Initialize vectorizer
        if self.config.use_hashing:
            from sklearn.feature_extraction.text import HashingVectorizer
            self.vectorizer = HashingVectorizer(
                n_features=self.config.n_features,
                ngram_range=self.config.ngram_range,
                stop_words=self.config.stop_words,
                lowercase=self.config.lowercase,
                norm=self.config.norm,
                binary=self.config.binary
            )
        else:
            self.vectorizer = TfidfVectorizer(
                max_features=self.config.max_features,
                min_df=self.config.min_df,
                max_df=self.config.max_df,
                ngram_range=self.config.ngram_range,
                use_idf=self.config.use_idf,
                smooth_idf=self.config.smooth_idf,
                sublinear_tf=self.config.sublinear_tf,
                norm=self.config.norm,
                lowercase=self.config.lowercase,
                strip_accents=self.config.strip_accents,
                stop_words=self.config.stop_words,
                vocabulary=self.config.vocabulary,
                binary=self.config.binary
            )
        
        # Fit and transform
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # Store feature information
        if not self.config.use_hashing:
            self.feature_names = self.vectorizer.get_feature_names_out().tolist()
            self.vocabulary = self.vectorizer.vocabulary_
            self.idf_scores = self.vectorizer.idf_
        
        logger.info(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        logger.info(f"Sparsity: {1 - (self.tfidf_matrix.nnz / self.tfidf_matrix.size):.1%}")
        
        return self.tfidf_matrix, self
    
    def transform(self, df: pd.DataFrame, text_columns: List[str]) -> np.ndarray:
        """
        Transform new data using fitted vectorizer.
        
        Args:
            df: Input DataFrame
            text_columns: List of column names containing text data
            
        Returns:
            TF-IDF matrix
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer must be fitted before transforming")
        
        texts = self._prepare_text_data(df, text_columns)
        return self.vectorizer.transform(texts)
    
    def get_feature_importance(self, top_n: int = 50) -> List[Tuple[str, float]]:
        """
        Get most important features by average TF-IDF score.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            List of (feature_name, average_score) tuples
        """
        if self.tfidf_matrix is None or not self.feature_names:
            return []
        
        # Calculate average TF-IDF scores
        avg_scores = np.array(self.tfidf_matrix.mean(axis=0)).flatten()
        
        # Get top features
        top_indices = np.argsort(avg_scores)[-top_n:][::-1]
        
        return [(self.feature_names[i], avg_scores[i]) for i in top_indices]
    
    def get_document_similarity(self, doc_idx: int, top_n: int = 10) -> List[Tuple[int, float]]:
        """
        Get most similar documents to a given document.
        
        Args:
            doc_idx: Index of the document
            top_n: Number of similar documents to return
            
        Returns:
            List of (document_index, similarity_score) tuples
        """
        if self.tfidf_matrix is None:
            return []
        
        # Calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        
        doc_vector = self.tfidf_matrix[doc_idx:doc_idx+1]
        similarities = cosine_similarity(doc_vector, self.tfidf_matrix).flatten()
        
        # Get top similar documents (excluding self)
        similar_indices = np.argsort(similarities)[-top_n-1:-1][::-1]
        
        return [(idx, similarities[idx]) for idx in similar_indices]
    
    def reduce_dimensions(self, n_components: int = 100, 
                         method: str = 'svd') -> np.ndarray:
        """
        Reduce dimensionality of TF-IDF matrix.
        
        Args:
            n_components: Number of components to keep
            method: Dimensionality reduction method ('svd', 'nmf')
            
        Returns:
            Reduced matrix
        """
        if self.tfidf_matrix is None:
            raise ValueError("TF-IDF matrix must be computed first")
        
        logger.info(f"Reducing dimensions to {n_components} components using {method}")
        
        if method == 'svd':
            reducer = TruncatedSVD(n_components=n_components, random_state=42)
        elif method == 'nmf':
            from sklearn.decomposition import NMF
            reducer = NMF(n_components=n_components, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        reduced_matrix = reducer.fit_transform(self.tfidf_matrix)
        
        logger.info(f"Explained variance ratio: {reducer.explained_variance_ratio_.sum():.3f}")
        
        return reduced_matrix
    
    def save_vectorizer(self, filepath: str):
        """
        Save fitted vectorizer to file.
        
        Args:
            filepath: Path to save vectorizer
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer must be fitted before saving")
        
        vectorizer_data = {
            'vectorizer': self.vectorizer,
            'feature_names': self.feature_names,
            'vocabulary': self.vocabulary,
            'idf_scores': self.idf_scores,
            'config': self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(vectorizer_data, f)
        
        logger.info(f"Vectorizer saved to {filepath}")
    
    def load_vectorizer(self, filepath: str):
        """
        Load fitted vectorizer from file.
        
        Args:
            filepath: Path to load vectorizer from
        """
        with open(filepath, 'rb') as f:
            vectorizer_data = pickle.load(f)
        
        self.vectorizer = vectorizer_data['vectorizer']
        self.feature_names = vectorizer_data['feature_names']
        self.vocabulary = vectorizer_data['vocabulary']
        self.idf_scores = vectorizer_data['idf_scores']
        self.config = vectorizer_data['config']
        
        logger.info(f"Vectorizer loaded from {filepath}")


def vectorize_programs_data(df: pd.DataFrame, 
                           text_columns: List[str] = None,
                           config: TfidfConfig = None) -> Tuple[np.ndarray, TfidfVectorizerWrapper]:
    """
    Vectorize program data using TF-IDF.
    
    Args:
        df: DataFrame containing program data
        text_columns: List of text column names to vectorize
        config: TF-IDF configuration
        
    Returns:
        Tuple of (TF-IDF matrix, fitted vectorizer)
    """
    if text_columns is None:
        # Auto-detect text columns
        text_columns = []
        for col in df.columns:
            if (col.endswith('_text') or col.endswith('_raw') or 
                col.endswith('_tokens') or col.endswith('_lemmatized') or
                col.endswith('_filtered')):
                text_columns.append(col)
    
    if not text_columns:
        raise ValueError("No text columns found for vectorization")
    
    logger.info(f"Vectorizing columns: {text_columns}")
    
    # Default configuration for programs data
    if config is None:
        config = TfidfConfig(
            max_features=15000,  # Larger vocabulary for domain-specific terms
            min_df=2,  # Require at least 2 documents
            max_df=0.8,  # Remove very common terms
            ngram_range=(1, 3),  # Include trigrams for technical terms
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True,
            norm='l2',
            lowercase=True,
            strip_accents='unicode'
        )
    
    vectorizer = TfidfVectorizerWrapper(config)
    tfidf_matrix = vectorizer.fit_transform(df, text_columns)
    
    return tfidf_matrix, vectorizer


def create_tfidf_features(df: pd.DataFrame, 
                         tfidf_matrix: np.ndarray,
                         vectorizer: TfidfVectorizerWrapper,
                         feature_prefix: str = 'tfidf') -> pd.DataFrame:
    """
    Create DataFrame with TF-IDF features.
    
    Args:
        df: Original DataFrame
        tfidf_matrix: TF-IDF matrix
        vectorizer: Fitted vectorizer
        feature_prefix: Prefix for feature column names
        
    Returns:
        DataFrame with TF-IDF features
    """
    # Create feature names
    if vectorizer.feature_names:
        feature_names = [f"{feature_prefix}_{name}" for name in vectorizer.feature_names]
    else:
        feature_names = [f"{feature_prefix}_{i}" for i in range(tfidf_matrix.shape[1])]
    
    # Convert to DataFrame
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=feature_names,
        index=df.index
    )
    
    # Combine with original data
    result_df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
    
    return result_df


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create TF-IDF vectors from preprocessed data")
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--output", required=True, help="Output parquet file")
    parser.add_argument("--vectorizer-output", help="Output vectorizer pickle file")
    parser.add_argument("--text-columns", nargs='+', 
                       help="Text columns to vectorize")
    parser.add_argument("--max-features", type=int, default=10000,
                       help="Maximum number of features")
    parser.add_argument("--min-df", type=int, default=2,
                       help="Minimum document frequency")
    parser.add_argument("--max-df", type=float, default=0.95,
                       help="Maximum document frequency")
    parser.add_argument("--ngram-range", nargs=2, type=int, default=[1, 2],
                       help="N-gram range (min max)")
    parser.add_argument("--reduce-dimensions", type=int,
                       help="Reduce dimensions to specified number")
    parser.add_argument("--use-hashing", action="store_true",
                       help="Use hashing vectorizer")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Load data
    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df)} records")
    
    # Configure TF-IDF
    config = TfidfConfig(
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        ngram_range=tuple(args.ngram_range),
        use_hashing=args.use_hashing
    )
    
    # Vectorize data
    tfidf_matrix, vectorizer = vectorize_programs_data(df, args.text_columns, config)
    
    # Reduce dimensions if requested
    if args.reduce_dimensions:
        tfidf_matrix = vectorizer.reduce_dimensions(args.reduce_dimensions)
    
    # Create features DataFrame
    features_df = create_tfidf_features(df, tfidf_matrix, vectorizer)
    
    # Save results
    features_df.to_parquet(args.output, index=False)
    logger.info(f"TF-IDF features saved to {args.output}")
    
    # Save vectorizer if requested
    if args.vectorizer_output:
        vectorizer.save_vectorizer(args.vectorizer_output)
    
    # Print statistics
    logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    logger.info(f"Features created: {len(features_df.columns) - len(df.columns)}")
    
    # Print top features
    top_features = vectorizer.get_feature_importance(20)
    if top_features:
        logger.info("Top 20 features by average TF-IDF score:")
        for feature, score in top_features:
            logger.info(f"  {feature}: {score:.4f}")


if __name__ == "__main__":
    main()
