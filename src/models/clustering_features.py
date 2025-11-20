"""
Clustering Feature Extraction Module.

This module provides feature extractors for different vector representations
used in clustering analysis: Doc2Vec, Word2Vec-TF-IDF weighted, and TF-IDF.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pickle
import gensim
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocess.tfidf import TfidfVectorizerWrapper, TfidfConfig

logger = logging.getLogger(__name__)


class Doc2VecFeatureExtractor:
    """Extract and standardize Doc2Vec vectors."""
    
    def __init__(self, standardize: bool = True):
        """Initialize Doc2Vec feature extractor.
        
        Args:
            standardize: Whether to standardize features
        """
        self.standardize = standardize
        self.scaler = StandardScaler() if standardize else None
        
    def extract_features(self, doc2vec_path: str) -> Tuple[np.ndarray, pd.DataFrame]:
        """Extract Doc2Vec features from parquet file.
        
        Args:
            doc2vec_path: Path to Doc2Vec vectors parquet file
            
        Returns:
            Tuple of (features array, metadata dataframe)
        """
        logger.info(f"Loading Doc2Vec features from {doc2vec_path}")
        
        df = pd.read_parquet(doc2vec_path)
        
        # Extract vector columns (vec_0, vec_1, ...)
        vector_cols = [col for col in df.columns if col.startswith('vec_')]
        features = df[vector_cols].values
        
        # Extract metadata
        metadata_cols = ['doc_id', 'doc_type', 'doc_index']
        metadata = df[metadata_cols].copy()
        
        logger.info(f"Extracted Doc2Vec features: {features.shape}")
        logger.info(f"Document types: {metadata['doc_type'].value_counts().to_dict()}")
        
        # Standardize if requested
        if self.standardize:
            features = self.scaler.fit_transform(features)
            logger.info("Doc2Vec features standardized")
            
        return features, metadata


class Word2VecTfidfFeatureExtractor:
    """Create TF-IDF-weighted Word2Vec document vectors."""
    
    def __init__(self, standardize: bool = True):
        """Initialize W2V-TF-IDF feature extractor.
        
        Args:
            standardize: Whether to standardize features
        """
        self.standardize = standardize
        self.scaler = StandardScaler() if standardize else None
        self.w2v_model = None
        self.tfidf_vectorizer = None
        
    def load_word2vec_model(self, model_path: str) -> None:
        """Load Word2Vec model.
        
        Args:
            model_path: Path to Word2Vec model file
        """
        logger.info(f"Loading Word2Vec model from {model_path}")
        self.w2v_model = Word2Vec.load(model_path)
        logger.info(f"Loaded Word2Vec model with vocabulary size: {len(self.w2v_model.wv)}")
        
    def prepare_tokenized_data(self, jobs_path: str, programs_path: str) -> Tuple[List[List[str]], List[str]]:
        """Load and prepare tokenized data from both jobs and programs.
        
        Args:
            jobs_path: Path to jobs tokenized parquet
            programs_path: Path to programs tokenized parquet
            
        Returns:
            Tuple of (tokenized_texts, doc_types)
        """
        logger.info("Loading tokenized data")
        
        # Load jobs data
        jobs_df = pd.read_parquet(jobs_path)
        jobs_tokens = jobs_df['description_text_tokens_filtered'].dropna().tolist()
        jobs_types = ['JOB'] * len(jobs_tokens)
        
        # Load programs data
        programs_df = pd.read_parquet(programs_path)
        programs_tokens = programs_df['description_text_tokens'].dropna().tolist()
        programs_types = ['PROGRAM'] * len(programs_tokens)
        
        # Combine
        all_tokens = jobs_tokens + programs_tokens
        all_types = jobs_types + programs_types
        
        logger.info(f"Loaded {len(jobs_tokens)} job tokens and {len(programs_tokens)} program tokens")
        
        return all_tokens, all_types
        
    def create_tfidf_weights(self, tokenized_texts: List[List[str]]) -> np.ndarray:
        """Create TF-IDF weights for tokens.
        
        Args:
            tokenized_texts: List of tokenized documents
            
        Returns:
            TF-IDF matrix
        """
        logger.info("Creating TF-IDF weights for tokens")
        
        # Convert tokenized texts to strings for TF-IDF
        text_strings = [' '.join(tokens) for tokens in tokenized_texts]
        
        # Create TF-IDF vectorizer
        config = TfidfConfig(
            max_features=10000,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 1),  # Only unigrams for token weighting
            lowercase=True
        )
        
        self.tfidf_vectorizer = TfidfVectorizerWrapper(config)
        tfidf_matrix, _ = self.tfidf_vectorizer.fit_transform(pd.DataFrame({'text': text_strings}), ['text'])
        
        logger.info(f"Created TF-IDF matrix: {tfidf_matrix.shape}")
        return tfidf_matrix
        
    def create_weighted_vectors(self, tokenized_texts: List[List[str]], tfidf_matrix: np.ndarray) -> np.ndarray:
        """Create TF-IDF-weighted Word2Vec document vectors.
        
        Args:
            tokenized_texts: List of tokenized documents
            tfidf_matrix: TF-IDF matrix
            
        Returns:
            Weighted document vectors
        """
        logger.info("Creating TF-IDF-weighted Word2Vec vectors")
        
        if self.w2v_model is None:
            raise ValueError("Word2Vec model must be loaded first")
            
        doc_vectors = []
        vector_dim = self.w2v_model.wv.vector_size
        
        for i, tokens in enumerate(tokenized_texts):
            # Get TF-IDF weights for this document
            doc_tfidf = tfidf_matrix[i].toarray().flatten()
            
            # Initialize document vector
            doc_vector = np.zeros(vector_dim)
            total_weight = 0
            
            # Weight each token by its TF-IDF score
            for token in tokens:
                if token in self.w2v_model.wv:
                    # Find token index in TF-IDF vocabulary
                    if hasattr(self.tfidf_vectorizer, 'feature_names'):
                        token_idx = None
                        for j, feature_name in enumerate(self.tfidf_vectorizer.feature_names):
                            if feature_name == token:
                                token_idx = j
                                break
                        
                        if token_idx is not None:
                            weight = doc_tfidf[token_idx]
                            doc_vector += weight * self.w2v_model.wv[token]
                            total_weight += weight
            
            # Normalize by total weight if non-zero
            if total_weight > 0:
                doc_vector = doc_vector / total_weight
                
            doc_vectors.append(doc_vector)
            
        doc_vectors = np.array(doc_vectors)
        
        logger.info(f"Created weighted document vectors: {doc_vectors.shape}")
        
        # Standardize if requested
        if self.standardize:
            doc_vectors = self.scaler.fit_transform(doc_vectors)
            logger.info("Weighted document vectors standardized")
            
        return doc_vectors
        
    def extract_features(self, w2v_model_path: str, jobs_path: str, programs_path: str) -> Tuple[np.ndarray, List[str]]:
        """Extract TF-IDF-weighted Word2Vec features.
        
        Args:
            w2v_model_path: Path to Word2Vec model
            jobs_path: Path to jobs tokenized data
            programs_path: Path to programs tokenized data
            
        Returns:
            Tuple of (features array, document types)
        """
        # Load Word2Vec model
        self.load_word2vec_model(w2v_model_path)
        
        # Prepare tokenized data
        tokenized_texts, doc_types = self.prepare_tokenized_data(jobs_path, programs_path)
        
        # Create TF-IDF weights
        tfidf_matrix = self.create_tfidf_weights(tokenized_texts)
        
        # Create weighted vectors
        features = self.create_weighted_vectors(tokenized_texts, tfidf_matrix)
        
        return features, doc_types


class TfidfFeatureExtractor:
    """Generate and reduce TF-IDF features."""
    
    def __init__(
        self, 
        max_features: int = 10000, 
        pca_dims: int = 50, 
        standardize: bool = True,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        stop_words: Union[str, List[str], None] = 'english'
    ):
        """Initialize TF-IDF feature extractor.
        
        Args:
            max_features: Maximum number of TF-IDF features
            pca_dims: Number of PCA dimensions for reduction
            standardize: Whether to standardize features
            ngram_range: Tuple of (min_n, max_n) for n-gram extraction
            min_df: Minimum document frequency for features
            max_df: Maximum document frequency for features
            stop_words: Stop words to remove ('english' or list or None)
        """
        self.max_features = max_features
        self.pca_dims = pca_dims
        self.standardize = standardize
        self.ngram_range = tuple(ngram_range) if isinstance(ngram_range, (list, tuple)) else ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.stop_words = stop_words
        self.scaler = StandardScaler() if standardize else None
        self.pca = PCA(n_components=pca_dims, random_state=42)
        
    def prepare_text_data(self, jobs_path: str, programs_path: str) -> Tuple[List[str], List[str]]:
        """Prepare text data from jobs and programs.
        
        Args:
            jobs_path: Path to jobs data
            programs_path: Path to programs data
            
        Returns:
            Tuple of (text_strings, doc_types)
        """
        logger.info("Preparing text data for TF-IDF")
        
        # Load jobs data
        jobs_df = pd.read_parquet(jobs_path)
        jobs_texts = jobs_df['description_text'].dropna().tolist()
        jobs_types = ['JOB'] * len(jobs_texts)
        
        # Load programs data
        programs_df = pd.read_parquet(programs_path)
        programs_texts = programs_df['description_text'].dropna().tolist()
        programs_types = ['PROGRAM'] * len(programs_texts)
        
        # Combine
        all_texts = jobs_texts + programs_texts
        all_types = jobs_types + programs_types
        
        logger.info(f"Prepared {len(jobs_texts)} job texts and {len(programs_texts)} program texts")
        
        return all_texts, all_types
        
    def extract_features(self, jobs_path: str, programs_path: str) -> Tuple[np.ndarray, List[str]]:
        """Extract TF-IDF features with PCA reduction.
        
        Args:
            jobs_path: Path to jobs data
            programs_path: Path to programs data
            
        Returns:
            Tuple of (features array, document types)
        """
        # Prepare text data
        texts, doc_types = self.prepare_text_data(jobs_path, programs_path)
        
        # Create TF-IDF vectors
        logger.info(f"Creating TF-IDF vectors with max_features={self.max_features}, ngram_range={self.ngram_range}")
        
        vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            lowercase=True,
            stop_words=self.stop_words
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        logger.info(f"Created TF-IDF matrix: {tfidf_matrix.shape}")
        
        tfidf_dense = tfidf_matrix.toarray()
        
        # Apply PCA reduction if requested
        if self.pca_dims is not None:
            logger.info(f"Applying PCA reduction to {self.pca_dims} dimensions")
            features = self.pca.fit_transform(tfidf_dense)
            logger.info(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
        else:
            features = tfidf_dense
            logger.info("Skipping PCA reduction")
        
        # Standardize if requested
        if self.standardize:
            features = self.scaler.fit_transform(features)
            logger.info("TF-IDF features standardized")
            
        return features, doc_types


def extract_doc2vec_features(df: pd.DataFrame) -> np.ndarray:
    """Extract Doc2Vec features from dataframe.
    
    Args:
        df: DataFrame with Doc2Vec vectors
        
    Returns:
        Features array
    """
    vector_cols = [col for col in df.columns if col.startswith('vec_')]
    return df[vector_cols].values


def create_w2v_tfidf_weighted_vectors(w2v_model, tokens: List[List[str]], tfidf_weights: np.ndarray) -> np.ndarray:
    """Create TF-IDF-weighted Word2Vec document vectors.
    
    Args:
        w2v_model: Trained Word2Vec model
        tokens: List of tokenized documents
        tfidf_weights: TF-IDF weights matrix
        
    Returns:
        Weighted document vectors
    """
    doc_vectors = []
    vector_dim = w2v_model.wv.vector_size
    
    for i, doc_tokens in enumerate(tokens):
        doc_vector = np.zeros(vector_dim)
        total_weight = 0
        
        for token in doc_tokens:
            if token in w2v_model.wv:
                # Simple TF-IDF weighting (assuming tokens match TF-IDF vocabulary)
                weight = 1.0  # Simplified - would need proper token-to-TF-IDF mapping
                doc_vector += weight * w2v_model.wv[token]
                total_weight += weight
        
        if total_weight > 0:
            doc_vector = doc_vector / total_weight
            
        doc_vectors.append(doc_vector)
    
    return np.array(doc_vectors)


def extract_tfidf_features(texts: List[str], max_features: int = 10000, pca_dims: int = 50) -> np.ndarray:
    """Extract TF-IDF features with PCA reduction.
    
    Args:
        texts: List of text documents
        max_features: Maximum number of TF-IDF features
        pca_dims: Number of PCA dimensions
        
    Returns:
        Reduced TF-IDF features
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        lowercase=True,
        stop_words='english'
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    tfidf_dense = tfidf_matrix.toarray()
    
    pca = PCA(n_components=pca_dims, random_state=42)
    features = pca.fit_transform(tfidf_dense)
    
    return features


def main():
    """Main function for testing feature extractors."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test clustering feature extractors")
    parser.add_argument("--input", required=True, help="Input data path")
    parser.add_argument("--output", required=True, help="Output features path")
    parser.add_argument("--method", choices=["doc2vec", "w2v_tfidf", "tfidf"], required=True)
    
    args = parser.parse_args()
    
    if args.method == "doc2vec":
        extractor = Doc2VecFeatureExtractor()
        features, metadata = extractor.extract_features(args.input)
        np.save(args.output, features)
        metadata.to_parquet(args.output.replace('.npy', '_metadata.parquet'))
        
    elif args.method == "w2v_tfidf":
        extractor = Word2VecTfidfFeatureExtractor()
        features, doc_types = extractor.extract_features(
            args.input + "/jobs_tokenized.parquet",
            args.input + "/programs_tokenized.parquet"
        )
        np.save(args.output, features)
        
    elif args.method == "tfidf":
        extractor = TfidfFeatureExtractor()
        features, doc_types = extractor.extract_features(
            args.input + "/jobs_tokenized.parquet",
            args.input + "/programs_tokenized.parquet"
        )
        np.save(args.output, features)
    
    print(f"Features saved to {args.output}")


if __name__ == "__main__":
    main()
