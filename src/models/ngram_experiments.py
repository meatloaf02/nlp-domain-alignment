"""
TF-IDF N-gram Experiment Module.

This module implements experiments for TF-IDF with word bigrams and trigrams,
varying ngram_range, min_df, and max_features parameters.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import pickle
from itertools import product

from src.models.clustering_features import TfidfFeatureExtractor

logger = logging.getLogger(__name__)


class TfidfNgramExperiment:
    """Experiment runner for TF-IDF word n-gram configurations."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize TF-IDF n-gram experiment runner.
        
        Args:
            config: Configuration dictionary with experiment parameters
        """
        self.config = config
        self.tfidf_config = config.get('tfidf_word_ngrams', {})
        self.data_config = config.get('data', {})
        self.output_config = config.get('output', {})
        
    def generate_configurations(self) -> List[Dict[str, Any]]:
        """Generate all experiment configurations from grid.
        
        Returns:
            List of configuration dictionaries
        """
        ngram_ranges = self.tfidf_config.get('ngram_ranges', [(1, 2)])
        min_df_values = self.tfidf_config.get('min_df_values', [2])
        max_features_values = self.tfidf_config.get('max_features_values', [10000])
        
        # Convert ngram ranges to tuples
        ngram_ranges = [tuple(r) if isinstance(r, list) else r for r in ngram_ranges]
        
        configurations = []
        for ngram_range, min_df, max_features in product(ngram_ranges, min_df_values, max_features_values):
            config = {
                'ngram_range': ngram_range,
                'min_df': min_df,
                'max_features': max_features,
                'max_df': self.tfidf_config.get('max_df', 0.95),
                'stop_words': self.tfidf_config.get('stop_words', 'english')
            }
            configurations.append(config)
        
        logger.info(f"Generated {len(configurations)} TF-IDF n-gram configurations")
        return configurations
    
    def run_experiment(
        self, 
        config: Dict[str, Any],
        texts: List[str],
        doc_types: List[str],
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Any, Dict[str, Any]]:
        """Run a single TF-IDF n-gram experiment.
        
        Args:
            config: Experiment configuration
            texts: List of text documents
            doc_types: List of document types ('JOB' or 'PROGRAM')
            train_indices: Optional array of training indices (for proper train/test split)
            test_indices: Optional array of test indices (for proper train/test split)
            
        Returns:
            Tuple of (features array, vectorizer, metadata)
        """
        logger.info(f"Running experiment: ngram_range={config['ngram_range']}, "
                   f"min_df={config['min_df']}, max_features={config['max_features']}")
        
        # For TF-IDF, we need to modify extract_features to work with text lists
        # Let's use the vectorizer directly
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        vectorizer = TfidfVectorizer(
            max_features=config['max_features'],
            min_df=config['min_df'],
            max_df=config['max_df'],
            ngram_range=config['ngram_range'],
            lowercase=True,
            stop_words=config['stop_words']
        )
        
        # Fix data leakage: fit only on training data if splits provided
        if train_indices is not None and test_indices is not None:
            train_texts = [texts[i] for i in train_indices]
            test_texts = [texts[i] for i in test_indices]
            
            # Fit on train only, transform separately
            train_features = vectorizer.fit_transform(train_texts)
            test_features = vectorizer.transform(test_texts)
            
            # Reassemble features in original order
            features_dense = np.zeros((len(texts), train_features.shape[1]))
            features_dense[train_indices] = train_features.toarray()
            features_dense[test_indices] = test_features.toarray()
            
            logger.info(f"Fit vectorizer on {len(train_texts)} train samples, "
                       f"transformed {len(test_texts)} test samples separately")
        else:
            # Backward compatibility: fit on all data (not recommended for evaluation)
            features = vectorizer.fit_transform(texts)
            features_dense = features.toarray()
            logger.warning("No train/test split provided - fitting on all data (potential data leakage)")
        
        metadata = {
            'n_documents': len(texts),
            'n_features': features_dense.shape[1],
            'feature_names': vectorizer.get_feature_names_out().tolist() if hasattr(vectorizer, 'get_feature_names_out') else None,
            'config': config,
            'train_test_split_used': train_indices is not None and test_indices is not None
        }
        
        logger.info(f"Created features: {features_dense.shape}")
        
        return features_dense, vectorizer, metadata
    
    def run_all_experiments(
        self,
        texts: List[str],
        doc_types: List[str],
        output_dir: Path,
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """Run all TF-IDF n-gram experiments and save results.
        
        Args:
            texts: List of text documents
            doc_types: List of document types
            output_dir: Output directory for results
            
        Returns:
            DataFrame with summary of all experiments
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        configurations = self.generate_configurations()
        
        results = []
        
        for i, config in enumerate(configurations):
            logger.info(f"Experiment {i+1}/{len(configurations)}")
            
            try:
                # Run experiment
                features, vectorizer, metadata = self.run_experiment(
                    config, texts, doc_types, train_indices, test_indices
                )
                
                # Create experiment directory
                config_name = (
                    f"ngram_{config['ngram_range'][0]}-{config['ngram_range'][1]}_"
                    f"min_df_{config['min_df']}_max_feat_{config['max_features']}"
                )
                exp_dir = output_dir / config_name
                exp_dir.mkdir(parents=True, exist_ok=True)
                
                # Save features
                features_path = exp_dir / "features.npy"
                np.save(str(features_path), features)
                
                # Save vectorizer
                vectorizer_path = exp_dir / "vectorizer.pkl"
                with open(vectorizer_path, 'wb') as f:
                    pickle.dump(vectorizer, f)
                
                # Save metadata
                metadata_path = exp_dir / "metadata.json"
                # Remove non-serializable objects
                metadata_clean = metadata.copy()
                if metadata_clean.get('feature_names'):
                    # Store only first 100 feature names for preview
                    metadata_clean['feature_names_preview'] = metadata_clean['feature_names'][:100]
                    metadata_clean['n_feature_names'] = len(metadata_clean['feature_names'])
                    del metadata_clean['feature_names']
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata_clean, f, indent=2)
                
                # Store result summary
                result = {
                    'experiment_id': config_name,
                    'ngram_range': str(config['ngram_range']),
                    'min_df': config['min_df'],
                    'max_features': config['max_features'],
                    'n_documents': metadata['n_documents'],
                    'n_features': metadata['n_features'],
                    'features_path': str(features_path),
                    'vectorizer_path': str(vectorizer_path),
                    'metadata_path': str(metadata_path)
                }
                results.append(result)
                
                logger.info(f"Saved experiment to {exp_dir}")
                
            except Exception as e:
                logger.error(f"Experiment failed for config {config}: {e}")
                continue
        
        # Save summary
        summary_df = pd.DataFrame(results)
        summary_path = output_dir / "summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Saved summary to {summary_path}")
        
        return summary_df


def run_tfidf_ngram_experiments(
    config: Dict[str, Any],
    texts: List[str],
    doc_types: List[str],
    output_dir: Path
) -> pd.DataFrame:
    """Convenience function to run TF-IDF n-gram experiments.
    
    Args:
        config: Experiment configuration
        texts: List of text documents
        doc_types: List of document types
        output_dir: Output directory
        
    Returns:
        Summary DataFrame
    """
    experiment = TfidfNgramExperiment(config)
    return experiment.run_all_experiments(texts, doc_types, output_dir)

