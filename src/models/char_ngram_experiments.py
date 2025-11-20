"""
Character N-gram Experiment Module.

This module implements experiments for character n-grams (n=3-5) 
alone and concatenated with word n-grams.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import pickle
from itertools import product
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class CharNGramExperiment:
    """Experiment runner for character n-gram configurations."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize character n-gram experiment runner.
        
        Args:
            config: Configuration dictionary with experiment parameters
        """
        self.config = config
        self.char_config = config.get('char_ngrams', {})
        self.data_config = config.get('data', {})
        self.output_config = config.get('output', {})
        
    def generate_configurations(self) -> List[Dict[str, Any]]:
        """Generate all experiment configurations from grid.
        
        Returns:
            List of configuration dictionaries
        """
        ngram_ranges = self.char_config.get('ngram_ranges', [(3, 3)])
        min_df_values = self.char_config.get('min_df_values', [2])
        max_features_values = self.char_config.get('max_features_values', [20000])
        combinations = self.char_config.get('combinations', ['char_only'])
        
        # Word n-grams config for combination
        word_config = self.char_config.get('word_ngrams_for_combination', {})
        
        # Convert ngram ranges to tuples
        ngram_ranges = [tuple(r) if isinstance(r, list) else r for r in ngram_ranges]
        
        configurations = []
        for ngram_range, min_df, max_features, combination in product(
            ngram_ranges, min_df_values, max_features_values, combinations
        ):
            config = {
                'char_ngram_range': ngram_range,
                'min_df': min_df,
                'max_features': max_features,
                'max_df': self.char_config.get('max_df', 0.95),
                'combination': combination,
                'word_ngram_range': word_config.get('ngram_range', (1, 2)) if combination == 'word_char_combined' else None,
                'word_min_df': word_config.get('min_df', 2) if combination == 'word_char_combined' else None,
                'word_max_features': word_config.get('max_features', 20000) if combination == 'word_char_combined' else None
            }
            configurations.append(config)
        
        logger.info(f"Generated {len(configurations)} character n-gram configurations")
        return configurations
    
    def extract_char_ngrams(
        self,
        texts: List[str],
        config: Dict[str, Any],
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, TfidfVectorizer]:
        """Extract character n-gram features.
        
        Args:
            texts: List of text documents
            config: Experiment configuration
            train_indices: Optional array of training indices (for proper train/test split)
            test_indices: Optional array of test indices (for proper train/test split)
            
        Returns:
            Tuple of (features, vectorizer)
        """
        logger.info(f"Extracting character n-grams: ngram_range={config['char_ngram_range']}")
        
        vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=config['char_ngram_range'],
            max_features=config['max_features'],
            min_df=config['min_df'],
            max_df=config['max_df'],
            lowercase=True
        )
        
        # Fix data leakage: fit only on training data if splits provided
        if train_indices is not None and test_indices is not None:
            train_texts = [texts[i] for i in train_indices]
            test_texts = [texts[i] for i in test_indices]
            
            train_features = vectorizer.fit_transform(train_texts)
            test_features = vectorizer.transform(test_texts)
            
            # Reassemble features in original order
            features_dense = np.zeros((len(texts), train_features.shape[1]))
            features_dense[train_indices] = train_features.toarray()
            features_dense[test_indices] = test_features.toarray()
        else:
            features = vectorizer.fit_transform(texts)
            features_dense = features.toarray()
        
        logger.info(f"Created character n-gram features: {features_dense.shape}")
        
        return features_dense, vectorizer
    
    def extract_word_ngrams(
        self,
        texts: List[str],
        config: Dict[str, Any],
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, TfidfVectorizer]:
        """Extract word n-gram features for combination.
        
        Args:
            texts: List of text documents
            config: Experiment configuration
            train_indices: Optional array of training indices (for proper train/test split)
            test_indices: Optional array of test indices (for proper train/test split)
            
        Returns:
            Tuple of (features, vectorizer)
        """
        logger.info(f"Extracting word n-grams: ngram_range={config['word_ngram_range']}")
        
        vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=config['word_ngram_range'],
            max_features=config['word_max_features'],
            min_df=config['word_min_df'],
            max_df=config['max_df'],
            lowercase=True,
            stop_words='english'
        )
        
        # Fix data leakage: fit only on training data if splits provided
        if train_indices is not None and test_indices is not None:
            train_texts = [texts[i] for i in train_indices]
            test_texts = [texts[i] for i in test_indices]
            
            train_features = vectorizer.fit_transform(train_texts)
            test_features = vectorizer.transform(test_texts)
            
            # Reassemble features in original order
            features_dense = np.zeros((len(texts), train_features.shape[1]))
            features_dense[train_indices] = train_features.toarray()
            features_dense[test_indices] = test_features.toarray()
        else:
            features = vectorizer.fit_transform(texts)
            features_dense = features.toarray()
        
        logger.info(f"Created word n-gram features: {features_dense.shape}")
        
        return features_dense, vectorizer
    
    def combine_features(
        self,
        char_features: np.ndarray,
        word_features: Optional[np.ndarray],
        normalize: bool = True
    ) -> np.ndarray:
        """Combine character and word n-gram features.
        
        Args:
            char_features: Character n-gram features
            word_features: Word n-gram features (None for char-only)
            normalize: Whether to normalize features before combining
            
        Returns:
            Combined features array
        """
        if word_features is None:
            # Char-only
            combined = char_features
        else:
            # Normalize if requested
            if normalize:
                scaler_char = StandardScaler()
                scaler_word = StandardScaler()
                char_features = scaler_char.fit_transform(char_features)
                word_features = scaler_word.fit_transform(word_features)
            
            # Concatenate features
            combined = np.hstack([char_features, word_features])
            logger.info(f"Combined features shape: {combined.shape}")
        
        return combined
    
    def run_experiment(
        self,
        texts: List[str],
        config: Dict[str, Any],
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
        """Run a single character n-gram experiment.
        
        Args:
            texts: List of text documents
            config: Experiment configuration
            train_indices: Optional array of training indices (for proper train/test split)
            test_indices: Optional array of test indices (for proper train/test split)
            
        Returns:
            Tuple of (features, char_vectorizer, word_vectorizer_or_None, metadata)
        """
        logger.info(f"Running experiment: char_ngram_range={config['char_ngram_range']}, "
                   f"combination={config['combination']}")
        
        # Extract character n-grams
        char_features, char_vectorizer = self.extract_char_ngrams(
            texts, config, train_indices, test_indices
        )
        
        word_vectorizer = None
        word_features = None
        
        # Extract word n-grams if combining
        if config['combination'] == 'word_char_combined':
            word_features, word_vectorizer = self.extract_word_ngrams(
                texts, config, train_indices, test_indices
            )
        
        # Combine features
        combined_features = self.combine_features(char_features, word_features)
        
        metadata = {
            'n_documents': len(texts),
            'n_features': combined_features.shape[1],
            'char_n_features': char_features.shape[1],
            'word_n_features': word_features.shape[1] if word_features is not None else 0,
            'config': config,
            'train_test_split_used': train_indices is not None and test_indices is not None
        }
        
        logger.info(f"Final features shape: {combined_features.shape}")
        
        return combined_features, char_vectorizer, word_vectorizer, metadata
    
    def run_all_experiments(
        self,
        texts: List[str],
        doc_types: List[str],
        output_dir: Path,
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """Run all character n-gram experiments and save results.
        
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
                features, char_vectorizer, word_vectorizer, metadata = self.run_experiment(
                    texts, config, train_indices, test_indices
                )
                
                # Create experiment directory name
                ngram_str = f"{config['char_ngram_range'][0]}-{config['char_ngram_range'][1]}"
                config_name = (
                    f"char_{ngram_str}_min_{config['min_df']}_max_{config['max_features']}_"
                    f"combined_{config['combination']}"
                )
                exp_dir = output_dir / config_name
                exp_dir.mkdir(parents=True, exist_ok=True)
                
                # Save features
                features_path = exp_dir / "features.npy"
                np.save(str(features_path), features)
                
                # Save vectorizers
                char_vectorizer_path = exp_dir / "char_vectorizer.pkl"
                with open(char_vectorizer_path, 'wb') as f:
                    pickle.dump(char_vectorizer, f)
                
                if word_vectorizer:
                    word_vectorizer_path = exp_dir / "word_vectorizer.pkl"
                    with open(word_vectorizer_path, 'wb') as f:
                        pickle.dump(word_vectorizer, f)
                
                # Save metadata
                metadata_path = exp_dir / "metadata.json"
                metadata_clean = {
                    'n_documents': metadata['n_documents'],
                    'n_features': metadata['n_features'],
                    'char_n_features': metadata['char_n_features'],
                    'word_n_features': metadata['word_n_features'],
                    'config': config
                }
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata_clean, f, indent=2)
                
                # Store result summary
                result = {
                    'experiment_id': config_name,
                    'char_ngram_range': str(config['char_ngram_range']),
                    'min_df': config['min_df'],
                    'max_features': config['max_features'],
                    'combination': config['combination'],
                    'n_documents': metadata['n_documents'],
                    'n_features': metadata['n_features'],
                    'char_n_features': metadata['char_n_features'],
                    'word_n_features': metadata['word_n_features'],
                    'features_path': str(features_path),
                    'metadata_path': str(metadata_path)
                }
                
                if config['combination'] == 'word_char_combined':
                    result['word_ngram_range'] = str(config['word_ngram_range'])
                    result['word_min_df'] = config['word_min_df']
                    result['word_max_features'] = config['word_max_features']
                
                results.append(result)
                
                logger.info(f"Saved experiment to {exp_dir}")
                
            except Exception as e:
                logger.error(f"Experiment failed for config {config}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        
        # Save summary
        summary_df = pd.DataFrame(results)
        summary_path = output_dir / "summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Saved summary to {summary_path}")
        
        return summary_df


def run_char_ngram_experiments(
    config: Dict[str, Any],
    texts: List[str],
    doc_types: List[str],
    output_dir: Path
) -> pd.DataFrame:
    """Convenience function to run character n-gram experiments.
    
    Args:
        config: Experiment configuration
        texts: List of text documents
        doc_types: List of document types
        output_dir: Output directory
        
    Returns:
        Summary DataFrame
    """
    experiment = CharNGramExperiment(config)
    return experiment.run_all_experiments(texts, doc_types, output_dir)

