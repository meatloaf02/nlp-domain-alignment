"""
Feature Fusion Experiment Module.

This module implements experiments for combining word n-grams (1-2) and 
character n-grams (3-5) in TF-IDF, with optional phrase modeling integration.
Tests additive vs weighted concatenation strategies.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import pickle
from itertools import product

from gensim.models.phrases import Phrases, Phraser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize as sklearn_normalize

logger = logging.getLogger(__name__)


class FeatureFusionExperiment:
    """Experiment runner for feature fusion with word+char n-grams and phrase modeling."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize feature fusion experiment runner.
        
        Args:
            config: Configuration dictionary with experiment parameters
        """
        self.config = config
        self.fusion_config = config.get('feature_fusion', {})
        self.data_config = config.get('data', {})
        self.output_config = config.get('output', {})
        
    def generate_configurations(self) -> List[Dict[str, Any]]:
        """Generate all experiment configurations from grid.
        
        Returns:
            List of configuration dictionaries
        """
        fusion_strategies = self.fusion_config.get('fusion_strategies', ['additive'])
        # Get phrase thresholds from phrase_modeling.thresholds config path
        phrase_modeling_config = self.fusion_config.get('phrase_modeling', {})
        phrase_thresholds = phrase_modeling_config.get('thresholds', [None])
        # Add None to phrase_thresholds if not present to test without phrases
        if None not in phrase_thresholds:
            phrase_thresholds = [None] + phrase_thresholds
        
        configurations = []
        for strategy, threshold in product(fusion_strategies, phrase_thresholds):
            config = {
                'fusion_strategy': strategy,
                'phrase_threshold': threshold,
                'word_ngram_range': tuple(self.fusion_config.get('word_ngrams', {}).get('ngram_range', [1, 2])),
                'char_ngram_range': tuple(self.fusion_config.get('char_ngrams', {}).get('ngram_range', [3, 5])),
                'word_max_features': self.fusion_config.get('word_ngrams', {}).get('max_features', 20000),
                'char_max_features': self.fusion_config.get('char_ngrams', {}).get('max_features', 20000),
                'word_min_df': self.fusion_config.get('word_ngrams', {}).get('min_df', 2),
                'char_min_df': self.fusion_config.get('char_ngrams', {}).get('min_df', 2),
                'word_weight': self.fusion_config.get('weighted_fusion', {}).get('word_weight', 0.6),
                'char_weight': self.fusion_config.get('weighted_fusion', {}).get('char_weight', 0.4),
                'phrase_min_count': self.fusion_config.get('phrase_modeling', {}).get('min_count', 5),
                'normalize': self.fusion_config.get('normalize', True)
            }
            configurations.append(config)
        
        logger.info(f"Generated {len(configurations)} feature fusion configurations")
        return configurations
    
    def apply_phrases_with_tracking(
        self,
        tokenized_docs: List[List[str]],
        threshold: int,
        min_count: int = 5
    ) -> Tuple[List[List[str]], Phrases, Dict[str, Any]]:
        """Apply phrase detection with merge-rate and vocabulary tracking.
        
        Args:
            tokenized_docs: List of tokenized documents
            threshold: Phrase threshold for gensim Phrases
            min_count: Minimum count for phrases
            
        Returns:
            Tuple of (phrased_docs, bigram_model, stats_dict)
        """
        # Calculate original statistics
        original_total_tokens = sum(len(doc) for doc in tokenized_docs)
        original_unique_tokens = len(set(token for doc in tokenized_docs for token in doc))
        
        # Apply bigrams
        logger.info(f"Training bigram model: min_count={min_count}, threshold={threshold}")
        bigram_model = Phrases(
            tokenized_docs,
            min_count=min_count,
            threshold=threshold,
            delimiter='_'
        )
        
        bigram_docs = [bigram_model[doc] for doc in tokenized_docs]
        
        # Apply trigrams (hierarchical)
        logger.info(f"Training trigram model: min_count={min_count}, threshold={threshold}")
        trigram_model = Phrases(
            bigram_docs,
            min_count=min_count,
            threshold=threshold,
            delimiter='_'
        )
        
        final_docs = [trigram_model[doc] for doc in bigram_docs]
        
        # Calculate final statistics
        final_total_tokens = sum(len(doc) for doc in final_docs)
        final_unique_tokens = len(set(token for doc in final_docs for token in doc))
        
        # Calculate merge rate
        merge_rate = (original_total_tokens - final_total_tokens) / original_total_tokens if original_total_tokens > 0 else 0.0
        
        stats = {
            'original_total_tokens': original_total_tokens,
            'original_unique_tokens': original_unique_tokens,
            'final_total_tokens': final_total_tokens,
            'final_unique_tokens': final_unique_tokens,
            'merge_rate': merge_rate,
            'tokens_merged': original_total_tokens - final_total_tokens,
            'vocab_reduction': original_unique_tokens - final_unique_tokens
        }
        
        logger.info(f"Phrase stats: merge_rate={merge_rate:.4f}, "
                   f"vocab: {original_unique_tokens} -> {final_unique_tokens}")
        
        # Combine bigram and trigram models conceptually
        # For simplicity, we'll save both separately
        return final_docs, bigram_model, stats
    
    def extract_word_ngrams(
        self,
        texts: List[str],
        config: Dict[str, Any],
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, TfidfVectorizer]:
        """Extract word n-gram features using TF-IDF.
        
        Args:
            texts: List of text documents
            config: Experiment configuration
            train_indices: Optional array of training indices
            test_indices: Optional array of test indices
            
        Returns:
            Tuple of (features, vectorizer)
        """
        logger.info(f"Extracting word n-grams: ngram_range={config['word_ngram_range']}")
        
        vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=config['word_ngram_range'],
            max_features=config['word_max_features'],
            min_df=config['word_min_df'],
            max_df=0.95,
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
    
    def extract_char_ngrams(
        self,
        texts: List[str],
        config: Dict[str, Any],
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, TfidfVectorizer]:
        """Extract character n-gram features using TF-IDF.
        
        Args:
            texts: List of text documents
            config: Experiment configuration
            train_indices: Optional array of training indices
            test_indices: Optional array of test indices
            
        Returns:
            Tuple of (features, vectorizer)
        """
        logger.info(f"Extracting character n-grams: ngram_range={config['char_ngram_range']}")
        
        vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=config['char_ngram_range'],
            max_features=config['char_max_features'],
            min_df=config['char_min_df'],
            max_df=0.95,
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
    
    def fuse_features(
        self,
        word_features: np.ndarray,
        char_features: np.ndarray,
        strategy: str,
        word_weight: float = 0.6,
        char_weight: float = 0.4,
        normalize: bool = True
    ) -> np.ndarray:
        """Fuse word and character n-gram features.
        
        Args:
            word_features: Word n-gram feature matrix
            char_features: Character n-gram feature matrix
            strategy: Fusion strategy ('additive' or 'weighted')
            word_weight: Weight for word features (for weighted strategy)
            char_weight: Weight for char features (for weighted strategy)
            normalize: Whether to normalize features before fusion
            
        Returns:
            Fused feature matrix
        """
        if normalize:
            # Normalize each feature set independently
            word_features = sklearn_normalize(word_features, norm='l2', axis=1)
            char_features = sklearn_normalize(char_features, norm='l2', axis=1)
        
        if strategy == 'additive':
            # Simple concatenation
            fused = np.hstack([word_features, char_features])
            logger.info(f"Additive fusion: {word_features.shape} + {char_features.shape} = {fused.shape}")
        
        elif strategy == 'weighted':
            # Weighted concatenation
            # Note: For weighted, we still concatenate but apply weights
            # This is different from weighted sum which would require same dimensions
            word_weighted = word_weight * word_features
            char_weighted = char_weight * char_features
            fused = np.hstack([word_weighted, char_weighted])
            logger.info(f"Weighted fusion (w={word_weight}, c={char_weight}): "
                       f"{word_features.shape} + {char_features.shape} = {fused.shape}")
        else:
            raise ValueError(f"Unknown fusion strategy: {strategy}")
        
        return fused
    
    def run_experiment(
        self,
        texts: List[str],
        tokenized_docs: Optional[List[List[str]]],
        config: Dict[str, Any],
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
        """Run a single feature fusion experiment.
        
        Args:
            texts: List of text documents
            tokenized_docs: List of tokenized documents (for phrase modeling)
            config: Experiment configuration
            train_indices: Optional array of training indices
            test_indices: Optional array of test indices
            
        Returns:
            Tuple of (features, metadata, vectorizers_dict)
        """
        logger.info(f"Running experiment: strategy={config['fusion_strategy']}, "
                   f"phrase_threshold={config['phrase_threshold']}")
        
        processed_texts = texts
        phrase_stats = None
        bigram_model = None
        trigram_model = None
        
        # Apply phrase modeling if threshold is specified
        if config['phrase_threshold'] is not None and tokenized_docs is not None:
            logger.info(f"Applying phrase modeling with threshold={config['phrase_threshold']}")
            phrased_docs, bigram_model, phrase_stats = self.apply_phrases_with_tracking(
                tokenized_docs,
                threshold=config['phrase_threshold'],
                min_count=config['phrase_min_count']
            )
            # Convert phrased docs back to strings for TF-IDF
            processed_texts = [' '.join(doc) for doc in phrased_docs]
        else:
            logger.info("Skipping phrase modeling")
        
        # Extract word n-grams
        word_features, word_vectorizer = self.extract_word_ngrams(
            processed_texts, config, train_indices, test_indices
        )
        
        # Extract character n-grams
        char_features, char_vectorizer = self.extract_char_ngrams(
            processed_texts, config, train_indices, test_indices
        )
        
        # Fuse features
        fused_features = self.fuse_features(
            word_features,
            char_features,
            strategy=config['fusion_strategy'],
            word_weight=config['word_weight'],
            char_weight=config['char_weight'],
            normalize=config['normalize']
        )
        
        # Prepare metadata
        metadata = {
            'n_documents': len(texts),
            'n_features': fused_features.shape[1],
            'word_n_features': word_features.shape[1],
            'char_n_features': char_features.shape[1],
            'config': config,
            'train_test_split_used': train_indices is not None and test_indices is not None,
            'phrase_stats': phrase_stats
        }
        
        vectorizers = {
            'word_vectorizer': word_vectorizer,
            'char_vectorizer': char_vectorizer
        }
        
        logger.info(f"Final fused features shape: {fused_features.shape}")
        
        return fused_features, metadata, vectorizers
    
    def run_all_experiments(
        self,
        texts: List[str],
        tokenized_docs: List[List[str]],
        doc_types: List[str],
        output_dir: Path,
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """Run all feature fusion experiments and save results.
        
        Args:
            texts: List of text documents
            tokenized_docs: List of tokenized documents
            doc_types: List of document types
            output_dir: Output directory for results
            train_indices: Optional array of training indices
            test_indices: Optional array of test indices
            
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
                features, metadata, vectorizers = self.run_experiment(
                    texts, tokenized_docs, config, train_indices, test_indices
                )
                
                # Create experiment directory name
                strategy = config['fusion_strategy']
                if config['phrase_threshold'] is not None:
                    config_name = f"{strategy}_phrases_threshold_{config['phrase_threshold']}"
                else:
                    config_name = f"{strategy}_no_phrases"
                
                exp_dir = output_dir / config_name
                exp_dir.mkdir(parents=True, exist_ok=True)
                
                # Create subdirectories
                vectorizers_dir = exp_dir / "vectorizers"
                vectorizers_dir.mkdir(exist_ok=True)
                
                # Save features
                features_path = exp_dir / "features.npy"
                np.save(str(features_path), features)
                
                # Save vectorizers
                word_vectorizer_path = vectorizers_dir / "word_vectorizer.pkl"
                with open(word_vectorizer_path, 'wb') as f:
                    pickle.dump(vectorizers['word_vectorizer'], f)
                
                char_vectorizer_path = vectorizers_dir / "char_vectorizer.pkl"
                with open(char_vectorizer_path, 'wb') as f:
                    pickle.dump(vectorizers['char_vectorizer'], f)
                
                # Save phrase models if applicable
                if metadata['phrase_stats'] is not None:
                    phrase_models_dir = exp_dir / "phrase_models"
                    phrase_models_dir.mkdir(exist_ok=True)
                    # Note: We don't save phrase models here as they're not returned
                    # This would need to be modified if we want to save them
                
                # Save metadata
                metadata_path = exp_dir / "metadata.json"
                metadata_clean = {
                    'n_documents': metadata['n_documents'],
                    'n_features': metadata['n_features'],
                    'word_n_features': metadata['word_n_features'],
                    'char_n_features': metadata['char_n_features'],
                    'config': config,
                    'phrase_stats': metadata['phrase_stats']
                }
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata_clean, f, indent=2)
                
                # Store result summary
                result = {
                    'experiment_id': config_name,
                    'fusion_strategy': strategy,
                    'phrase_threshold': config['phrase_threshold'],
                    'word_ngram_range': str(config['word_ngram_range']),
                    'char_ngram_range': str(config['char_ngram_range']),
                    'n_documents': metadata['n_documents'],
                    'n_features': metadata['n_features'],
                    'word_n_features': metadata['word_n_features'],
                    'char_n_features': metadata['char_n_features'],
                    'features_path': str(features_path),
                    'metadata_path': str(metadata_path)
                }
                
                # Add phrase statistics if available
                if metadata['phrase_stats']:
                    result['merge_rate'] = metadata['phrase_stats']['merge_rate']
                    result['vocab_size_before'] = metadata['phrase_stats']['original_unique_tokens']
                    result['vocab_size_after'] = metadata['phrase_stats']['final_unique_tokens']
                    result['tokens_merged'] = metadata['phrase_stats']['tokens_merged']
                
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


def run_feature_fusion_experiments(
    config: Dict[str, Any],
    texts: List[str],
    tokenized_docs: List[List[str]],
    doc_types: List[str],
    output_dir: Path,
    train_indices: Optional[np.ndarray] = None,
    test_indices: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """Convenience function to run feature fusion experiments.
    
    Args:
        config: Experiment configuration
        texts: List of text documents
        tokenized_docs: List of tokenized documents
        doc_types: List of document types
        output_dir: Output directory
        train_indices: Optional array of training indices
        test_indices: Optional array of test indices
        
    Returns:
        Summary DataFrame
    """
    experiment = FeatureFusionExperiment(config)
    return experiment.run_all_experiments(
        texts, tokenized_docs, doc_types, output_dir, train_indices, test_indices
    )

