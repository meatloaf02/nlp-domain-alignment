"""
Phrase Modeling Experiment Module.

This module implements experiments for phrase modeling (bigrams/trigrams) 
before TF-IDF and LDA, using gensim Phrases with hierarchical detection.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import pickle

from gensim.models.phrases import Phrases, Phraser
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class PhraseExperiment:
    """Experiment runner for phrase modeling configurations."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize phrase experiment runner.
        
        Args:
            config: Configuration dictionary with experiment parameters
        """
        self.config = config
        self.phrase_config = config.get('phrase_modeling', {})
        self.data_config = config.get('data', {})
        self.output_config = config.get('output', {})
        
    def generate_configurations(self) -> List[Dict[str, Any]]:
        """Generate all experiment configurations.
        
        Returns:
            List of configuration dictionaries
        """
        configurations = []
        applications = self.phrase_config.get('applications', ['tfidf', 'lda'])
        
        # Bigram configuration (single config)
        bigram_config = self.phrase_config.get('bigrams', {})
        bigram_config_dict = {
            'bigram_min_count': bigram_config.get('min_count', 5),
            'bigram_threshold': bigram_config.get('threshold', 10),
            'trigram_min_count': None,
            'trigram_threshold': None,
            'phrase_type': 'bigram'
        }
        
        # Trigram configurations
        trigram_config = self.phrase_config.get('trigrams', {})
        trigram_thresholds = trigram_config.get('thresholds', [7, 8, 9, 10])
        
        for app in applications:
            # Add bigram config
            config = bigram_config_dict.copy()
            config['application'] = app
            configurations.append(config)
            
            # Add trigram configs
            for threshold in trigram_thresholds:
                config = {
                    'bigram_min_count': bigram_config.get('min_count', 5),
                    'bigram_threshold': bigram_config.get('threshold', 10),
                    'trigram_min_count': trigram_config.get('min_count', 3),
                    'trigram_threshold': threshold,
                    'phrase_type': 'trigram',
                    'application': app
                }
                configurations.append(config)
        
        logger.info(f"Generated {len(configurations)} phrase modeling configurations")
        return configurations
    
    def apply_phrases(
        self,
        tokenized_docs: List[List[str]],
        config: Dict[str, Any]
    ) -> Tuple[List[List[str]], Phrases, Optional[Phrases]]:
        """Apply hierarchical phrase detection to tokenized documents.
        
        Args:
            tokenized_docs: List of tokenized documents
            config: Phrase configuration
            
        Returns:
            Tuple of (phrased_docs, bigram_model, trigram_model_or_None)
        """
        bigram_min_count = config['bigram_min_count']
        bigram_threshold = config['bigram_threshold']
        trigram_min_count = config.get('trigram_min_count')
        trigram_threshold = config.get('trigram_threshold')
        
        # Step 1: Apply bigrams
        logger.info(f"Training bigram model: min_count={bigram_min_count}, threshold={bigram_threshold}")
        bigram_model = Phrases(
            tokenized_docs,
            min_count=bigram_min_count,
            threshold=bigram_threshold,
            delimiter='_'
        )
        
        bigram_docs = [bigram_model[doc] for doc in tokenized_docs]
        logger.info(f"Applied bigrams to {len(bigram_docs)} documents")
        
        trigram_model = None
        
        # Step 2: Apply trigrams if configured
        if trigram_min_count is not None and trigram_threshold is not None:
            logger.info(f"Training trigram model: min_count={trigram_min_count}, threshold={trigram_threshold}")
            trigram_model = Phrases(
                bigram_docs,
                min_count=trigram_min_count,
                threshold=trigram_threshold,
                delimiter='_'
            )
            
            final_docs = [trigram_model[doc] for doc in bigram_docs]
            logger.info(f"Applied trigrams to {len(final_docs)} documents")
        else:
            final_docs = bigram_docs
        
        return final_docs, bigram_model, trigram_model
    
    def run_tfidf_experiment(
        self,
        phrased_docs: List[List[str]],
        config: Dict[str, Any],
        tfidf_config: Dict[str, Any],
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, TfidfVectorizer, Dict[str, Any]]:
        """Run TF-IDF experiment with phrased documents.
        
        Args:
            phrased_docs: Documents with phrases applied
            config: Phrase configuration
            tfidf_config: TF-IDF configuration
            train_indices: Optional array of training indices (for proper train/test split)
            test_indices: Optional array of test indices (for proper train/test split)
            
        Returns:
            Tuple of (features, vectorizer, metadata)
        """
        logger.info("Running TF-IDF with phrases")
        
        # Convert phrased docs to strings
        text_strings = [' '.join(doc) for doc in phrased_docs]
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=tfidf_config.get('max_features', 20000),
            min_df=tfidf_config.get('min_df', 2),
            max_df=tfidf_config.get('max_df', 0.95),
            ngram_range=tfidf_config.get('ngram_range', (1, 1)),  # Unigrams only after phrases
            lowercase=True,
            stop_words=tfidf_config.get('stop_words', 'english')
        )
        
        # Fix data leakage: fit only on training data if splits provided
        if train_indices is not None and test_indices is not None:
            train_texts = [text_strings[i] for i in train_indices]
            test_texts = [text_strings[i] for i in test_indices]
            
            train_features = vectorizer.fit_transform(train_texts)
            test_features = vectorizer.transform(test_texts)
            
            # Reassemble features in original order
            features_dense = np.zeros((len(text_strings), train_features.shape[1]))
            features_dense[train_indices] = train_features.toarray()
            features_dense[test_indices] = test_features.toarray()
        else:
            features = vectorizer.fit_transform(text_strings)
            features_dense = features.toarray()
        
        metadata = {
            'n_documents': len(text_strings),
            'n_features': features_dense.shape[1],
            'phrase_config': config,
            'tfidf_config': tfidf_config,
            'train_test_split_used': train_indices is not None and test_indices is not None
        }
        
        logger.info(f"Created TF-IDF features: {features_dense.shape}")
        
        return features_dense, vectorizer, metadata
    
    def run_all_experiments(
        self,
        tokenized_docs: List[List[str]],
        texts: List[str],
        doc_types: List[str],
        output_dir: Path,
        tfidf_config: Optional[Dict[str, Any]] = None,
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """Run all phrase modeling experiments and save results.
        
        Args:
            tokenized_docs: Tokenized documents (list of token lists)
            texts: Original text documents (for reference)
            doc_types: List of document types
            output_dir: Output directory
            tfidf_config: TF-IDF configuration (defaults if None)
            
        Returns:
            DataFrame with summary of all experiments
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        configurations = self.generate_configurations()
        
        if tfidf_config is None:
            tfidf_config = {
                'max_features': 20000,
                'min_df': 2,
                'max_df': 0.95,
                'ngram_range': (1, 1),
                'stop_words': 'english'
            }
        
        results = []
        
        for i, config in enumerate(configurations):
            logger.info(f"Experiment {i+1}/{len(configurations)}")
            app = config['application']
            phrase_type = config['phrase_type']
            
            try:
                # Apply phrases
                phrased_docs, bigram_model, trigram_model = self.apply_phrases(tokenized_docs, config)
                
                # Create experiment directory name
                if phrase_type == 'bigram':
                    config_name = f"{app}_bigram"
                else:
                    config_name = f"{app}_trigram_threshold_{config['trigram_threshold']}"
                
                exp_dir = output_dir / config_name
                exp_dir.mkdir(parents=True, exist_ok=True)
                
                if app == 'tfidf':
                    # Run TF-IDF experiment
                    features, vectorizer, metadata = self.run_tfidf_experiment(
                        phrased_docs, config, tfidf_config, train_indices, test_indices
                    )
                    
                    # Save features
                    features_path = exp_dir / "features.npy"
                    np.save(str(features_path), features)
                    
                    # Save vectorizer
                    vectorizer_path = exp_dir / "vectorizer.pkl"
                    with open(vectorizer_path, 'wb') as f:
                        pickle.dump(vectorizer, f)
                    
                    # Save phrase models
                    bigram_path = exp_dir / "bigram_model.pkl"
                    with open(bigram_path, 'wb') as f:
                        pickle.dump(bigram_model, f)
                    
                    if trigram_model:
                        trigram_path = exp_dir / "trigram_model.pkl"
                        with open(trigram_path, 'wb') as f:
                            pickle.dump(trigram_model, f)
                    
                    # Save metadata
                    metadata_path = exp_dir / "metadata.json"
                    metadata_clean = {
                        'n_documents': metadata['n_documents'],
                        'n_features': metadata['n_features'],
                        'phrase_config': config,
                        'tfidf_config': tfidf_config
                    }
                    
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata_clean, f, indent=2)
                    
                    result = {
                        'experiment_id': config_name,
                        'application': app,
                        'phrase_type': phrase_type,
                        'bigram_min_count': config['bigram_min_count'],
                        'bigram_threshold': config['bigram_threshold'],
                        'trigram_min_count': config.get('trigram_min_count'),
                        'trigram_threshold': config.get('trigram_threshold'),
                        'n_documents': metadata['n_documents'],
                        'n_features': metadata['n_features'],
                        'features_path': str(features_path),
                        'metadata_path': str(metadata_path)
                    }
                    results.append(result)
                    
                elif app == 'lda':
                    # For LDA, we save the phrased documents for use with LDA trainer
                    # The LDA coherence will be evaluated separately
                    phrased_texts = [' '.join(doc) for doc in phrased_docs]
                    
                    # Save phrased documents
                    phrased_path = exp_dir / "phrased_documents.json"
                    with open(phrased_path, 'w') as f:
                        json.dump(phrased_texts, f, indent=2)
                    
                    # Save phrase models
                    bigram_path = exp_dir / "bigram_model.pkl"
                    with open(bigram_path, 'wb') as f:
                        pickle.dump(bigram_model, f)
                    
                    if trigram_model:
                        trigram_path = exp_dir / "trigram_model.pkl"
                        with open(trigram_path, 'wb') as f:
                            pickle.dump(trigram_model, f)
                    
                    # Save metadata
                    metadata = {
                        'n_documents': len(phrased_texts),
                        'phrase_config': config,
                        'phrased_documents_path': str(phrased_path)
                    }
                    
                    metadata_path = exp_dir / "metadata.json"
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    result = {
                        'experiment_id': config_name,
                        'application': app,
                        'phrase_type': phrase_type,
                        'bigram_min_count': config['bigram_min_count'],
                        'bigram_threshold': config['bigram_threshold'],
                        'trigram_min_count': config.get('trigram_min_count'),
                        'trigram_threshold': config.get('trigram_threshold'),
                        'n_documents': len(phrased_texts),
                        'phrased_documents_path': str(phrased_path),
                        'metadata_path': str(metadata_path)
                    }
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


def run_phrase_experiments(
    config: Dict[str, Any],
    tokenized_docs: List[List[str]],
    texts: List[str],
    doc_types: List[str],
    output_dir: Path,
    tfidf_config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """Convenience function to run phrase experiments.
    
    Args:
        config: Experiment configuration
        tokenized_docs: Tokenized documents
        texts: Original text documents
        doc_types: Document types
        output_dir: Output directory
        tfidf_config: TF-IDF configuration (optional)
        
    Returns:
        Summary DataFrame
    """
    experiment = PhraseExperiment(config)
    return experiment.run_all_experiments(
        tokenized_docs, texts, doc_types, output_dir, tfidf_config
    )

