"""
N-gram Experiment Runner.

This module orchestrates all n-gram experiments, manages data loading,
creates fixed train/test splits, runs evaluations, and saves results.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import yaml
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, f1_score
from sklearn.preprocessing import LabelEncoder
import gensim
from gensim.utils import simple_preprocess

from src.models.ngram_experiments import TfidfNgramExperiment
from src.models.phrase_experiments import PhraseExperiment
from src.models.char_ngram_experiments import CharNGramExperiment
from src.models.feature_fusion_experiments import FeatureFusionExperiment
from src.models.clustering_features import TfidfFeatureExtractor
from src.models.unified_evaluation import RetrievalEvaluator, ClusteringEvaluator
from src.models.sklearn_baseline import JobClassificationBaseline
from src.models.lda_trainer import LDATrainer

logger = logging.getLogger(__name__)


class NgramExperimentRunner:
    """Main orchestrator for n-gram experiments."""
    
    def __init__(self, config_path: str):
        """Initialize experiment runner.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = Path(config_path)
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.experiment_config = self.config.get('experiment', {})
        self.data_config = self.config.get('data', {})
        self.output_config = self.config.get('output', {})
        self.eval_config = self.config.get('evaluation', {})
        
        self.random_state = self.experiment_config.get('random_state', 42)
        self.test_size = self.experiment_config.get('test_size', 0.2)
        
        # Data storage
        self.jobs_df = None
        self.programs_df = None
        self.all_texts = None
        self.all_doc_types = None
        self.tokenized_docs = None
        
        # Fixed splits
        self.train_indices = None
        self.test_indices = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Baseline results
        self.baseline_metrics = {}
        
        # Output directory
        base_dir = Path(self.output_config.get('base_dir', 'artifacts/ngram_experiments'))
        self.output_dir = base_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized experiment runner with output: {self.output_dir}")
    
    def load_data(self) -> None:
        """Load job and program data."""
        logger.info("Loading data")
        
        jobs_path = self.data_config.get('jobs_path')
        programs_path = self.data_config.get('programs_path')
        
        if jobs_path:
            self.jobs_df = pd.read_parquet(jobs_path)
            logger.info(f"Loaded {len(self.jobs_df)} jobs")
        
        if programs_path:
            self.programs_df = pd.read_parquet(programs_path)
            logger.info(f"Loaded {len(self.programs_df)} programs")
        
        # Prepare texts
        text_column = self.experiment_config.get('text_column', 'description_text')
        
        job_texts = []
        if self.jobs_df is not None and text_column in self.jobs_df.columns:
            job_texts = self.jobs_df[text_column].dropna().astype(str).tolist()
        
        program_texts = []
        if self.programs_df is not None and text_column in self.programs_df.columns:
            program_texts = self.programs_df[text_column].dropna().astype(str).tolist()
        
        self.all_texts = job_texts + program_texts
        self.all_doc_types = ['JOB'] * len(job_texts) + ['PROGRAM'] * len(program_texts)
        
        logger.info(f"Total texts: {len(self.all_texts)} ({len(job_texts)} jobs, {len(program_texts)} programs)")
        
        # Tokenize for phrase experiments
        self.tokenized_docs = [simple_preprocess(text, deacc=True, min_len=2, max_len=50) 
                              for text in self.all_texts]
        logger.info(f"Tokenized {len(self.tokenized_docs)} documents")
    
    def create_fixed_splits(self) -> None:
        """Create fixed train/test splits (stratified)."""
        logger.info("Creating fixed train/test splits")
        
        # For classification, we need labels - use doc_type as label
        # Convert doc_types to numeric labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(self.all_doc_types)
        
        # Create split
        stratify = y if self.experiment_config.get('stratify', True) else None
        self.train_indices, self.test_indices = train_test_split(
            np.arange(len(self.all_texts)),
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify
        )
        
        self.X_train = [self.all_texts[i] for i in self.train_indices]
        self.X_test = [self.all_texts[i] for i in self.test_indices]
        self.y_train = y[self.train_indices]
        self.y_test = y[self.test_indices]
        
        logger.info(f"Train size: {len(self.X_train)}, Test size: {len(self.X_test)}")
        
        # Save split info
        split_info = {
            'train_size': len(self.X_train),
            'test_size': len(self.X_test),
            'train_indices': self.train_indices.tolist(),
            'test_indices': self.test_indices.tolist(),
            'random_state': self.random_state,
            'test_size_ratio': self.test_size
        }
        
        split_path = self.output_dir / "data_splits.json"
        with open(split_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        logger.info(f"Saved split info to {split_path}")
    
    def run_baseline(self) -> Dict[str, Any]:
        """Run baseline experiment.
        
        Returns:
            Dictionary with baseline metrics
        """
        logger.info("=" * 80)
        logger.info("RUNNING BASELINE EXPERIMENT")
        logger.info("=" * 80)
        
        baseline_config = self.config.get('baseline', {})
        
        # Create TF-IDF features
        extractor = TfidfFeatureExtractor(
            max_features=baseline_config.get('max_features', 10000),
            pca_dims=None,  # No PCA
            standardize=False,
            ngram_range=tuple(baseline_config.get('ngram_range', [1, 2])),
            min_df=baseline_config.get('min_df', 2),
            max_df=baseline_config.get('max_df', 0.95),
            stop_words=baseline_config.get('stop_words', 'english')
        )
        
        # Extract features for all texts
        all_features, _ = extractor.extract_features(
            self.data_config['jobs_path'],
            self.data_config['programs_path']
        )
        
        # Split features
        train_features = all_features[self.train_indices]
        test_features = all_features[self.test_indices]
        
        # Separate job and program features for retrieval
        job_indices = [i for i, dt in enumerate(self.all_doc_types) if dt == 'JOB']
        program_indices = [i for i, dt in enumerate(self.all_doc_types) if dt == 'PROGRAM']
        
        job_features = all_features[job_indices]
        program_features = all_features[program_indices]
        job_texts = [self.all_texts[i] for i in job_indices]
        program_texts = [self.all_texts[i] for i in program_indices]
        
        # Evaluate all metrics
        metrics = {}
        
        # 1. Retrieval (P@5, MRR)
        retrieval_config = self.eval_config.get('retrieval', {})
        retrieval_evaluator = RetrievalEvaluator(retrieval_config)
        retrieval_metrics = retrieval_evaluator.evaluate_retrieval(
            program_features, job_features, program_texts, job_texts, 'baseline'
        )
        metrics['retrieval'] = retrieval_metrics
        
        # 2. Classification F1
        classifier_config = self.eval_config.get('classification', {})
        # Prepare labels - use doc_type as label (convert to category column format)
        train_labels = self.y_train
        test_labels = self.y_test
        
        # Use sklearn baseline
        baseline_model = JobClassificationBaseline(
            vectorizer_type='tfidf',
            max_features=baseline_config.get('max_features', 10000),
            ngram_range=tuple(baseline_config.get('ngram_range', [1, 2])),
            random_state=self.random_state
        )
        
        # Vectorize texts (using same vectorizer as baseline)
        train_texts_vec = baseline_model.vectorizer.fit_transform(self.X_train)
        test_texts_vec = baseline_model.vectorizer.transform(self.X_test)
        
        # Train models and get best F1
        best_f1 = 0.0
        for name, model in baseline_model.models.items():
            model.fit(train_texts_vec, train_labels)
            y_pred = model.predict(test_texts_vec)
            f1 = f1_score(test_labels, y_pred, average='weighted')
            if f1 > best_f1:
                best_f1 = f1
        
        metrics['classification'] = {
            'f1_weighted': best_f1
        }
        
        # 3. Clustering Silhouette
        clustering_config = self.eval_config.get('clustering', {})
        k_values = clustering_config.get('k_values', [5, 8, 10, 12, 15, 20])
        best_silhouette = -1.0
        best_k = None
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(all_features)
            silhouette = silhouette_score(all_features, cluster_labels)
            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_k = k
        
        metrics['clustering'] = {
            'silhouette_score': best_silhouette,
            'best_k': best_k
        }
        
        # 4. LDA Coherence C_V
        # This requires running LDA - we'll do a simple version
        # For baseline, use existing LDA config if available
        lda_config_existing = self.config.get('lda', {})
        if lda_config_existing:
            # Create minimal LDA config
            lda_config = {
                'preprocessing': {
                    'text_fields': ['description_text'],
                    'bigrams': {
                        'min_count': 5,
                        'threshold': 10
                    },
                    'token_filtering': {
                        'min_frequency': 5,
                        'max_document_frequency': 0.8
                    }
                },
                'lda': {
                    'k_values': [10, 15, 20],
                    'training': {
                        'passes': 5,
                        'iterations': 200,
                        'random_state': self.random_state,
                        'alpha': 'auto',
                        'eta': 'auto'
                    },
                    'evaluation': {
                        'coherence_metrics': ['c_v']
                    }
                }
            }
            
            try:
                trainer = LDATrainer(lda_config)
                texts = trainer.prepare_text_data(pd.DataFrame({'description_text': self.all_texts}))
                tokenized = trainer.tokenize_documents(texts)
                bigrammed = trainer.detect_bigrams(tokenized)
                trainer.build_corpus(bigrammed)
                trainer.train_lda_models()
                
                # Get best C_V coherence
                best_cv = -float('inf')
                for k, scores in trainer.coherence_scores.items():
                    cv_score = scores.get('c_v', -float('inf'))
                    if cv_score > best_cv:
                        best_cv = cv_score
                
                metrics['lda'] = {
                    'c_v_coherence': best_cv if best_cv > -float('inf') else None
                }
            except Exception as e:
                logger.warning(f"LDA baseline evaluation failed: {e}")
                metrics['lda'] = {
                    'c_v_coherence': None
                }
        else:
            metrics['lda'] = {
                'c_v_coherence': None
            }
        
        # Save baseline metrics
        baseline_dir = self.output_dir / "baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_path = baseline_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info("Baseline metrics:")
        logger.info(f"  Retrieval P@5: {retrieval_metrics['precision_at_k']:.4f}")
        logger.info(f"  Retrieval MRR: {retrieval_metrics['mean_reciprocal_rank']:.4f}")
        logger.info(f"  Classification F1: {best_f1:.4f}")
        logger.info(f"  Clustering Silhouette: {best_silhouette:.4f} (k={best_k})")
        if metrics['lda'].get('c_v_coherence'):
            logger.info(f"  LDA C_V: {metrics['lda']['c_v_coherence']:.4f}")
        
        self.baseline_metrics = metrics
        return metrics
    
    def evaluate_features(
        self,
        features: np.ndarray,
        experiment_name: str
    ) -> Dict[str, Any]:
        """Evaluate features using all metrics.
        
        Args:
            features: Feature matrix
            experiment_name: Name of experiment
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # Split features
        train_features = features[self.train_indices]
        test_features = features[self.test_indices]
        
        # Separate job and program features
        job_indices = [i for i, dt in enumerate(self.all_doc_types) if dt == 'JOB']
        program_indices = [i for i, dt in enumerate(self.all_doc_types) if dt == 'PROGRAM']
        
        job_features = features[job_indices]
        program_features = features[program_indices]
        job_texts = [self.all_texts[i] for i in job_indices]
        program_texts = [self.all_texts[i] for i in program_indices]
        
        # 1. Retrieval
        retrieval_config = self.eval_config.get('retrieval', {})
        retrieval_evaluator = RetrievalEvaluator(retrieval_config)
        retrieval_metrics = retrieval_evaluator.evaluate_retrieval(
            program_features, job_features, program_texts, job_texts, experiment_name
        )
        metrics['retrieval'] = retrieval_metrics
        
        # 2. Classification F1
        # Note: For experiments, we use pre-extracted features directly
        # So we need to train on train_features
        train_labels = self.y_train
        test_labels = self.y_test
        
        # Use simple classifier on features
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        
        best_f1 = 0.0
        for model_name, model in [
            ('lr', LogisticRegression(random_state=self.random_state, max_iter=1000)),
            ('rf', RandomForestClassifier(random_state=self.random_state, n_estimators=100))
        ]:
            model.fit(train_features, train_labels)
            y_pred = model.predict(test_features)
            f1 = f1_score(test_labels, y_pred, average='weighted')
            if f1 > best_f1:
                best_f1 = f1
        
        metrics['classification'] = {
            'f1_weighted': best_f1
        }
        
        # 3. Clustering
        clustering_config = self.eval_config.get('clustering', {})
        k_values = clustering_config.get('k_values', [5, 8, 10, 12, 15, 20])
        best_silhouette = -1.0
        best_k = None
        
        for k in k_values:
            try:
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                cluster_labels = kmeans.fit_predict(features)
                silhouette = silhouette_score(features, cluster_labels)
                if silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_k = k
            except Exception as e:
                logger.warning(f"Clustering evaluation failed for k={k}: {e}")
                continue
        
        metrics['clustering'] = {
            'silhouette_score': best_silhouette,
            'best_k': best_k
        }
        
        # 4. LDA - skip for now (requires text processing)
        # Will be handled separately for phrase experiments
        metrics['lda'] = {
            'c_v_coherence': None
        }
        
        return metrics
    
    def run_all_experiments(self) -> pd.DataFrame:
        """Run all experiments and collect results.
        
        Returns:
            DataFrame with all experiment results
        """
        logger.info("=" * 80)
        logger.info("RUNNING ALL EXPERIMENTS")
        logger.info("=" * 80)
        
        all_results = []
        
        # 1. TF-IDF word n-grams
        logger.info("\n" + "=" * 80)
        logger.info("TF-IDF Word N-grams Experiments")
        logger.info("=" * 80)
        tfidf_dir = self.output_dir / "tfidf_word_ngrams"
        tfidf_experiment = TfidfNgramExperiment(self.config)
        tfidf_summary = tfidf_experiment.run_all_experiments(
            self.all_texts, self.all_doc_types, tfidf_dir,
            train_indices=self.train_indices, test_indices=self.test_indices
        )
        
        # Evaluate each TF-IDF experiment
        for _, row in tfidf_summary.iterrows():
            features = np.load(row['features_path'])
            metrics = self.evaluate_features(features, row['experiment_id'])
            
            # Calculate relative gains
            rel_gains = {}
            if self.baseline_metrics:
                baseline_retrieval = self.baseline_metrics.get('retrieval', {})
                baseline_f1 = self.baseline_metrics.get('classification', {}).get('f1_weighted', 0)
                baseline_silhouette = self.baseline_metrics.get('clustering', {}).get('silhouette_score', 0)
                
                retrieval_p5 = metrics['retrieval']['precision_at_k']
                baseline_p5 = baseline_retrieval.get('precision_at_k', 0)
                if baseline_p5 > 0:
                    rel_gains['retrieval_p5'] = ((retrieval_p5 - baseline_p5) / baseline_p5) * 100
                
                retrieval_mrr = metrics['retrieval']['mean_reciprocal_rank']
                baseline_mrr = baseline_retrieval.get('mean_reciprocal_rank', 0)
                if baseline_mrr > 0:
                    rel_gains['retrieval_mrr'] = ((retrieval_mrr - baseline_mrr) / baseline_mrr) * 100
                
                f1 = metrics['classification']['f1_weighted']
                if baseline_f1 > 0:
                    rel_gains['classification_f1'] = ((f1 - baseline_f1) / baseline_f1) * 100
                
                silhouette = metrics['clustering']['silhouette_score']
                if baseline_silhouette > 0:
                    rel_gains['clustering_silhouette'] = ((silhouette - baseline_silhouette) / baseline_silhouette) * 100
            
            result = {
                'experiment_type': 'tfidf_word_ngrams',
                'experiment_id': row['experiment_id'],
                **row.to_dict(),
                **metrics,
                'relative_gains': rel_gains
            }
            all_results.append(result)
            
            # Save metrics to experiment directory
            exp_dir = Path(row['features_path']).parent
            metrics_path = exp_dir / "metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump({**metrics, 'relative_gains': rel_gains}, f, indent=2)
        
        # 2. Character n-grams
        logger.info("\n" + "=" * 80)
        logger.info("Character N-grams Experiments")
        logger.info("=" * 80)
        char_dir = self.output_dir / "char_ngrams"
        char_experiment = CharNGramExperiment(self.config)
        char_summary = char_experiment.run_all_experiments(
            self.all_texts, self.all_doc_types, char_dir,
            train_indices=self.train_indices, test_indices=self.test_indices
        )
        
        # Evaluate each character n-gram experiment
        for _, row in char_summary.iterrows():
            features = np.load(row['features_path'])
            metrics = self.evaluate_features(features, row['experiment_id'])
            
            # Calculate relative gains
            rel_gains = {}
            if self.baseline_metrics:
                baseline_retrieval = self.baseline_metrics.get('retrieval', {})
                baseline_f1 = self.baseline_metrics.get('classification', {}).get('f1_weighted', 0)
                baseline_silhouette = self.baseline_metrics.get('clustering', {}).get('silhouette_score', 0)
                
                retrieval_p5 = metrics['retrieval']['precision_at_k']
                baseline_p5 = baseline_retrieval.get('precision_at_k', 0)
                if baseline_p5 > 0:
                    rel_gains['retrieval_p5'] = ((retrieval_p5 - baseline_p5) / baseline_p5) * 100
                
                retrieval_mrr = metrics['retrieval']['mean_reciprocal_rank']
                baseline_mrr = baseline_retrieval.get('mean_reciprocal_rank', 0)
                if baseline_mrr > 0:
                    rel_gains['retrieval_mrr'] = ((retrieval_mrr - baseline_mrr) / baseline_mrr) * 100
                
                f1 = metrics['classification']['f1_weighted']
                if baseline_f1 > 0:
                    rel_gains['classification_f1'] = ((f1 - baseline_f1) / baseline_f1) * 100
                
                silhouette = metrics['clustering']['silhouette_score']
                if baseline_silhouette > 0:
                    rel_gains['clustering_silhouette'] = ((silhouette - baseline_silhouette) / baseline_silhouette) * 100
            
            result = {
                'experiment_type': 'char_ngrams',
                'experiment_id': row['experiment_id'],
                **row.to_dict(),
                **metrics,
                'relative_gains': rel_gains
            }
            all_results.append(result)
            
            # Save metrics
            exp_dir = Path(row['features_path']).parent
            metrics_path = exp_dir / "metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump({**metrics, 'relative_gains': rel_gains}, f, indent=2)
        
        # 3. Phrase modeling (TF-IDF only for now, LDA handled separately)
        logger.info("\n" + "=" * 80)
        logger.info("Phrase Modeling Experiments")
        logger.info("=" * 80)
        phrase_dir = self.output_dir / "phrase_modeling"
        
        # Get TF-IDF config for phrase experiments
        tfidf_config_for_phrases = {
            'max_features': 20000,
            'min_df': 2,
            'max_df': 0.95,
            'ngram_range': (1, 1),
            'stop_words': 'english'
        }
        
        phrase_experiment = PhraseExperiment(self.config)
        phrase_summary = phrase_experiment.run_all_experiments(
            self.tokenized_docs, self.all_texts, self.all_doc_types,
            phrase_dir, tfidf_config_for_phrases,
            train_indices=self.train_indices, test_indices=self.test_indices
        )
        
        # Evaluate TF-IDF phrase experiments
        for _, row in phrase_summary.iterrows():
            if (row['application'] == 'tfidf' and 
                'features_path' in row and 
                pd.notna(row['features_path']) and
                Path(row['features_path']).exists()):
                features = np.load(row['features_path'])
                metrics = self.evaluate_features(features, row['experiment_id'])
                
                # Calculate relative gains
                rel_gains = {}
                if self.baseline_metrics:
                    baseline_retrieval = self.baseline_metrics.get('retrieval', {})
                    baseline_f1 = self.baseline_metrics.get('classification', {}).get('f1_weighted', 0)
                    baseline_silhouette = self.baseline_metrics.get('clustering', {}).get('silhouette_score', 0)
                    
                    retrieval_p5 = metrics['retrieval']['precision_at_k']
                    baseline_p5 = baseline_retrieval.get('precision_at_k', 0)
                    if baseline_p5 > 0:
                        rel_gains['retrieval_p5'] = ((retrieval_p5 - baseline_p5) / baseline_p5) * 100
                    
                    retrieval_mrr = metrics['retrieval']['mean_reciprocal_rank']
                    baseline_mrr = baseline_retrieval.get('mean_reciprocal_rank', 0)
                    if baseline_mrr > 0:
                        rel_gains['retrieval_mrr'] = ((retrieval_mrr - baseline_mrr) / baseline_mrr) * 100
                    
                    f1 = metrics['classification']['f1_weighted']
                    if baseline_f1 > 0:
                        rel_gains['classification_f1'] = ((f1 - baseline_f1) / baseline_f1) * 100
                    
                    silhouette = metrics['clustering']['silhouette_score']
                    if baseline_silhouette > 0:
                        rel_gains['clustering_silhouette'] = ((silhouette - baseline_silhouette) / baseline_silhouette) * 100
                
                result = {
                    'experiment_type': 'phrase_modeling',
                    'experiment_id': row['experiment_id'],
                    **row.to_dict(),
                    **metrics,
                    'relative_gains': rel_gains
                }
                all_results.append(result)
                
                # Save metrics
                exp_dir = Path(row['features_path']).parent
                metrics_path = exp_dir / "metrics.json"
                with open(metrics_path, 'w') as f:
                    json.dump({**metrics, 'relative_gains': rel_gains}, f, indent=2)
        
        # 4. Feature Fusion experiments
        logger.info("\n" + "=" * 80)
        logger.info("Feature Fusion Experiments")
        logger.info("=" * 80)
        fusion_dir = self.output_dir / "feature_fusion"
        fusion_experiment = FeatureFusionExperiment(self.config)
        fusion_summary = fusion_experiment.run_all_experiments(
            self.all_texts, self.tokenized_docs, self.all_doc_types,
            fusion_dir,
            train_indices=self.train_indices, test_indices=self.test_indices
        )
        
        # Evaluate each feature fusion experiment
        for _, row in fusion_summary.iterrows():
            features = np.load(row['features_path'])
            metrics = self.evaluate_features(features, row['experiment_id'])
            
            # Calculate relative gains
            rel_gains = {}
            if self.baseline_metrics:
                baseline_retrieval = self.baseline_metrics.get('retrieval', {})
                baseline_f1 = self.baseline_metrics.get('classification', {}).get('f1_weighted', 0)
                baseline_silhouette = self.baseline_metrics.get('clustering', {}).get('silhouette_score', 0)
                
                retrieval_p5 = metrics['retrieval']['precision_at_k']
                baseline_p5 = baseline_retrieval.get('precision_at_k', 0)
                if baseline_p5 > 0:
                    rel_gains['retrieval_p5'] = ((retrieval_p5 - baseline_p5) / baseline_p5) * 100
                
                retrieval_mrr = metrics['retrieval']['mean_reciprocal_rank']
                baseline_mrr = baseline_retrieval.get('mean_reciprocal_rank', 0)
                if baseline_mrr > 0:
                    rel_gains['retrieval_mrr'] = ((retrieval_mrr - baseline_mrr) / baseline_mrr) * 100
                
                f1 = metrics['classification']['f1_weighted']
                if baseline_f1 > 0:
                    rel_gains['classification_f1'] = ((f1 - baseline_f1) / baseline_f1) * 100
                
                silhouette = metrics['clustering']['silhouette_score']
                if baseline_silhouette > 0:
                    rel_gains['clustering_silhouette'] = ((silhouette - baseline_silhouette) / baseline_silhouette) * 100
            
            result = {
                'experiment_type': 'feature_fusion',
                'experiment_id': row['experiment_id'],
                **row.to_dict(),
                **metrics,
                'relative_gains': rel_gains
            }
            all_results.append(result)
            
            # Save metrics
            exp_dir = Path(row['features_path']).parent
            metrics_path = exp_dir / "metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump({**metrics, 'relative_gains': rel_gains}, f, indent=2)
        
        # Create summary DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Save comprehensive summary
        summary_path = self.output_dir / "experiment_summary.csv"
        results_df.to_csv(summary_path, index=False)
        logger.info(f"Saved experiment summary to {summary_path}")
        
        return results_df
    
    def generate_report(self, results_df: pd.DataFrame) -> str:
        """Generate experiment report.
        
        Args:
            results_df: DataFrame with all results
            
        Returns:
            Report string
        """
        report_lines = []
        report_lines.append("# N-gram Experiment Report\n")
        report_lines.append(f"Generated: {pd.Timestamp.now()}\n")
        report_lines.append("\n")
        
        # Baseline summary
        report_lines.append("## Baseline Metrics\n")
        if self.baseline_metrics:
            baseline_retrieval = self.baseline_metrics.get('retrieval', {})
            baseline_class = self.baseline_metrics.get('classification', {})
            baseline_cluster = self.baseline_metrics.get('clustering', {})
            baseline_lda = self.baseline_metrics.get('lda', {})
            
            report_lines.append(f"- **Retrieval P@5**: {baseline_retrieval.get('precision_at_k', 0):.4f}\n")
            report_lines.append(f"- **Retrieval MRR**: {baseline_retrieval.get('mean_reciprocal_rank', 0):.4f}\n")
            report_lines.append(f"- **Classification F1**: {baseline_class.get('f1_weighted', 0):.4f}\n")
            report_lines.append(f"- **Clustering Silhouette**: {baseline_cluster.get('silhouette_score', 0):.4f}\n")
            if baseline_lda.get('c_v_coherence'):
                report_lines.append(f"- **LDA C_V**: {baseline_lda.get('c_v_coherence', 0):.4f}\n")
        report_lines.append("\n")
        
        # Best results per metric
        report_lines.append("## Best Results per Metric\n")
        
        if len(results_df) > 0:
            # Best retrieval P@5
            best_p5 = results_df.loc[results_df['retrieval'].apply(lambda x: x.get('precision_at_k', 0)).idxmax()]
            report_lines.append(f"### Best Retrieval P@5\n")
            report_lines.append(f"- **Experiment**: {best_p5['experiment_id']}\n")
            report_lines.append(f"- **P@5**: {best_p5['retrieval']['precision_at_k']:.4f}\n")
            if best_p5.get('relative_gains'):
                report_lines.append(f"- **Relative Gain**: {best_p5['relative_gains'].get('retrieval_p5', 0):.2f}%\n")
            report_lines.append("\n")
            
            # Best F1
            best_f1 = results_df.loc[results_df['classification'].apply(lambda x: x.get('f1_weighted', 0)).idxmax()]
            report_lines.append(f"### Best Classification F1\n")
            report_lines.append(f"- **Experiment**: {best_f1['experiment_id']}\n")
            report_lines.append(f"- **F1**: {best_f1['classification']['f1_weighted']:.4f}\n")
            if best_f1.get('relative_gains'):
                report_lines.append(f"- **Relative Gain**: {best_f1['relative_gains'].get('classification_f1', 0):.2f}%\n")
            report_lines.append("\n")
            
            # Best Silhouette
            best_sil = results_df.loc[results_df['clustering'].apply(lambda x: x.get('silhouette_score', -1)).idxmax()]
            report_lines.append(f"### Best Clustering Silhouette\n")
            report_lines.append(f"- **Experiment**: {best_sil['experiment_id']}\n")
            report_lines.append(f"- **Silhouette**: {best_sil['clustering']['silhouette_score']:.4f}\n")
            if best_sil.get('relative_gains'):
                report_lines.append(f"- **Relative Gain**: {best_sil['relative_gains'].get('clustering_silhouette', 0):.2f}%\n")
            report_lines.append("\n")
        
        report_text = "".join(report_lines)
        
        # Save report
        report_path = self.output_dir / "experiment_report.md"
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Generated report: {report_path}")
        
        return report_text

