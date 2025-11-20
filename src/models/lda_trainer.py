"""
LDA Topic Modeling Trainer.

This module implements the LDATrainer class for building document-term matrices,
training LDA models, and calculating coherence metrics.
"""

import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import gensim
from gensim import corpora, models
from gensim.models import LdaModel, CoherenceModel
from gensim.models.phrases import Phrases, Phraser
from gensim.models.phrases import Phrases, Phraser
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import yaml

logger = logging.getLogger(__name__)


class LDATrainer:
    """Trainer class for LDA topic modeling with bigram detection and coherence evaluation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize LDA trainer with configuration.
        
        Args:
            config: Configuration dictionary from YAML file
        """
        self.config = config
        self.preprocessing_config = config['preprocessing']
        self.lda_config = config['lda']
        
        # Initialize components
        self.phrases_model = None
        self.dictionary = None
        self.corpus = None
        self.tfidf_corpus = None
        self.tfidf_model = None
        self.lda_models = {}
        self.coherence_scores = {}
        self.tokenized_docs = None  # Store original tokenized docs for coherence
        
        logger.info("Initialized LDA trainer")
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load job postings data.
        
        Args:
            data_path: Path to parquet file containing job data
            
        Returns:
            DataFrame with job postings
        """
        logger.info(f"Loading data from {data_path}")
        df = pd.read_parquet(data_path)
        logger.info(f"Loaded {len(df)} job postings")
        
        # Check required columns
        required_cols = ['title', 'description']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return df
    
    def prepare_text_data(self, df: pd.DataFrame) -> List[str]:
        """Prepare text data by combining multiple fields.
        
        Args:
            df: DataFrame with job postings
            
        Returns:
            List of combined text strings
        """
        logger.info("Preparing text data for LDA")
        
        # Check if tokenized versions exist
        tokenized_fields = self.preprocessing_config.get('tokenized_fields', [])
        available_tokenized = [col for col in tokenized_fields if col in df.columns]
        
        if available_tokenized:
            logger.info(f"Using existing tokenized fields: {available_tokenized}")
            # Use the first available tokenized field
            tokenized_col = available_tokenized[0]
            texts = df[tokenized_col].dropna().tolist()
            
            # Convert token lists to strings if needed
            if texts and isinstance(texts[0], list):
                texts = [' '.join(tokens) for tokens in texts]
        else:
            logger.info("No tokenized fields found, combining raw text fields")
            text_fields = self.preprocessing_config.get('text_fields', ['title', 'description'])
            
            # Combine text fields
            texts = []
            for _, row in df.iterrows():
                combined_text = []
                for field in text_fields:
                    if field in df.columns and pd.notna(row[field]):
                        combined_text.append(str(row[field]))
                
                if combined_text:
                    texts.append(' '.join(combined_text))
        
        logger.info(f"Prepared {len(texts)} text documents")
        return texts
    
    def tokenize_documents(self, texts: List[str]) -> List[List[str]]:
        """Tokenize documents into word lists.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of tokenized documents
        """
        logger.info("Tokenizing documents")
        
        # Simple tokenization (can be enhanced with spaCy if needed)
        tokenized_docs = []
        for text in texts:
            # Basic tokenization: split on whitespace and punctuation
            tokens = gensim.utils.simple_preprocess(text, deacc=True, min_len=2, max_len=50)
            tokenized_docs.append(tokens)
        
        logger.info(f"Tokenized {len(tokenized_docs)} documents")
        return tokenized_docs
    
    def detect_bigrams(self, tokenized_docs: List[List[str]]) -> List[List[str]]:
        """Detect and apply bigrams to tokenized documents.
        
        Args:
            tokenized_docs: List of tokenized documents
            
        Returns:
            List of documents with bigrams applied
        """
        logger.info("Detecting bigrams")
        
        bigram_config = self.preprocessing_config['bigrams']
        min_count = bigram_config['min_count']
        threshold = bigram_config['threshold']
        
        # Train bigram model
        self.phrases_model = Phrases(
            tokenized_docs,
            min_count=min_count,
            threshold=threshold,
            delimiter='_'
        )
        
        # Apply bigrams
        bigram_docs = [self.phrases_model[doc] for doc in tokenized_docs]
        
        # Store tokenized docs for coherence calculation
        self.tokenized_docs = bigram_docs
        
        logger.info(f"Applied bigrams to {len(bigram_docs)} documents")
        
        # Log sample bigrams (if available)
        try:
            sample_bigrams = list(self.phrases_model.phrasegrams.keys())[:10]
            logger.info(f"Sample bigrams: {sample_bigrams}")
        except AttributeError:
            logger.info("Bigram phrases model created successfully")
        
        return bigram_docs
    
    def detect_hierarchical_phrases(
        self, 
        tokenized_docs: List[List[str]],
        bigram_min_count: int = 5,
        bigram_threshold: int = 10,
        trigram_min_count: Optional[int] = None,
        trigram_threshold: Optional[int] = None
    ) -> Tuple[List[List[str]], Phrases, Optional[Phrases]]:
        """Detect and apply hierarchical bigrams and trigrams.
        
        Args:
            tokenized_docs: List of tokenized documents
            bigram_min_count: Minimum count for bigram phrases
            bigram_threshold: Threshold for bigram phrases
            trigram_min_count: Minimum count for trigram phrases (None to skip trigrams)
            trigram_threshold: Threshold for trigram phrases (None to skip trigrams)
            
        Returns:
            Tuple of (phrased_docs, bigram_model, trigram_model_or_None)
        """
        logger.info("Detecting hierarchical phrases (bigrams → trigrams)")
        
        # Step 1: Detect and apply bigrams
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
        
        # Step 2: Detect and apply trigrams on bigrammed documents (if requested)
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
            self.tokenized_docs = final_docs
        else:
            final_docs = bigram_docs
            self.tokenized_docs = final_docs
        
        # Store phrases models
        self.phrases_model = bigram_model  # Store bigram model for compatibility
        
        return final_docs, bigram_model, trigram_model
    
    def build_corpus(self, bigram_docs: List[List[str]]) -> None:
        """Build document-term matrix and corpus.
        
        Args:
            bigram_docs: List of documents with bigrams applied
        """
        logger.info("Building document-term matrix")
        
        # Create dictionary
        self.dictionary = corpora.Dictionary(bigram_docs)
        
        # Filter dictionary
        filter_config = self.preprocessing_config['token_filtering']
        min_freq = filter_config['min_frequency']
        max_df = filter_config['max_document_frequency']
        
        # Remove very rare and very common words
        self.dictionary.filter_extremes(
            no_below=min_freq,
            no_above=max_df,
            keep_n=None
        )
        
        # Build corpus
        self.corpus = [self.dictionary.doc2bow(doc) for doc in bigram_docs]
        
        # Create TF-IDF corpus for labeling
        self.tfidf_model = models.TfidfModel(self.corpus)
        self.tfidf_corpus = self.tfidf_model[self.corpus]
        
        logger.info(f"Built corpus with {len(self.dictionary)} unique terms")
        logger.info(f"Corpus size: {len(self.corpus)} documents")
        logger.info(f"Average document length: {np.mean([len(doc) for doc in self.corpus]):.1f} terms")
    
    def train_lda_models(self) -> None:
        """Train LDA models for all k values."""
        logger.info("Training LDA models")
        
        k_values = self.lda_config['k_values']
        training_config = self.lda_config['training']
        
        for k in k_values:
            logger.info(f"Training LDA model with k={k}")
            
            # Train LDA model
            lda_model = LdaModel(
                corpus=self.corpus,
                id2word=self.dictionary,
                num_topics=k,
                passes=training_config['passes'],
                iterations=training_config['iterations'],
                random_state=training_config['random_state'],
                alpha=training_config['alpha'],
                eta=training_config['eta']
            )
            
            self.lda_models[k] = lda_model
            
            # Calculate coherence
            coherence_config = self.lda_config['evaluation']
            coherence_metrics = coherence_config['coherence_metrics']
            
            coherence_scores = {}
            for metric in coherence_metrics:
                coherence_model = CoherenceModel(
                    model=lda_model,
                    texts=self._get_texts_for_coherence(),
                    dictionary=self.dictionary,
                    coherence=metric
                )
                coherence_scores[metric] = coherence_model.get_coherence()
            
            self.coherence_scores[k] = coherence_scores
            
            logger.info(f"LDA k={k} coherence: {coherence_scores}")
    
    def _get_texts_for_coherence(self) -> List[List[str]]:
        """Get tokenized texts for coherence calculation."""
        # Return the stored tokenized documents
        return self.tokenized_docs
    
    def get_document_topic_distributions(self, k: int) -> np.ndarray:
        """Get document-topic distributions for a specific k.
        
        Args:
            k: Number of topics
            
        Returns:
            Document-topic distribution matrix (n_docs × k)
        """
        if k not in self.lda_models:
            raise ValueError(f"No LDA model found for k={k}")
        
        lda_model = self.lda_models[k]
        doc_topic_dist = []
        
        for doc in self.corpus:
            topic_dist = lda_model.get_document_topics(doc, minimum_probability=0.0)
            # Convert to array format
            topic_probs = [prob for _, prob in topic_dist]
            doc_topic_dist.append(topic_probs)
        
        return np.array(doc_topic_dist)
    
    def get_top_words_per_topic(self, k: int, num_words: int = 15) -> Dict[int, List[Tuple[str, float]]]:
        """Get top words for each topic.
        
        Args:
            k: Number of topics
            num_words: Number of top words per topic
            
        Returns:
            Dictionary mapping topic_id to list of (word, probability) tuples
        """
        if k not in self.lda_models:
            raise ValueError(f"No LDA model found for k={k}")
        
        lda_model = self.lda_models[k]
        top_words = {}
        
        for topic_id in range(k):
            topic_words = lda_model.show_topic(topic_id, topn=num_words)
            top_words[topic_id] = topic_words
        
        return top_words
    
    def get_exemplar_documents(self, k: int, num_exemplars: int = 5) -> Dict[int, List[int]]:
        """Get exemplar documents for each topic.
        
        Args:
            k: Number of topics
            num_exemplars: Number of exemplar documents per topic
            
        Returns:
            Dictionary mapping topic_id to list of document indices
        """
        doc_topic_dist = self.get_document_topic_distributions(k)
        exemplars = {}
        
        for topic_id in range(k):
            # Get documents with highest probability for this topic
            topic_probs = doc_topic_dist[:, topic_id]
            top_doc_indices = np.argsort(topic_probs)[-num_exemplars:][::-1]
            exemplars[topic_id] = top_doc_indices.tolist()
        
        return exemplars
    
    def save_model(self, k: int, output_dir: Path) -> None:
        """Save LDA model and related artifacts.
        
        Args:
            k: Number of topics
            output_dir: Output directory
        """
        if k not in self.lda_models:
            raise ValueError(f"No LDA model found for k={k}")
        
        # Create k-specific directory
        k_dir = output_dir / f"k{k}"
        k_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = k_dir / "lda_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.lda_models[k], f)
        
        # Save dictionary
        dict_path = k_dir / "dictionary.pkl"
        with open(dict_path, 'wb') as f:
            pickle.dump(self.dictionary, f)
        
        # Save phrases model
        if self.phrases_model:
            phrases_path = k_dir / "phrases_model.pkl"
            with open(phrases_path, 'wb') as f:
                pickle.dump(self.phrases_model, f)
        
        logger.info(f"Saved LDA model for k={k} to {k_dir}")
    
    def get_coherence_summary(self) -> pd.DataFrame:
        """Get coherence scores summary as DataFrame.
        
        Returns:
            DataFrame with coherence scores for all k values
        """
        rows = []
        for k, scores in self.coherence_scores.items():
            row = {'k': k}
            row.update(scores)
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def calculate_perplexity(self, k: int) -> float:
        """Calculate perplexity for a specific k.
        
        Args:
            k: Number of topics
            
        Returns:
            Perplexity score
        """
        if k not in self.lda_models:
            raise ValueError(f"No LDA model found for k={k}")
        
        lda_model = self.lda_models[k]
        return lda_model.log_perplexity(self.corpus)
