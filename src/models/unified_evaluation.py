"""
Unified Evaluation and Visualization Pipeline

This module provides comprehensive evaluation of different embedding approaches
(Word2Vec, Doc2Vec, Transformers) across intrinsic and extrinsic metrics,
with rich visualizations for analysis.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.models import Word2Vec, Doc2Vec
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import umap
from sklearn.manifold import TSNE

from ..utils import (
    load_word2vec_model,
    load_doc2vec_model,
    load_transformer_embeddings,
    load_embeddings_metadata,
    load_clustering_results,
    load_lda_model,
    get_document_embeddings_word2vec,
    get_document_embeddings_doc2vec,
    create_output_directory,
)

logger = logging.getLogger(__name__)


class IntrinsicEvaluator:
    """Evaluates intrinsic quality of embeddings through neighborhood analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the intrinsic evaluator.
        
        Args:
            config: Configuration dictionary with evaluation parameters
        """
        self.config = config
        self.test_terms = config.get('test_terms', [])
        self.n_neighbors = config.get('n_neighbors', 10)
    
    def evaluate_word2vec_neighborhoods(
        self, 
        model: Word2Vec, 
        model_name: str
    ) -> Dict[str, Any]:
        """Evaluate Word2Vec neighborhood quality for domain terms.
        
        Args:
            model: Trained Word2Vec model
            model_name: Name of the model for reporting
            
        Returns:
            Dictionary with neighborhood analysis results
        """
        results = {
            'model_name': model_name,
            'neighborhoods': {},
            'avg_similarity': 0.0,
            'coherent_terms': 0
        }
        
        similarities = []
        coherent_count = 0
        
        for term in self.test_terms:
            if term in model.wv:
                # Get most similar words
                similar_words = model.wv.most_similar(term, topn=self.n_neighbors)
                
                # Calculate average similarity
                avg_sim = np.mean([sim for _, sim in similar_words])
                similarities.append(avg_sim)
                
                # Check if neighbors are semantically coherent
                # (simple heuristic: average similarity > 0.3)
                is_coherent = avg_sim > 0.3
                if is_coherent:
                    coherent_count += 1
                
                results['neighborhoods'][term] = {
                    'similar_words': similar_words,
                    'avg_similarity': avg_sim,
                    'is_coherent': is_coherent
                }
            else:
                results['neighborhoods'][term] = {
                    'similar_words': [],
                    'avg_similarity': 0.0,
                    'is_coherent': False
                }
        
        results['avg_similarity'] = np.mean(similarities) if similarities else 0.0
        results['coherent_terms'] = coherent_count
        
        return results
    
    def evaluate_cluster_purity(
        self, 
        embeddings: np.ndarray, 
        cluster_labels: np.ndarray,
        embedding_name: str
    ) -> Dict[str, Any]:
        """Evaluate cluster neighborhood purity.
        
        Args:
            embeddings: Document embeddings
            cluster_labels: Cluster assignments
            embedding_name: Name of the embedding method
            
        Returns:
            Dictionary with cluster purity metrics
        """
        # Use k-nearest neighbors to check cluster purity
        n_neighbors = min(self.n_neighbors, len(embeddings) - 1)
        nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
        nn.fit(embeddings)
        
        purities = []
        
        for i, embedding in enumerate(embeddings):
            # Get nearest neighbors (excluding self)
            distances, indices = nn.kneighbors([embedding])
            neighbor_indices = indices[0][1:]  # Exclude self
            
            # Get cluster labels of neighbors
            neighbor_clusters = cluster_labels[neighbor_indices]
            current_cluster = cluster_labels[i]
            
            # Calculate purity (fraction of neighbors in same cluster)
            purity = np.mean(neighbor_clusters == current_cluster)
            purities.append(purity)
        
        return {
            'embedding_name': embedding_name,
            'avg_purity': np.mean(purities),
            'std_purity': np.std(purities),
            'min_purity': np.min(purities),
            'max_purity': np.max(purities)
        }


class RetrievalEvaluator:
    """Evaluates embedding quality through program-to-job retrieval task."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the retrieval evaluator.
        
        Args:
            config: Configuration dictionary with evaluation parameters
        """
        self.config = config
        self.k = config.get('k', 5)
        self.metrics = config.get('metrics', ['precision_at_k'])
    
    def evaluate_retrieval(
        self,
        program_embeddings: np.ndarray,
        job_embeddings: np.ndarray,
        program_texts: List[str],
        job_texts: List[str],
        embedding_name: str
    ) -> Dict[str, Any]:
        """Evaluate program-to-job retrieval performance.
        
        Args:
            program_embeddings: Embeddings for programs
            job_embeddings: Embeddings for jobs
            program_texts: Text descriptions of programs
            job_texts: Text descriptions of jobs
            embedding_name: Name of the embedding method
            
        Returns:
            Dictionary with retrieval metrics
        """
        # Calculate cosine similarities
        similarities = cosine_similarity(program_embeddings, job_embeddings)
        
        precision_scores = []
        mrr_scores = []
        
        for i, program_text in enumerate(program_texts):
            # Get top-k most similar jobs
            top_k_indices = np.argsort(similarities[i])[-self.k:][::-1]
            top_k_jobs = [job_texts[idx] for idx in top_k_indices]
            
            # Calculate semantic relevance (simple keyword overlap)
            precision = self._calculate_semantic_precision(program_text, top_k_jobs)
            precision_scores.append(precision)
            
            # Calculate Mean Reciprocal Rank
            mrr = self._calculate_mrr(program_text, top_k_jobs)
            mrr_scores.append(mrr)
        
        return {
            'embedding_name': embedding_name,
            'precision_at_k': np.mean(precision_scores),
            'std_precision': np.std(precision_scores),
            'mean_reciprocal_rank': np.mean(mrr_scores),
            'std_mrr': np.std(mrr_scores),
            'num_programs': len(program_texts),
            'num_jobs': len(job_texts)
        }
    
    def _calculate_semantic_precision(
        self, 
        program_text: str, 
        job_texts: List[str]
    ) -> float:
        """Calculate semantic precision based on keyword overlap.
        
        Args:
            program_text: Program description
            job_texts: List of job descriptions
            
        Returns:
            Precision score (0-1)
        """
        # Simple keyword-based relevance
        program_words = set(program_text.lower().split())
        relevant_jobs = 0
        
        for job_text in job_texts:
            job_words = set(job_text.lower().split())
            # Calculate Jaccard similarity
            intersection = len(program_words & job_words)
            union = len(program_words | job_words)
            
            if union > 0:
                jaccard = intersection / union
                if jaccard > 0.1:  # Threshold for relevance
                    relevant_jobs += 1
        
        return relevant_jobs / len(job_texts)
    
    def _calculate_mrr(
        self, 
        program_text: str, 
        job_texts: List[str]
    ) -> float:
        """Calculate Mean Reciprocal Rank.
        
        Args:
            program_text: Program description
            job_texts: List of job descriptions (ordered by similarity)
            
        Returns:
            MRR score
        """
        program_words = set(program_text.lower().split())
        
        for rank, job_text in enumerate(job_texts, 1):
            job_words = set(job_text.lower().split())
            intersection = len(program_words & job_words)
            union = len(program_words | job_words)
            
            if union > 0:
                jaccard = intersection / union
                if jaccard > 0.1:  # Threshold for relevance
                    return 1.0 / rank
        
        return 0.0


class ClusteringEvaluator:
    """Evaluates clustering quality using various metrics."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the clustering evaluator.
        
        Args:
            config: Configuration dictionary with evaluation parameters
        """
        self.config = config
        self.algorithms = config.get('algorithms', ['kmeans'])
        self.k_values = config.get('k_values', [5, 8, 10, 12, 15, 20])
    
    def evaluate_clustering_quality(
        self,
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        embedding_name: str,
        algorithm: str
    ) -> Dict[str, Any]:
        """Evaluate clustering quality metrics.
        
        Args:
            embeddings: Document embeddings
            cluster_labels: Cluster assignments
            embedding_name: Name of the embedding method
            algorithm: Name of the clustering algorithm
            
        Returns:
            Dictionary with clustering metrics
        """
        n_clusters = len(np.unique(cluster_labels))
        
        # Skip if too few clusters or too many clusters
        if n_clusters < 2 or n_clusters >= len(embeddings):
            return {
                'embedding_name': embedding_name,
                'algorithm': algorithm,
                'n_clusters': n_clusters,
                'silhouette_score': 0.0,
                'davies_bouldin_score': float('inf'),
                'calinski_harabasz_score': 0.0,
                'valid': False
            }
        
        try:
            # Calculate clustering metrics
            silhouette = silhouette_score(embeddings, cluster_labels)
            davies_bouldin = davies_bouldin_score(embeddings, cluster_labels)
            calinski_harabasz = calinski_harabasz_score(embeddings, cluster_labels)
            
            return {
                'embedding_name': embedding_name,
                'algorithm': algorithm,
                'n_clusters': n_clusters,
                'silhouette_score': silhouette,
                'davies_bouldin_score': davies_bouldin,
                'calinski_harabasz_score': calinski_harabasz,
                'valid': True
            }
        except Exception as e:
            logger.warning(f"Error calculating clustering metrics: {e}")
            return {
                'embedding_name': embedding_name,
                'algorithm': algorithm,
                'n_clusters': n_clusters,
                'silhouette_score': 0.0,
                'davies_bouldin_score': float('inf'),
                'calinski_harabasz_score': 0.0,
                'valid': False
            }
    
    def analyze_cluster_quality(
        self,
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        texts: List[str],
        embedding_name: str
    ) -> Dict[str, Any]:
        """Analyze cluster quality with qualitative metrics.
        
        Args:
            embeddings: Document embeddings
            cluster_labels: Cluster assignments
            texts: Document texts
            embedding_name: Name of the embedding method
            
        Returns:
            Dictionary with cluster analysis results
        """
        n_clusters = len(np.unique(cluster_labels))
        cluster_profiles = {}
        
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_texts = [texts[i] for i in range(len(texts)) if cluster_mask[i]]
            
            if not cluster_texts:
                continue
            
            # Extract top terms using TF-IDF
            vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
            try:
                tfidf_matrix = vectorizer.fit_transform(cluster_texts)
                feature_names = vectorizer.get_feature_names_out()
                
                # Get top terms
                mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
                top_indices = np.argsort(mean_scores)[-10:][::-1]
                top_terms = [feature_names[i] for i in top_indices]
                top_scores = [mean_scores[i] for i in top_indices]
                
                # Select exemplar documents (closest to cluster centroid)
                cluster_embeddings = embeddings[cluster_mask]
                centroid = np.mean(cluster_embeddings, axis=0)
                
                distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
                exemplar_idx = np.argmin(distances)
                exemplar_text = cluster_texts[exemplar_idx]
                
                cluster_profiles[cluster_id] = {
                    'size': len(cluster_texts),
                    'top_terms': list(zip(top_terms, top_scores)),
                    'exemplar': exemplar_text[:200] + "..." if len(exemplar_text) > 200 else exemplar_text
                }
            except Exception as e:
                logger.warning(f"Error analyzing cluster {cluster_id}: {e}")
                cluster_profiles[cluster_id] = {
                    'size': len(cluster_texts),
                    'top_terms': [],
                    'exemplar': cluster_texts[0][:200] + "..." if cluster_texts else ""
                }
        
        return {
            'embedding_name': embedding_name,
            'n_clusters': n_clusters,
            'cluster_profiles': cluster_profiles
        }


class TopicClusterAnalyzer:
    """Analyzes relationship between LDA topics and clusters."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the topic-cluster analyzer.
        
        Args:
            config: Configuration dictionary with evaluation parameters
        """
        self.config = config
        self.lda_k_values = config.get('lda_k_values', [20, 25, 30])
        self.top_n_terms = config.get('top_n_terms', 15)
    
    def analyze_topic_cluster_alignment(
        self,
        lda_model: LatentDirichletAllocation,
        dictionary: Any,
        cluster_labels: np.ndarray,
        texts: List[str],
        k: int
    ) -> Dict[str, Any]:
        """Analyze alignment between LDA topics and clusters.
        
        Args:
            lda_model: Trained LDA model
            dictionary: Gensim dictionary
            cluster_labels: Cluster assignments
            texts: Document texts
            k: Number of topics
            
        Returns:
            Dictionary with topic-cluster alignment analysis
        """
        # Get topic distributions for documents
        # Convert texts to bag-of-words format
        from gensim.corpora import Dictionary as GensimDictionary
        from gensim.models import LdaModel
        
        # Create corpus
        texts_tokenized = [text.split() for text in texts]
        corpus = [dictionary.doc2bow(text) for text in texts_tokenized]
        
        # Get document-topic distributions
        doc_topic_dist = lda_model.transform(corpus)
        
        # Analyze topic-cluster alignment
        n_clusters = len(np.unique(cluster_labels))
        n_topics = lda_model.n_components
        
        # Calculate dominant topic per cluster
        cluster_topic_alignments = {}
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_topic_dist = doc_topic_dist[cluster_mask]
            
            if len(cluster_topic_dist) == 0:
                continue
            
            # Average topic distribution for this cluster
            avg_topic_dist = np.mean(cluster_topic_dist, axis=0)
            dominant_topic = np.argmax(avg_topic_dist)
            dominant_topic_prob = avg_topic_dist[dominant_topic]
            
            cluster_topic_alignments[cluster_id] = {
                'dominant_topic': int(dominant_topic),
                'dominant_topic_prob': float(dominant_topic_prob),
                'topic_distribution': avg_topic_dist.tolist()
            }
        
        # Calculate topic purity within clusters
        topic_purities = {}
        for topic_id in range(n_topics):
            topic_docs = np.argmax(doc_topic_dist, axis=1) == topic_id
            if np.sum(topic_docs) == 0:
                continue
            
            topic_clusters = cluster_labels[topic_docs]
            unique_clusters, counts = np.unique(topic_clusters, return_counts=True)
            purity = np.max(counts) / np.sum(counts)
            
            topic_purities[topic_id] = {
                'purity': float(purity),
                'dominant_cluster': int(unique_clusters[np.argmax(counts)]),
                'cluster_distribution': dict(zip(unique_clusters.tolist(), counts.tolist()))
            }
        
        # Calculate mutual information
        from sklearn.metrics import mutual_info_score
        dominant_topics = np.argmax(doc_topic_dist, axis=1)
        mi_score = mutual_info_score(cluster_labels, dominant_topics)
        
        return {
            'k': k,
            'n_clusters': n_clusters,
            'n_topics': n_topics,
            'cluster_topic_alignments': cluster_topic_alignments,
            'topic_purities': topic_purities,
            'mutual_information': float(mi_score)
        }
    
    def identify_topic_cluster_mismatches(
        self,
        lda_model: LatentDirichletAllocation,
        dictionary: Any,
        cluster_labels: np.ndarray,
        texts: List[str],
        k: int
    ) -> List[Dict[str, Any]]:
        """Identify documents where topic and cluster assignments conflict.
        
        Args:
            lda_model: Trained LDA model
            dictionary: Gensim dictionary
            cluster_labels: Cluster assignments
            texts: Document texts
            k: Number of topics
            
        Returns:
            List of mismatch cases with analysis
        """
        # Get topic distributions
        texts_tokenized = [text.split() for text in texts]
        corpus = [dictionary.doc2bow(text) for text in texts_tokenized]
        doc_topic_dist = lda_model.transform(corpus)
        
        # Find dominant topic for each document
        dominant_topics = np.argmax(doc_topic_dist, axis=1)
        
        # Find mismatches (documents where topic and cluster don't align well)
        mismatches = []
        
        for i, (topic, cluster) in enumerate(zip(dominant_topics, cluster_labels)):
            # Check if this topic is dominant in this cluster
            cluster_mask = cluster_labels == cluster
            cluster_topic_dist = doc_topic_dist[cluster_mask]
            
            if len(cluster_topic_dist) == 0:
                continue
            
            avg_topic_dist = np.mean(cluster_topic_dist, axis=0)
            cluster_dominant_topic = np.argmax(avg_topic_dist)
            
            # If document's dominant topic is not the cluster's dominant topic
            if topic != cluster_dominant_topic:
                topic_prob = doc_topic_dist[i][topic]
                cluster_topic_prob = doc_topic_dist[i][cluster_dominant_topic]
                
                # Only consider significant mismatches
                if topic_prob > cluster_topic_prob + 0.1:  # Threshold
                    mismatches.append({
                        'doc_index': i,
                        'text': texts[i][:200] + "..." if len(texts[i]) > 200 else texts[i],
                        'assigned_topic': int(topic),
                        'assigned_cluster': int(cluster),
                        'cluster_dominant_topic': int(cluster_dominant_topic),
                        'topic_prob': float(topic_prob),
                        'cluster_topic_prob': float(cluster_topic_prob)
                    })
        
        return mismatches


class VisualizationGenerator:
    """Generates comprehensive visualizations for evaluation results."""
    
    def __init__(self, config: Dict[str, Any], output_dir: Path):
        """Initialize the visualization generator.
        
        Args:
            config: Configuration dictionary with visualization parameters
            output_dir: Output directory for saving visualizations
        """
        self.config = config
        self.output_dir = output_dir
        self.umap_params = config.get('umap_params', {})
        self.tsne_params = config.get('tsne_params', {})
        self.figure_dpi = config.get('figure_dpi', 300)
        self.figure_size = config.get('figure_size', [12, 8])
        
        # Set up matplotlib
        plt.style.use('default')
        sns.set_palette("husl")
    
    def create_embedding_projections(
        self,
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        embedding_name: str,
        method: str = 'umap'
    ) -> None:
        """Create 2D projections of embeddings.
        
        Args:
            embeddings: Document embeddings
            cluster_labels: Cluster assignments
            embedding_name: Name of the embedding method
            method: Projection method ('umap' or 'tsne')
        """
        if method == 'umap':
            reducer = umap.UMAP(
                n_neighbors=self.umap_params.get('n_neighbors', 15),
                min_dist=self.umap_params.get('min_dist', 0.1),
                metric=self.umap_params.get('metric', 'cosine'),
                random_state=42
            )
        elif method == 'tsne':
            reducer = TSNE(
                perplexity=self.tsne_params.get('perplexity', 30),
                n_iter=self.tsne_params.get('n_iter', 1000),
                random_state=42
            )
        else:
            raise ValueError(f"Unknown projection method: {method}")
        
        # Reduce dimensions
        embedding_2d = reducer.fit_transform(embeddings)
        
        # Create plot
        plt.figure(figsize=self.figure_size)
        
        # Get unique clusters and colors
        unique_clusters = np.unique(cluster_labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
        
        for i, cluster_id in enumerate(unique_clusters):
            mask = cluster_labels == cluster_id
            plt.scatter(
                embedding_2d[mask, 0],
                embedding_2d[mask, 1],
                c=[colors[i]],
                label=f'Cluster {cluster_id}',
                alpha=0.7,
                s=20
            )
        
        plt.title(f'{method.upper()} Projection - {embedding_name}')
        plt.xlabel(f'{method.upper()} 1')
        plt.ylabel(f'{method.upper()} 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save plot
        output_file = self.output_dir / 'embeddings' / f'{method}_{embedding_name.lower().replace(" ", "_")}.png'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=self.figure_dpi, bbox_inches='tight')
        plt.close()
    
    def create_clustering_comparison(
        self,
        clustering_results: List[Dict[str, Any]]
    ) -> None:
        """Create comparison plots for clustering results.
        
        Args:
            clustering_results: List of clustering evaluation results
        """
        # Prepare data for plotting
        df = pd.DataFrame(clustering_results)
        df = df[df['valid'] == True]  # Only valid results
        
        if len(df) == 0:
            logger.warning("No valid clustering results to plot")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Silhouette scores
        sns.barplot(data=df, x='embedding_name', y='silhouette_score', hue='algorithm', ax=axes[0, 0])
        axes[0, 0].set_title('Silhouette Score by Embedding and Algorithm')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Davies-Bouldin scores (lower is better)
        sns.barplot(data=df, x='embedding_name', y='davies_bouldin_score', hue='algorithm', ax=axes[0, 1])
        axes[0, 1].set_title('Davies-Bouldin Score by Embedding and Algorithm')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Calinski-Harabasz scores
        sns.barplot(data=df, x='embedding_name', y='calinski_harabasz_score', hue='algorithm', ax=axes[1, 0])
        axes[1, 0].set_title('Calinski-Harabasz Score by Embedding and Algorithm')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Number of clusters
        sns.barplot(data=df, x='embedding_name', y='n_clusters', hue='algorithm', ax=axes[1, 1])
        axes[1, 1].set_title('Number of Clusters by Embedding and Algorithm')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        output_file = self.output_dir / 'clustering' / 'clustering_comparison.png'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=self.figure_dpi, bbox_inches='tight')
        plt.close()
    
    def create_topic_visualization(
        self,
        lda_model: LatentDirichletAllocation,
        feature_names: List[str],
        k: int,
        top_n: int = 15
    ) -> None:
        """Create visualization of LDA topics.
        
        Args:
            lda_model: Trained LDA model
            feature_names: List of feature names (vocabulary)
            k: Number of topics
            top_n: Number of top terms to show per topic
        """
        # Get topic-term distributions
        topic_term_dist = lda_model.components_
        
        # Create subplots
        n_cols = 3
        n_rows = (k + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for topic_idx in range(k):
            row = topic_idx // n_cols
            col = topic_idx % n_cols
            
            # Get top terms for this topic
            top_indices = np.argsort(topic_term_dist[topic_idx])[-top_n:][::-1]
            top_terms = [feature_names[i] for i in top_indices]
            top_scores = [topic_term_dist[topic_idx][i] for i in top_indices]
            
            # Create bar plot
            axes[row, col].barh(range(len(top_terms)), top_scores)
            axes[row, col].set_yticks(range(len(top_terms)))
            axes[row, col].set_yticklabels(top_terms)
            axes[row, col].set_title(f'Topic {topic_idx}')
            axes[row, col].invert_yaxis()
        
        # Hide empty subplots
        for topic_idx in range(k, n_rows * n_cols):
            row = topic_idx // n_cols
            col = topic_idx % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        output_file = self.output_dir / 'topics' / f'lda_k{k}_top_terms.png'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=self.figure_dpi, bbox_inches='tight')
        plt.close()
    
    def create_retrieval_comparison(
        self,
        retrieval_results: List[Dict[str, Any]]
    ) -> None:
        """Create comparison plots for retrieval results.
        
        Args:
            retrieval_results: List of retrieval evaluation results
        """
        df = pd.DataFrame(retrieval_results)
        
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Precision@K comparison
        sns.barplot(data=df, x='embedding_name', y='precision_at_k', ax=axes[0])
        axes[0].set_title('Precision@5 by Embedding Method')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].set_ylabel('Precision@5')
        
        # MRR comparison
        sns.barplot(data=df, x='embedding_name', y='mean_reciprocal_rank', ax=axes[1])
        axes[1].set_title('Mean Reciprocal Rank by Embedding Method')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].set_ylabel('MRR')
        
        plt.tight_layout()
        
        # Save plot
        output_file = self.output_dir / 'retrieval' / 'embedding_comparison.png'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=self.figure_dpi, bbox_inches='tight')
        plt.close()


class UnifiedEvaluationPipeline:
    """Main pipeline for unified evaluation and visualization."""
    
    def __init__(self, config_path: Union[str, Path]):
        """Initialize the unified evaluation pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Create output directory
        self.output_dir = create_output_directory("artifacts/unified_evaluation")
        
        # Initialize evaluators
        self.intrinsic_evaluator = IntrinsicEvaluator(self.config['intrinsic'])
        self.retrieval_evaluator = RetrievalEvaluator(self.config['retrieval'])
        self.clustering_evaluator = ClusteringEvaluator(self.config['clustering'])
        self.topic_analyzer = TopicClusterAnalyzer(self.config['topics'])
        self.viz_generator = VisualizationGenerator(
            self.config['visualization'], 
            self.output_dir
        )
        
        # Results storage
        self.results = {
            'intrinsic': {},
            'retrieval': {},
            'clustering': {},
            'topics': {}
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        import yaml
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run the complete unified evaluation pipeline."""
        logger.info("Starting unified evaluation pipeline")
        
        # Load data and embeddings
        self._load_data_and_embeddings()
        
        # Run intrinsic evaluation
        self._run_intrinsic_evaluation()
        
        # Run retrieval evaluation
        self._run_retrieval_evaluation()
        
        # Run clustering evaluation
        self._run_clustering_evaluation()
        
        # Run topic-cluster analysis
        self._run_topic_analysis()
        
        # Generate visualizations
        self._generate_visualizations()
        
        # Generate report
        self._generate_report()
        
        logger.info("Unified evaluation pipeline completed")
        return self.results
    
    def _load_data_and_embeddings(self):
        """Load all required data and embeddings."""
        logger.info("Loading data and embeddings")
        
        # Load combined data
        self.data = pd.read_parquet("data/interim/programs_core_final.parquet")
        
        # Load tokenized data for embeddings
        self.jobs_data = pd.read_parquet("data/interim/jobs_tokenized.parquet")
        self.programs_data = pd.read_parquet("data/interim/programs_tokenized.parquet")
        
        # Combine texts
        self.all_texts = list(self.jobs_data['description_text']) + list(self.programs_data['description_text'])
        self.program_texts = list(self.programs_data['description_text'])
        self.job_texts = list(self.jobs_data['description_text'])
        
        # Load embeddings
        self.embeddings = {}
        
        # Word2Vec models
        for model_name in self.config['embeddings']['word2vec_models']:
            model_path = f"artifacts/word2vec/{model_name}/model.model"
            model = load_word2vec_model(model_path)
            
            # Get document embeddings
            all_tokenized = list(self.jobs_data['description_text_tokens']) + list(self.programs_data['description_text_tokens'])
            embeddings = get_document_embeddings_word2vec(model, all_tokenized)
            self.embeddings[f"word2vec_{model_name}"] = embeddings
        
        # Doc2Vec models
        for model_name in self.config['embeddings']['doc2vec_models']:
            model_path = f"artifacts/doc2vec/{model_name}/model.model"
            model = load_doc2vec_model(model_path)
            
            # Get document embeddings
            all_tokenized = list(self.jobs_data['description_text_tokens']) + list(self.programs_data['description_text_tokens'])
            embeddings = get_document_embeddings_doc2vec(model, all_tokenized)
            self.embeddings[f"doc2vec_{model_name}"] = embeddings
        
        # Transformer embeddings (jobs only)
        transformer_path = self.config['embeddings']['transformer']['path']
        transformer_embeddings = load_transformer_embeddings(transformer_path)
        
        # For transformer embeddings, we only have job embeddings
        # We need to create dummy embeddings for programs or skip retrieval evaluation
        if transformer_embeddings.shape[0] == len(self.jobs_data):
            # Only job embeddings available
            self.embeddings['transformer_jobs'] = transformer_embeddings
            logger.warning("Transformer embeddings only available for jobs, skipping program-job retrieval")
        else:
            self.embeddings['transformer'] = transformer_embeddings
        
        logger.info(f"Loaded {len(self.embeddings)} embedding types")
    
    def _run_intrinsic_evaluation(self):
        """Run intrinsic evaluation."""
        logger.info("Running intrinsic evaluation")
        
        # Evaluate Word2Vec neighborhoods
        for model_name in self.config['embeddings']['word2vec_models']:
            model_path = f"artifacts/word2vec/{model_name}/model.model"
            model = load_word2vec_model(model_path)
            
            result = self.intrinsic_evaluator.evaluate_word2vec_neighborhoods(
                model, f"word2vec_{model_name}"
            )
            self.results['intrinsic'][f"word2vec_{model_name}"] = result
        
        # Evaluate cluster purity for each embedding
        for embedding_name, embeddings in self.embeddings.items():
            # Load clustering results (use K-means with k=10 as example)
            try:
                cluster_path = f"artifacts/clustering/features/{embedding_name}_features.npy"
                if Path(cluster_path).exists():
                    # This would need to be adapted based on actual clustering results structure
                    # For now, create dummy cluster labels
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=10, random_state=42)
                    cluster_labels = kmeans.fit_predict(embeddings)
                    
                    result = self.intrinsic_evaluator.evaluate_cluster_purity(
                        embeddings, cluster_labels, embedding_name
                    )
                    self.results['intrinsic'][f"{embedding_name}_purity"] = result
            except Exception as e:
                logger.warning(f"Could not evaluate cluster purity for {embedding_name}: {e}")
    
    def _run_retrieval_evaluation(self):
        """Run retrieval evaluation."""
        logger.info("Running retrieval evaluation")
        
        # Split embeddings into programs and jobs
        n_programs = len(self.programs_data)
        
        for embedding_name, embeddings in self.embeddings.items():
            # Skip transformer_jobs as it only has job embeddings
            if embedding_name == 'transformer_jobs':
                logger.info(f"Skipping retrieval evaluation for {embedding_name} (jobs only)")
                continue
                
            program_embeddings = embeddings[:n_programs]
            job_embeddings = embeddings[n_programs:]
            
            result = self.retrieval_evaluator.evaluate_retrieval(
                program_embeddings,
                job_embeddings,
                self.program_texts,
                self.job_texts,
                embedding_name
            )
            self.results['retrieval'][embedding_name] = result
    
    def _run_clustering_evaluation(self):
        """Run clustering evaluation."""
        logger.info("Running clustering evaluation")
        
        # Load existing clustering results
        clustering_dir = Path("artifacts/clustering")
        
        for embedding_name, embeddings in self.embeddings.items():
            # Skip transformer_jobs for clustering as it only has job embeddings
            if embedding_name == 'transformer_jobs':
                logger.info(f"Skipping clustering evaluation for {embedding_name} (jobs only)")
                continue
                
            # Try to load existing clustering results
            for algorithm in self.config['clustering']['algorithms']:
                for k in self.config['clustering']['k_values']:
                    try:
                        # This would need to be adapted based on actual file structure
                        # For now, create new clustering
                        if algorithm == 'kmeans':
                            from sklearn.cluster import KMeans
                            clusterer = KMeans(n_clusters=k, random_state=42)
                            cluster_labels = clusterer.fit_predict(embeddings)
                            
                            result = self.clustering_evaluator.evaluate_clustering_quality(
                                embeddings, cluster_labels, embedding_name, algorithm
                            )
                            self.results['clustering'][f"{embedding_name}_{algorithm}_k{k}"] = result
                    except Exception as e:
                        logger.warning(f"Could not evaluate clustering for {embedding_name}_{algorithm}_k{k}: {e}")
    
    def _run_topic_analysis(self):
        """Run topic-cluster analysis."""
        logger.info("Running topic-cluster analysis")
        
        # Load LDA models
        for k in self.config['topics']['lda_k_values']:
            try:
                lda_path = f"artifacts/lda/k{k}"
                lda_model, dictionary, metadata = load_lda_model(lda_path)
                
                # Use the first available embedding for clustering (skip transformer_jobs)
                available_embeddings = {k: v for k, v in self.embeddings.items() if k != 'transformer_jobs'}
                if not available_embeddings:
                    logger.warning("No suitable embeddings available for topic analysis")
                    continue
                    
                embedding_name, embeddings = next(iter(available_embeddings.items()))
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=10, random_state=42)
                cluster_labels = kmeans.fit_predict(embeddings)
                
                # Analyze topic-cluster alignment
                alignment_result = self.topic_analyzer.analyze_topic_cluster_alignment(
                    lda_model, dictionary, cluster_labels, self.all_texts, k
                )
                self.results['topics'][f"k{k}_alignment"] = alignment_result
                
                # Identify mismatches
                mismatches = self.topic_analyzer.identify_topic_cluster_mismatches(
                    lda_model, dictionary, cluster_labels, self.all_texts, k
                )
                self.results['topics'][f"k{k}_mismatches"] = mismatches
                
            except Exception as e:
                logger.warning(f"Could not analyze topics for k={k}: {e}")
    
    def _generate_visualizations(self):
        """Generate all visualizations."""
        logger.info("Generating visualizations")
        
        # Create embedding projections
        for embedding_name, embeddings in self.embeddings.items():
            # Skip transformer_jobs for visualization as it only has job embeddings
            if embedding_name == 'transformer_jobs':
                logger.info(f"Skipping visualization for {embedding_name} (jobs only)")
                continue
                
            # Create dummy cluster labels for visualization
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=10, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # UMAP projections
            self.viz_generator.create_embedding_projections(
                embeddings, cluster_labels, embedding_name, 'umap'
            )
            
            # t-SNE projections
            self.viz_generator.create_embedding_projections(
                embeddings, cluster_labels, embedding_name, 'tsne'
            )
        
        # Clustering comparison
        clustering_results = list(self.results['clustering'].values())
        if clustering_results:
            self.viz_generator.create_clustering_comparison(clustering_results)
        
        # Retrieval comparison
        retrieval_results = list(self.results['retrieval'].values())
        if retrieval_results:
            self.viz_generator.create_retrieval_comparison(retrieval_results)
        
        # Topic visualizations
        for k in self.config['topics']['lda_k_values']:
            try:
                lda_path = f"artifacts/lda/k{k}"
                lda_model, dictionary, metadata = load_lda_model(lda_path)
                
                # Get feature names from dictionary
                feature_names = [dictionary[i] for i in range(len(dictionary))]
                
                self.viz_generator.create_topic_visualization(
                    lda_model, feature_names, k
                )
            except Exception as e:
                logger.warning(f"Could not create topic visualization for k={k}: {e}")
    
    def _generate_report(self):
        """Generate comprehensive evaluation report."""
        logger.info("Generating evaluation report")
        
        # Save results as JSON
        results_file = self.output_dir / 'metrics' / 'evaluation_results.json'
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Generate markdown report
        self._create_markdown_report()
    
    def _create_markdown_report(self):
        """Create comprehensive markdown report."""
        report_file = self.output_dir / 'evaluation_report.md'
        
        with open(report_file, 'w') as f:
            f.write("# Unified Evaluation & Visualization Report\n\n")
            f.write(f"Generated on: {pd.Timestamp.now()}\n\n")
            
            # Intrinsic evaluation results
            f.write("## Intrinsic Evaluation Results\n\n")
            for name, result in self.results['intrinsic'].items():
                f.write(f"### {name}\n")
                if 'avg_similarity' in result:
                    f.write(f"- Average Similarity: {result['avg_similarity']:.4f}\n")
                if 'coherent_terms' in result:
                    f.write(f"- Coherent Terms: {result['coherent_terms']}/{len(self.config['intrinsic']['test_terms'])}\n")
                if 'avg_purity' in result:
                    f.write(f"- Average Cluster Purity: {result['avg_purity']:.4f}\n")
                f.write("\n")
            
            # Retrieval evaluation results
            f.write("## Retrieval Evaluation Results\n\n")
            f.write("| Embedding Method | Precision@5 | MRR |\n")
            f.write("|------------------|-------------|-----|\n")
            for name, result in self.results['retrieval'].items():
                f.write(f"| {name} | {result['precision_at_k']:.4f} | {result['mean_reciprocal_rank']:.4f} |\n")
            f.write("\n")
            
            # Clustering evaluation results
            f.write("## Clustering Evaluation Results\n\n")
            f.write("| Embedding | Algorithm | K | Silhouette | Davies-Bouldin | Calinski-Harabasz |\n")
            f.write("|-----------|-----------|---|------------|----------------|-------------------|\n")
            for name, result in self.results['clustering'].items():
                if result.get('valid', False):
                    f.write(f"| {result['embedding_name']} | {result['algorithm']} | {result['n_clusters']} | "
                           f"{result['silhouette_score']:.4f} | {result['davies_bouldin_score']:.4f} | "
                           f"{result['calinski_harabasz_score']:.4f} |\n")
            f.write("\n")
            
            # Topic analysis results
            f.write("## Topic-Cluster Analysis Results\n\n")
            for name, result in self.results['topics'].items():
                if 'mutual_information' in result:
                    f.write(f"### {name}\n")
                    f.write(f"- Mutual Information: {result['mutual_information']:.4f}\n")
                    f.write(f"- Number of Clusters: {result['n_clusters']}\n")
                    f.write(f"- Number of Topics: {result['n_topics']}\n\n")
        
        logger.info(f"Report saved to {report_file}")
