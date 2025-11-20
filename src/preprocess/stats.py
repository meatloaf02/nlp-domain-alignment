"""
Statistics collection utilities for preprocessing pipeline metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
from collections import Counter
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TextStatistics:
    """Statistics for text data."""
    total_records: int
    total_chunks: int
    total_tokens: int
    unique_tokens: int
    avg_tokens_per_record: float
    avg_tokens_per_chunk: float
    max_tokens_per_record: int
    min_tokens_per_record: int
    vocabulary_size: int
    avg_chunk_size: float
    max_chunk_size: int
    min_chunk_size: int


@dataclass
class ProcessingStatistics:
    """Statistics for processing steps."""
    cleaning_stats: Dict[str, Any]
    segmentation_stats: Dict[str, Any]
    tokenization_stats: Dict[str, Any]
    stopwords_stats: Dict[str, Any]
    deduplication_stats: Dict[str, Any]
    tfidf_stats: Dict[str, Any]


class StatisticsCollector:
    """Collect and analyze statistics throughout the preprocessing pipeline."""
    
    def __init__(self):
        self.stats_history = []
        self.current_stats = {}
    
    def collect_text_statistics(self, df: pd.DataFrame, 
                              text_columns: List[str] = None,
                              chunk_columns: List[str] = None) -> TextStatistics:
        """
        Collect basic text statistics from DataFrame.
        
        Args:
            df: Input DataFrame
            text_columns: List of text column names
            chunk_columns: List of chunk column names
            
        Returns:
            TextStatistics object
        """
        if text_columns is None:
            text_columns = [col for col in df.columns if col.endswith('_text') or col.endswith('_raw')]
        
        if chunk_columns is None:
            chunk_columns = [col for col in df.columns if col.endswith('_chunk')]
        
        # Basic counts
        total_records = len(df)
        total_chunks = len(df) if not chunk_columns else sum(
            df[col].notna().sum() for col in chunk_columns if col in df.columns
        )
        
        # Token statistics
        token_columns = [col for col in df.columns if col.endswith('_tokens')]
        total_tokens = 0
        unique_tokens = set()
        token_counts_per_record = []
        
        for col in token_columns:
            if col in df.columns:
                for tokens in df[col].dropna():
                    if isinstance(tokens, list):
                        total_tokens += len(tokens)
                        unique_tokens.update(tokens)
                        token_counts_per_record.append(len(tokens))
        
        # Chunk size statistics
        chunk_sizes = []
        for col in chunk_columns:
            if col in df.columns:
                for chunk in df[col].dropna():
                    if isinstance(chunk, str):
                        chunk_sizes.append(len(chunk))
        
        return TextStatistics(
            total_records=total_records,
            total_chunks=total_chunks,
            total_tokens=total_tokens,
            unique_tokens=len(unique_tokens),
            avg_tokens_per_record=total_tokens / total_records if total_records > 0 else 0,
            avg_tokens_per_chunk=total_tokens / total_chunks if total_chunks > 0 else 0,
            max_tokens_per_record=max(token_counts_per_record) if token_counts_per_record else 0,
            min_tokens_per_record=min(token_counts_per_record) if token_counts_per_record else 0,
            vocabulary_size=len(unique_tokens),
            avg_chunk_size=np.mean(chunk_sizes) if chunk_sizes else 0,
            max_chunk_size=max(chunk_sizes) if chunk_sizes else 0,
            min_chunk_size=min(chunk_sizes) if chunk_sizes else 0
        )
    
    def collect_token_frequency_stats(self, df: pd.DataFrame, 
                                    token_columns: List[str]) -> Dict[str, Any]:
        """
        Collect token frequency statistics.
        
        Args:
            df: Input DataFrame
            token_columns: List of token column names
            
        Returns:
            Dictionary with frequency statistics
        """
        all_tokens = []
        token_frequencies = Counter()
        
        for col in token_columns:
            if col in df.columns:
                for tokens in df[col].dropna():
                    if isinstance(tokens, list):
                        all_tokens.extend(tokens)
                        token_frequencies.update(tokens)
        
        if not all_tokens:
            return {
                'total_tokens': 0,
                'unique_tokens': 0,
                'most_common_tokens': [],
                'frequency_distribution': {},
                'vocabulary_coverage': {}
            }
        
        # Calculate frequency distribution
        total_tokens = len(all_tokens)
        unique_tokens = len(token_frequencies)
        
        # Most common tokens
        most_common = token_frequencies.most_common(50)
        
        # Frequency distribution
        freq_dist = {}
        for freq in token_frequencies.values():
            freq_dist[freq] = freq_dist.get(freq, 0) + 1
        
        # Vocabulary coverage (how many tokens appear once, twice, etc.)
        vocab_coverage = {
            'singletons': sum(1 for freq in token_frequencies.values() if freq == 1),
            'doubletons': sum(1 for freq in token_frequencies.values() if freq == 2),
            'rare_tokens': sum(1 for freq in token_frequencies.values() if freq <= 5),
            'common_tokens': sum(1 for freq in token_frequencies.values() if freq >= 10)
        }
        
        return {
            'total_tokens': total_tokens,
            'unique_tokens': unique_tokens,
            'most_common_tokens': most_common,
            'frequency_distribution': freq_dist,
            'vocabulary_coverage': vocab_coverage,
            'type_token_ratio': unique_tokens / total_tokens if total_tokens > 0 else 0
        }
    
    def collect_domain_entity_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Collect statistics about domain-specific entities.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with entity statistics
        """
        entity_stats = {}
        
        # Find entity columns
        entity_columns = [col for col in df.columns if col.endswith('_degrees') or 
                        col.endswith('_medical_terms') or col.endswith('_technical_terms') or
                        col.endswith('_skills') or col.endswith('_certifications')]
        
        for col in entity_columns:
            entity_type = col.split('_')[-1]  # Extract entity type
            all_entities = []
            
            for entities in df[col].dropna():
                if isinstance(entities, list):
                    all_entities.extend(entities)
            
            if all_entities:
                entity_counter = Counter(all_entities)
                entity_stats[entity_type] = {
                    'total_count': len(all_entities),
                    'unique_count': len(entity_counter),
                    'most_common': entity_counter.most_common(20),
                    'coverage': len(entity_counter) / len(df) if len(df) > 0 else 0
                }
            else:
                entity_stats[entity_type] = {
                    'total_count': 0,
                    'unique_count': 0,
                    'most_common': [],
                    'coverage': 0
                }
        
        return entity_stats
    
    def collect_processing_step_stats(self, step_name: str, 
                                    input_df: pd.DataFrame,
                                    output_df: pd.DataFrame,
                                    additional_stats: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Collect statistics for a processing step.
        
        Args:
            step_name: Name of the processing step
            input_df: Input DataFrame
            output_df: Output DataFrame
            additional_stats: Additional statistics to include
            
        Returns:
            Dictionary with step statistics
        """
        stats = {
            'step_name': step_name,
            'input_records': len(input_df),
            'output_records': len(output_df),
            'records_removed': len(input_df) - len(output_df),
            'removal_rate': (len(input_df) - len(output_df)) / len(input_df) if len(input_df) > 0 else 0,
            'columns_added': len(output_df.columns) - len(input_df.columns),
            'new_columns': list(set(output_df.columns) - set(input_df.columns))
        }
        
        if additional_stats:
            stats.update(additional_stats)
        
        return stats
    
    def collect_tfidf_stats(self, tfidf_matrix, feature_names: List[str]) -> Dict[str, Any]:
        """
        Collect TF-IDF statistics.
        
        Args:
            tfidf_matrix: TF-IDF matrix
            feature_names: List of feature names
            
        Returns:
            Dictionary with TF-IDF statistics
        """
        # Convert to dense matrix for easier analysis
        dense_matrix = tfidf_matrix.toarray()
        
        # Document statistics
        doc_stats = {
            'total_documents': dense_matrix.shape[0],
            'total_features': dense_matrix.shape[1],
            'avg_features_per_doc': np.mean(np.sum(dense_matrix > 0, axis=1)),
            'max_features_per_doc': np.max(np.sum(dense_matrix > 0, axis=1)),
            'min_features_per_doc': np.min(np.sum(dense_matrix > 0, axis=1)),
            'sparsity': 1 - (np.count_nonzero(dense_matrix) / dense_matrix.size)
        }
        
        # Feature statistics
        feature_stats = {
            'avg_tfidf_score': np.mean(dense_matrix[dense_matrix > 0]),
            'max_tfidf_score': np.max(dense_matrix),
            'features_with_zero_tfidf': np.sum(np.sum(dense_matrix, axis=0) == 0),
            'most_important_features': []
        }
        
        # Find most important features
        feature_importance = np.sum(dense_matrix, axis=0)
        top_features_idx = np.argsort(feature_importance)[-20:][::-1]
        feature_stats['most_important_features'] = [
            (feature_names[i], feature_importance[i]) 
            for i in top_features_idx
        ]
        
        return {
            'document_statistics': doc_stats,
            'feature_statistics': feature_stats,
            'matrix_shape': dense_matrix.shape,
            'memory_usage_mb': dense_matrix.nbytes / (1024 * 1024)
        }
    
    def generate_summary_report(self, stats: Dict[str, Any]) -> str:
        """
        Generate a summary report from collected statistics.
        
        Args:
            stats: Dictionary of statistics
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=== PREPROCESSING PIPELINE STATISTICS REPORT ===")
        report.append("")
        
        # Text statistics
        if 'text_stats' in stats:
            ts = stats['text_stats']
            report.append("TEXT STATISTICS:")
            report.append(f"  Total records: {ts.total_records:,}")
            report.append(f"  Total chunks: {ts.total_chunks:,}")
            report.append(f"  Total tokens: {ts.total_tokens:,}")
            report.append(f"  Unique tokens: {ts.unique_tokens:,}")
            report.append(f"  Vocabulary size: {ts.vocabulary_size:,}")
            report.append(f"  Avg tokens per record: {ts.avg_tokens_per_record:.1f}")
            report.append(f"  Avg tokens per chunk: {ts.avg_tokens_per_chunk:.1f}")
            report.append(f"  Avg chunk size: {ts.avg_chunk_size:.1f} characters")
            report.append("")
        
        # Token frequency statistics
        if 'token_freq_stats' in stats:
            tfs = stats['token_freq_stats']
            report.append("TOKEN FREQUENCY STATISTICS:")
            report.append(f"  Type-token ratio: {tfs['type_token_ratio']:.3f}")
            report.append(f"  Singletons: {tfs['vocabulary_coverage']['singletons']:,}")
            report.append(f"  Rare tokens (≤5 occurrences): {tfs['vocabulary_coverage']['rare_tokens']:,}")
            report.append(f"  Common tokens (≥10 occurrences): {tfs['vocabulary_coverage']['common_tokens']:,}")
            report.append("")
            
            if tfs['most_common_tokens']:
                report.append("MOST COMMON TOKENS:")
                for token, freq in tfs['most_common_tokens'][:10]:
                    report.append(f"  {token}: {freq:,}")
                report.append("")
        
        # Domain entity statistics
        if 'entity_stats' in stats:
            es = stats['entity_stats']
            report.append("DOMAIN ENTITY STATISTICS:")
            for entity_type, entity_data in es.items():
                report.append(f"  {entity_type.title()}:")
                report.append(f"    Total: {entity_data['total_count']:,}")
                report.append(f"    Unique: {entity_data['unique_count']:,}")
                report.append(f"    Coverage: {entity_data['coverage']:.1%}")
                if entity_data['most_common']:
                    report.append(f"    Top: {entity_data['most_common'][0][0]} ({entity_data['most_common'][0][1]})")
            report.append("")
        
        # Processing steps
        if 'processing_steps' in stats:
            report.append("PROCESSING STEPS:")
            for step in stats['processing_steps']:
                report.append(f"  {step['step_name'].upper()}:")
                report.append(f"    Input records: {step['input_records']:,}")
                report.append(f"    Output records: {step['output_records']:,}")
                report.append(f"    Records removed: {step['records_removed']:,}")
                report.append(f"    Removal rate: {step['removal_rate']:.1%}")
                report.append(f"    Columns added: {step['columns_added']}")
            report.append("")
        
        # TF-IDF statistics
        if 'tfidf_stats' in stats:
            tfs = stats['tfidf_stats']
            doc_stats = tfs['document_statistics']
            feat_stats = tfs['feature_statistics']
            
            report.append("TF-IDF STATISTICS:")
            report.append(f"  Matrix shape: {tfs['matrix_shape']}")
            report.append(f"  Sparsity: {doc_stats['sparsity']:.1%}")
            report.append(f"  Avg features per doc: {doc_stats['avg_features_per_doc']:.1f}")
            report.append(f"  Memory usage: {tfs['memory_usage_mb']:.1f} MB")
            report.append("")
            
            if feat_stats['most_important_features']:
                report.append("MOST IMPORTANT FEATURES:")
                for feature, importance in feat_stats['most_important_features'][:10]:
                    report.append(f"  {feature}: {importance:.3f}")
                report.append("")
        
        return "\n".join(report)
    
    def save_statistics(self, stats: Dict[str, Any], output_path: str):
        """
        Save statistics to JSON file.
        
        Args:
            stats: Dictionary of statistics
            output_path: Path to save statistics
        """
        # Convert dataclasses to dictionaries
        serializable_stats = {}
        for key, value in stats.items():
            if hasattr(value, '__dict__'):
                serializable_stats[key] = asdict(value)
            else:
                serializable_stats[key] = value
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_stats, f, indent=2, default=str)
        
        logger.info(f"Statistics saved to {output_path}")


def collect_programs_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Collect comprehensive statistics for programs data.
    
    Args:
        df: DataFrame containing program data
        
    Returns:
        Dictionary with all collected statistics
    """
    collector = StatisticsCollector()
    
    # Collect basic text statistics
    text_stats = collector.collect_text_statistics(df)
    
    # Collect token frequency statistics
    token_columns = [col for col in df.columns if col.endswith('_tokens')]
    token_freq_stats = collector.collect_token_frequency_stats(df, token_columns)
    
    # Collect domain entity statistics
    entity_stats = collector.collect_domain_entity_stats(df)
    
    return {
        'text_stats': text_stats,
        'token_freq_stats': token_freq_stats,
        'entity_stats': entity_stats
    }


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect statistics from preprocessed data")
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--output", required=True, help="Output statistics JSON file")
    parser.add_argument("--report", help="Output report text file")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Load data
    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df)} records")
    
    # Collect statistics
    stats = collect_programs_statistics(df)
    
    # Save statistics
    collector = StatisticsCollector()
    collector.save_statistics(stats, args.output)
    
    # Generate and save report
    if args.report:
        report = collector.generate_summary_report(stats)
        with open(args.report, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Report saved to {args.report}")
    
    logger.info(f"Statistics saved to {args.output}")


if __name__ == "__main__":
    main()
