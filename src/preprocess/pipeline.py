"""
Main preprocessing pipeline orchestrator for programs.jsonl data.
Follows the flow: clean → segment → tokenize/lemma → stopwords → dedupe → stats → TF-IDF
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path
from dataclasses import dataclass
import time

# Import preprocessing modules
from .clean_text import TextCleaner
from .segment import TextSegmenter, SegmentationConfig
from .tokenize import DomainAwareTokenizer, TokenizationConfig
from .stopwords import DomainStopwordsFilter, StopwordsConfig
from .dedupe import JobDeduplicator, DeduplicationConfig
from .stats import StatisticsCollector
from .tfidf import TfidfVectorizerWrapper, TfidfConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the preprocessing pipeline."""
    # Input/Output paths
    input_file: str = "data/processed/programs.jsonl"
    output_dir: str = "data/interim"
    
    # Processing steps
    enable_cleaning: bool = True
    enable_segmentation: bool = True
    enable_tokenization: bool = True
    enable_stopwords: bool = True
    enable_deduplication: bool = True
    enable_statistics: bool = True
    enable_tfidf: bool = True
    
    # Step-specific configurations
    cleaning_config: Dict[str, Any] = None
    segmentation_config: Dict[str, Any] = None
    tokenization_config: Dict[str, Any] = None
    stopwords_config: Dict[str, Any] = None
    deduplication_config: Dict[str, Any] = None
    tfidf_config: Dict[str, Any] = None
    
    # Statistics and reporting
    save_intermediate_files: bool = True
    generate_reports: bool = True
    verbose: bool = True


class ProgramsPreprocessingPipeline:
    """Main preprocessing pipeline for programs data."""
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.stats_collector = StatisticsCollector()
        self.processing_stats = {}
        self.start_time = None
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all preprocessing components."""
        # Text cleaner
        if self.config.enable_cleaning:
            cleaning_params = self.config.cleaning_config or {}
            self.text_cleaner = TextCleaner(**cleaning_params)
        
        # Text segmenter
        if self.config.enable_segmentation:
            seg_config = SegmentationConfig(**(self.config.segmentation_config or {}))
            self.text_segmenter = TextSegmenter(seg_config)
        
        # Tokenizer
        if self.config.enable_tokenization:
            tok_config = TokenizationConfig(**(self.config.tokenization_config or {}))
            self.tokenizer = DomainAwareTokenizer(tok_config)
        
        # Stopwords filter
        if self.config.enable_stopwords:
            stop_config = StopwordsConfig(**(self.config.stopwords_config or {}))
            self.stopwords_filter = DomainStopwordsFilter(stop_config)
        
        # Deduplicator
        if self.config.enable_deduplication:
            dedup_config = DeduplicationConfig(**(self.config.deduplication_config or {}))
            self.deduplicator = JobDeduplicator(dedup_config)
        
        # TF-IDF vectorizer
        if self.config.enable_tfidf:
            tfidf_config = TfidfConfig(**(self.config.tfidf_config or {}))
            self.tfidf_vectorizer = TfidfVectorizerWrapper(tfidf_config)
    
    def _load_data(self) -> pd.DataFrame:
        """Load input data from JSONL file."""
        logger.info(f"Loading data from {self.config.input_file}")
        
        data = []
        with open(self.config.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line: {e}")
        
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} records")
        
        return df
    
    def _save_intermediate(self, df: pd.DataFrame, step_name: str):
        """Save intermediate results if configured."""
        if self.config.save_intermediate_files:
            output_path = Path(self.config.output_dir) / f"programs_{step_name}.parquet"
            df.to_parquet(output_path, index=False)
            logger.info(f"Saved intermediate results to {output_path}")
    
    def _clean_step(self, df: pd.DataFrame) -> pd.DataFrame:
        """Text cleaning step."""
        if not self.config.enable_cleaning:
            return df
        
        logger.info("=== CLEANING STEP ===")
        start_time = time.time()
        
        # Clean text columns
        text_columns = ['description_raw', 'description_text', 'program_name']
        df_clean = self.text_cleaner.clean_dataframe(df, text_columns)
        
        # Collect statistics
        step_stats = self.stats_collector.collect_processing_step_stats(
            "cleaning", df, df_clean
        )
        step_stats['processing_time'] = time.time() - start_time
        self.processing_stats['cleaning'] = step_stats
        
        self._save_intermediate(df_clean, "cleaned")
        
        logger.info(f"Cleaning completed in {step_stats['processing_time']:.2f}s")
        return df_clean
    
    def _segment_step(self, df: pd.DataFrame) -> pd.DataFrame:
        """Text segmentation step."""
        if not self.config.enable_segmentation:
            return df
        
        logger.info("=== SEGMENTATION STEP ===")
        start_time = time.time()
        
        # Segment long descriptions
        df_segmented = self.text_segmenter.segment_dataframe(
            df, ['description_raw', 'description_text']
        )
        
        # Collect statistics
        step_stats = self.stats_collector.collect_processing_step_stats(
            "segmentation", df, df_segmented
        )
        step_stats['processing_time'] = time.time() - start_time
        self.processing_stats['segmentation'] = step_stats
        
        self._save_intermediate(df_segmented, "segmented")
        
        logger.info(f"Segmentation completed in {step_stats['processing_time']:.2f}s")
        return df_segmented
    
    def _tokenize_step(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tokenization and lemmatization step."""
        if not self.config.enable_tokenization:
            return df
        
        logger.info("=== TOKENIZATION STEP ===")
        start_time = time.time()
        
        # Tokenize text columns
        text_columns = ['description_raw', 'description_text', 'program_name']
        df_tokenized = self.tokenizer.tokenize_dataframe(df, text_columns)
        
        # Collect statistics
        step_stats = self.stats_collector.collect_processing_step_stats(
            "tokenization", df, df_tokenized
        )
        step_stats['processing_time'] = time.time() - start_time
        self.processing_stats['tokenization'] = step_stats
        
        self._save_intermediate(df_tokenized, "tokenized")
        
        logger.info(f"Tokenization completed in {step_stats['processing_time']:.2f}s")
        return df_tokenized
    
    def _stopwords_step(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stopwords filtering step."""
        if not self.config.enable_stopwords:
            return df
        
        logger.info("=== STOPWORDS FILTERING STEP ===")
        start_time = time.time()
        
        # Filter stopwords from token columns
        token_columns = [col for col in df.columns if col.endswith('_tokens') or col.endswith('_lemmatized')]
        df_filtered = self.stopwords_filter.filter_text_columns(df, token_columns)
        
        # Collect statistics
        step_stats = self.stats_collector.collect_processing_step_stats(
            "stopwords_filtering", df, df_filtered
        )
        step_stats['processing_time'] = time.time() - start_time
        self.processing_stats['stopwords'] = step_stats
        
        self._save_intermediate(df_filtered, "stopwords_filtered")
        
        logger.info(f"Stopwords filtering completed in {step_stats['processing_time']:.2f}s")
        return df_filtered
    
    def _dedupe_step(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deduplication step."""
        if not self.config.enable_deduplication:
            return df
        
        logger.info("=== DEDUPLICATION STEP ===")
        start_time = time.time()
        
        # Deduplicate records
        df_deduped, dedup_stats = self.deduplicator.deduplicate(df)
        
        # Collect statistics
        step_stats = self.stats_collector.collect_processing_step_stats(
            "deduplication", df, df_deduped, dedup_stats
        )
        step_stats['processing_time'] = time.time() - start_time
        self.processing_stats['deduplication'] = step_stats
        
        self._save_intermediate(df_deduped, "deduped")
        
        logger.info(f"Deduplication completed in {step_stats['processing_time']:.2f}s")
        return df_deduped
    
    def _statistics_step(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Statistics collection step."""
        if not self.config.enable_statistics:
            return {}
        
        logger.info("=== STATISTICS COLLECTION STEP ===")
        start_time = time.time()
        
        # Collect comprehensive statistics
        stats = self.stats_collector.collect_text_statistics(df)
        token_freq_stats = self.stats_collector.collect_token_frequency_stats(
            df, [col for col in df.columns if col.endswith('_tokens')]
        )
        entity_stats = self.stats_collector.collect_domain_entity_stats(df)
        
        all_stats = {
            'text_stats': stats,
            'token_freq_stats': token_freq_stats,
            'entity_stats': entity_stats,
            'processing_stats': self.processing_stats
        }
        
        # Save statistics
        stats_file = Path(self.config.output_dir) / "preprocessing_statistics.json"
        self.stats_collector.save_statistics(all_stats, str(stats_file))
        
        # Generate report
        if self.config.generate_reports:
            report = self.stats_collector.generate_summary_report(all_stats)
            report_file = Path(self.config.output_dir) / "preprocessing_report.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Report saved to {report_file}")
        
        processing_time = time.time() - start_time
        logger.info(f"Statistics collection completed in {processing_time:.2f}s")
        
        return all_stats
    
    def _tfidf_step(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """TF-IDF vectorization step."""
        if not self.config.enable_tfidf:
            return df, None
        
        logger.info("=== TF-IDF VECTORIZATION STEP ===")
        start_time = time.time()
        
        # Determine text columns for vectorization
        text_columns = []
        for col in df.columns:
            if (col.endswith('_text') or col.endswith('_raw') or 
                col.endswith('_tokens') or col.endswith('_lemmatized') or
                col.endswith('_filtered')):
                text_columns.append(col)
        
        if not text_columns:
            logger.warning("No text columns found for TF-IDF vectorization")
            return df, None
        
        # Vectorize data
        tfidf_matrix, vectorizer = self.tfidf_vectorizer.fit_transform(df, text_columns)
        
        # Create features DataFrame
        from .tfidf import create_tfidf_features
        df_with_features = create_tfidf_features(df, tfidf_matrix, vectorizer)
        
        # Save vectorizer
        vectorizer_file = Path(self.config.output_dir) / "tfidf_vectorizer.pkl"
        vectorizer.save_vectorizer(str(vectorizer_file))
        
        # Collect statistics
        tfidf_stats = self.stats_collector.collect_tfidf_stats(
            tfidf_matrix, vectorizer.feature_names
        )
        
        step_stats = self.stats_collector.collect_processing_step_stats(
            "tfidf_vectorization", df, df_with_features, tfidf_stats
        )
        step_stats['processing_time'] = time.time() - start_time
        self.processing_stats['tfidf'] = step_stats
        
        self._save_intermediate(df_with_features, "tfidf_features")
        
        logger.info(f"TF-IDF vectorization completed in {step_stats['processing_time']:.2f}s")
        return df_with_features, tfidf_matrix
    
    def run_pipeline(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run the complete preprocessing pipeline.
        
        Returns:
            Tuple of (final processed DataFrame, statistics)
        """
        self.start_time = time.time()
        logger.info("Starting preprocessing pipeline")
        
        # Load data
        df = self._load_data()
        
        # Run processing steps in sequence
        df = self._clean_step(df)
        df = self._segment_step(df)
        df = self._tokenize_step(df)
        df = self._stopwords_step(df)
        df = self._dedupe_step(df)
        
        # Collect statistics
        stats = self._statistics_step(df)
        
        # TF-IDF vectorization
        df_final, tfidf_matrix = self._tfidf_step(df)
        
        # Save final results
        output_file = Path(self.config.output_dir) / "programs_preprocessed.parquet"
        df_final.to_parquet(output_file, index=False)
        logger.info(f"Final results saved to {output_file}")
        
        # Print summary
        total_time = time.time() - self.start_time
        logger.info(f"Pipeline completed in {total_time:.2f}s")
        logger.info(f"Final dataset shape: {df_final.shape}")
        
        return df_final, stats
    
    def run_step(self, step_name: str, input_file: str = None) -> pd.DataFrame:
        """
        Run a specific preprocessing step.
        
        Args:
            step_name: Name of the step to run
            input_file: Input file (if different from config)
            
        Returns:
            Processed DataFrame
        """
        # Load data
        if input_file:
            if input_file.endswith('.parquet'):
                df = pd.read_parquet(input_file)
            else:
                # Temporarily update config to load from specified file
                original_input = self.config.input_file
                self.config.input_file = input_file
                df = self._load_data()
                self.config.input_file = original_input
        else:
            df = self._load_data()
        
        # Run specific step
        if step_name == "clean":
            return self._clean_step(df)
        elif step_name == "segment":
            return self._segment_step(df)
        elif step_name == "tokenize":
            return self._tokenize_step(df)
        elif step_name == "stopwords":
            return self._stopwords_step(df)
        elif step_name == "dedupe":
            return self._dedupe_step(df)
        elif step_name == "tfidf":
            df_result, _ = self._tfidf_step(df)
            return df_result
        else:
            raise ValueError(f"Unknown step: {step_name}")


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline on programs data")
    parser.add_argument("--input", default="data/processed/programs.jsonl", 
                       help="Input JSONL file")
    parser.add_argument("--output-dir", default="data/interim", 
                       help="Output directory")
    parser.add_argument("--step", choices=['clean', 'segment', 'tokenize', 'stopwords', 'dedupe', 'tfidf'],
                       help="Run specific step only")
    parser.add_argument("--no-cleaning", action="store_true", 
                       help="Skip cleaning step")
    parser.add_argument("--no-segmentation", action="store_true", 
                       help="Skip segmentation step")
    parser.add_argument("--no-tokenization", action="store_true", 
                       help="Skip tokenization step")
    parser.add_argument("--no-stopwords", action="store_true", 
                       help="Skip stopwords filtering step")
    parser.add_argument("--no-deduplication", action="store_true", 
                       help="Skip deduplication step")
    parser.add_argument("--no-tfidf", action="store_true", 
                       help="Skip TF-IDF vectorization step")
    parser.add_argument("--no-intermediate", action="store_true", 
                       help="Don't save intermediate files")
    parser.add_argument("--no-reports", action="store_true", 
                       help="Don't generate reports")
    parser.add_argument("--verbose", action="store_true", 
                       help="Verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure pipeline
    config = PipelineConfig(
        input_file=args.input,
        output_dir=args.output_dir,
        enable_cleaning=not args.no_cleaning,
        enable_segmentation=not args.no_segmentation,
        enable_tokenization=not args.no_tokenization,
        enable_stopwords=not args.no_stopwords,
        enable_deduplication=not args.no_deduplication,
        enable_tfidf=not args.no_tfidf,
        save_intermediate_files=not args.no_intermediate,
        generate_reports=not args.no_reports,
        verbose=args.verbose
    )
    
    # Create and run pipeline
    pipeline = ProgramsPreprocessingPipeline(config)
    
    if args.step:
        # Run specific step
        result_df = pipeline.run_step(args.step, args.input)
        logger.info(f"Step '{args.step}' completed. Result shape: {result_df.shape}")
    else:
        # Run complete pipeline
        result_df, stats = pipeline.run_pipeline()
        logger.info(f"Pipeline completed. Final shape: {result_df.shape}")


if __name__ == "__main__":
    main()
