"""
Job description preprocessing pipeline.

This module processes job descriptions through the preprocessing pipeline,
concatenating title and description for richer context.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import time

# Import preprocessing modules
from .clean_text import TextCleaner
from .tokenize_text import DomainAwareTokenizer, TokenizationConfig
from .stopwords import DomainStopwordsFilter, StopwordsConfig

logger = logging.getLogger(__name__)


class JobPreprocessor:
    """Preprocessing pipeline specifically for job descriptions."""
    
    def __init__(self):
        """Initialize preprocessing components."""
        # Text cleaner
        self.text_cleaner = TextCleaner()
        
        # Tokenizer with domain-specific config
        token_config = TokenizationConfig(
            use_spacy=True,
            use_lemmatization=True,
            use_stemming=False,
            preserve_case=False,
            min_token_length=2,
            max_token_length=50,
            remove_numbers=False,  # Keep numbers for job IDs, years, etc.
            remove_punctuation=True,
            preserve_hyphenated=True,
            preserve_contractions=True,
            domain_specific_patterns=True
        )
        self.tokenizer = DomainAwareTokenizer(token_config)
        
        # Stopwords filter
        stop_config = StopwordsConfig(
            use_domain_stopwords=True,
            use_general_stopwords=True,
            preserve_important_terms=True
        )
        self.stopwords_filter = DomainStopwordsFilter(stop_config)
    
    def concatenate_title_description(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Concatenate title and description for richer context.
        
        Args:
            df: DataFrame with 'title' and 'description' columns
            
        Returns:
            DataFrame with concatenated 'combined_text' column
        """
        logger.info("Concatenating title and description fields")
        
        df_combined = df.copy()
        
        # Handle missing values
        df_combined['title'] = df_combined['title'].fillna('')
        df_combined['description'] = df_combined['description'].fillna('')
        
        # Concatenate title and description
        df_combined['combined_text'] = (
            df_combined['title'].astype(str) + ' ' + 
            df_combined['description'].astype(str)
        )
        
        # Clean up extra whitespace
        df_combined['combined_text'] = df_combined['combined_text'].str.replace(r'\s+', ' ', regex=True)
        df_combined['combined_text'] = df_combined['combined_text'].str.strip()
        
        # Remove empty combined texts
        empty_mask = df_combined['combined_text'].str.len() == 0
        if empty_mask.any():
            logger.warning(f"Found {empty_mask.sum()} records with empty combined text")
            df_combined = df_combined[~empty_mask]
        
        logger.info(f"Created combined text for {len(df_combined)} records")
        logger.info(f"Average combined text length: {df_combined['combined_text'].str.len().mean():.0f} characters")
        
        return df_combined
    
    def clean_job_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean job description text.
        
        Args:
            df: DataFrame with 'combined_text' column
            
        Returns:
            DataFrame with cleaned text
        """
        logger.info("Cleaning job description text")
        
        # Clean the combined text
        df_cleaned = self.text_cleaner.clean_dataframe(df, ['combined_text'])
        
        # Rename to match expected column names
        df_cleaned = df_cleaned.rename(columns={'combined_text': 'description_text'})
        
        logger.info(f"Cleaned text for {len(df_cleaned)} records")
        
        return df_cleaned
    
    def tokenize_job_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tokenize job description text.
        
        Args:
            df: DataFrame with 'description_text' column
            
        Returns:
            DataFrame with tokenized text
        """
        logger.info("Tokenizing job description text")
        
        # Tokenize the description text
        df_tokenized = self.tokenizer.tokenize_dataframe(df, ['description_text'])
        
        logger.info(f"Tokenized text for {len(df_tokenized)} records")
        
        return df_tokenized
    
    def filter_stopwords(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter stopwords from tokenized text.
        
        Args:
            df: DataFrame with token columns
            
        Returns:
            DataFrame with stopwords filtered
        """
        logger.info("Filtering stopwords from tokenized text")
        
        # Find token columns
        token_columns = [col for col in df.columns if col.endswith('_tokens')]
        
        if not token_columns:
            logger.warning("No token columns found for stopwords filtering")
            return df
        
        # Filter stopwords
        df_filtered = self.stopwords_filter.filter_text_columns(df, token_columns)
        
        logger.info(f"Filtered stopwords for {len(df_filtered)} records")
        
        return df_filtered
    
    def process_jobs(self, input_file: str, output_file: str) -> pd.DataFrame:
        """
        Process job descriptions through the complete preprocessing pipeline.
        
        Args:
            input_file: Path to input parquet file
            output_file: Path to output parquet file
            
        Returns:
            Processed DataFrame
        """
        start_time = time.time()
        logger.info(f"Starting job preprocessing pipeline")
        logger.info(f"Input: {input_file}")
        logger.info(f"Output: {output_file}")
        
        # Load data
        logger.info("Loading job descriptions data")
        df = pd.read_parquet(input_file)
        logger.info(f"Loaded {len(df)} job records")
        
        # Check required columns
        required_columns = ['title', 'description']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Process through pipeline
        df = self.concatenate_title_description(df)
        df = self.clean_job_text(df)
        df = self.tokenize_job_text(df)
        df = self.filter_stopwords(df)
        
        # Save processed data
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        
        # Log final statistics
        processing_time = time.time() - start_time
        logger.info(f"Job preprocessing completed in {processing_time:.2f}s")
        logger.info(f"Final dataset shape: {df.shape}")
        
        # Log token statistics
        token_columns = [col for col in df.columns if col.endswith('_tokens')]
        for col in token_columns:
            if col in df.columns:
                token_counts = df[col].apply(len)
                logger.info(f"{col}: avg {token_counts.mean():.1f} tokens per record")
        
        return df


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess job descriptions")
    parser.add_argument("--input", default="data/raw/core_programs_adzuna_cleaned.parquet",
                       help="Input parquet file")
    parser.add_argument("--output", default="data/interim/jobs_tokenized.parquet",
                       help="Output parquet file")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Process jobs
    preprocessor = JobPreprocessor()
    df = preprocessor.process_jobs(args.input, args.output)
    
    logger.info(f"Job preprocessing completed successfully")
    logger.info(f"Processed {len(df)} job records")


if __name__ == "__main__":
    main()
