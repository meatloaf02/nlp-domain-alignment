"""
Program Data Cleaning Module

This module provides utilities to clean and deduplicate program data,
removing non-program entries and enforcing quality filters.
"""

import logging
import re
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ProgramCleaner:
    """Clean and deduplicate program data."""
    
    # Patterns to identify non-program entries
    NON_PROGRAM_PATTERNS = [
        r'campus',
        r'admissions?',
        r'financial\s+aid',
        r'apply\s+now',
        r'schedule',
        r'catalog',
        r'about\s+us',
        r'contact',
        r'location',
        r'address',
        r'phone',
        r'email',
        r'career\s+services',
        r'student\s+services',
    ]
    
    def __init__(self, min_word_count: int = 50):
        """Initialize the program cleaner.
        
        Args:
            min_word_count: Minimum number of words in description
        """
        self.min_word_count = min_word_count
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.NON_PROGRAM_PATTERNS]
    
    def normalize_program_name(self, name: str) -> str:
        """Normalize program name for deduplication.
        
        Args:
            name: Program name
            
        Returns:
            Normalized program name
        """
        if pd.isna(name) or not name:
            return ''
        
        # Convert to lowercase
        normalized = str(name).lower()
        
        # Remove punctuation
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def is_non_program(self, program_name: str, description: str = '') -> bool:
        """Check if a record is a non-program entry (campus info, etc.).
        
        Args:
            program_name: Program name
            description: Program description
            
        Returns:
            True if this appears to be a non-program entry
        """
        text = f"{program_name} {description}".lower()
        
        # Check against patterns
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                return True
        
        return False
    
    def count_words(self, text: str) -> int:
        """Count words in text.
        
        Args:
            text: Text to count
            
        Returns:
            Number of words
        """
        if pd.isna(text) or not text:
            return 0
        
        return len(str(text).split())
    
    def deduplicate(self, df: pd.DataFrame, program_name_col: str = 'program_name') -> pd.DataFrame:
        """Deduplicate programs based on normalized names.
        
        Args:
            df: DataFrame with programs
            program_name_col: Column name for program name
            
        Returns:
            Deduplicated DataFrame
        """
        logger.info(f"Deduplicating {len(df)} programs")
        
        # Create normalized name column
        df = df.copy()
        df['_normalized_name'] = df[program_name_col].apply(self.normalize_program_name)
        
        # Get domain column if available
        domain_col = None
        for col in ['domain', 'domain_label']:
            if col in df.columns:
                domain_col = col
                break
        
        # Group by normalized name (and domain if available)
        if domain_col:
            # Keep first occurrence per (normalized_name, domain) pair
            df_dedup = df.drop_duplicates(subset=['_normalized_name', domain_col], keep='first')
        else:
            # Keep first occurrence per normalized_name
            df_dedup = df.drop_duplicates(subset=['_normalized_name'], keep='first')
        
        # Remove temporary column
        df_dedup = df_dedup.drop(columns=['_normalized_name'])
        
        logger.info(f"After deduplication: {len(df_dedup)} programs ({len(df) - len(df_dedup)} removed)")
        
        return df_dedup
    
    def remove_non_programs(self, df: pd.DataFrame, 
                           program_name_col: str = 'program_name',
                           description_col: str = 'description_text') -> pd.DataFrame:
        """Remove non-program entries (campus info, admissions, etc.).
        
        Args:
            df: DataFrame with programs
            program_name_col: Column name for program name
            description_col: Column name for description
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Removing non-program entries from {len(df)} records")
        
        df = df.copy()
        
        # Get description column (try alternatives)
        desc_col = description_col
        if desc_col not in df.columns:
            for alt in ['description', 'description_raw', 'description_text']:
                if alt in df.columns:
                    desc_col = alt
                    break
        
        # Check each row
        mask = []
        removed_count = 0
        
        for idx, row in df.iterrows():
            program_name = row.get(program_name_col, '')
            description = row.get(desc_col, '') if desc_col in df.columns else ''
            
            if self.is_non_program(program_name, description):
                mask.append(False)
                removed_count += 1
            else:
                mask.append(True)
        
        df_clean = df[mask].copy()
        
        logger.info(f"After removing non-programs: {len(df_clean)} programs ({removed_count} removed)")
        
        return df_clean
    
    def enforce_min_length(self, df: pd.DataFrame, 
                          description_col: str = 'description_text',
                          min_word_count: Optional[int] = None) -> pd.DataFrame:
        """Remove programs with descriptions shorter than minimum word count.
        
        Args:
            df: DataFrame with programs
            description_col: Column name for description
            min_word_count: Minimum word count (uses self.min_word_count if None)
            
        Returns:
            Filtered DataFrame
        """
        if min_word_count is None:
            min_word_count = self.min_word_count
        
        logger.info(f"Filtering programs with < {min_word_count} words")
        
        df = df.copy()
        
        # Get description column (try alternatives)
        desc_col = description_col
        if desc_col not in df.columns:
            for alt in ['description', 'description_raw', 'description_text']:
                if alt in df.columns:
                    desc_col = alt
                    break
        
        if desc_col not in df.columns:
            logger.warning(f"Description column not found, skipping length filter")
            return df
        
        # Count words and filter
        word_counts = df[desc_col].apply(self.count_words)
        mask = word_counts >= min_word_count
        
        df_filtered = df[mask].copy()
        
        logger.info(f"After length filter: {len(df_filtered)} programs ({len(df) - len(df_filtered)} removed)")
        
        return df_filtered
    
    def clean_programs(self, df: pd.DataFrame,
                      program_name_col: str = 'program_name',
                      description_col: str = 'description_text',
                      deduplicate: bool = True,
                      remove_non_programs: bool = True,
                      enforce_min_length: bool = True) -> pd.DataFrame:
        """Apply all cleaning steps to program data.
        
        Args:
            df: DataFrame with programs
            program_name_col: Column name for program name
            description_col: Column name for description
            deduplicate: Whether to deduplicate
            remove_non_programs: Whether to remove non-program entries
            enforce_min_length: Whether to enforce minimum length
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("=" * 80)
        logger.info("CLEANING PROGRAM DATA")
        logger.info("=" * 80)
        logger.info(f"Starting with {len(df)} programs")
        
        df_clean = df.copy()
        
        # Step 1: Remove non-program entries
        if remove_non_programs:
            df_clean = self.remove_non_programs(df_clean, program_name_col, description_col)
        
        # Step 2: Enforce minimum length
        if enforce_min_length:
            df_clean = self.enforce_min_length(df_clean, description_col)
        
        # Step 3: Deduplicate
        if deduplicate:
            df_clean = self.deduplicate(df_clean, program_name_col)
        
        logger.info("=" * 80)
        logger.info(f"Cleaning complete: {len(df_clean)} programs (removed {len(df) - len(df_clean)})")
        logger.info("=" * 80)
        
        return df_clean

