"""
Domain-specific stopwords filtering utilities for vocational program data.
"""

import pandas as pd
from typing import List, Dict, Any, Optional, Set
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class StopwordsConfig:
    """Configuration for stopwords filtering."""
    use_general_stopwords: bool = True
    use_domain_stopwords: bool = True
    use_custom_stopwords: bool = True
    domain_stopwords_file: str = "data/interim/domain_stopwords.txt"
    custom_stopwords_file: Optional[str] = None
    preserve_important_terms: bool = True
    min_word_frequency: int = 2  # Minimum frequency to consider a word as stopword


class DomainStopwordsFilter:
    """Filter stopwords with domain-specific considerations."""
    
    def __init__(self, config: StopwordsConfig = None):
        self.config = config or StopwordsConfig()
        
        # Initialize stopword sets
        self.general_stopwords = set()
        self.domain_stopwords = set()
        self.custom_stopwords = set()
        self.important_terms = set()
        
        self._load_stopwords()
        self._load_important_terms()
    
    def _load_stopwords(self):
        """Load various stopword sets."""
        # Load general English stopwords
        if self.config.use_general_stopwords:
            try:
                import nltk
                from nltk.corpus import stopwords
                self.general_stopwords = set(stopwords.words('english'))
                logger.info(f"Loaded {len(self.general_stopwords)} general stopwords")
            except ImportError:
                logger.warning("NLTK not available, skipping general stopwords")
        
        # Load domain-specific stopwords
        if self.config.use_domain_stopwords:
            domain_file = Path(self.config.domain_stopwords_file)
            if domain_file.exists():
                with open(domain_file, 'r', encoding='utf-8') as f:
                    self.domain_stopwords = set(line.strip().lower() for line in f if line.strip())
                logger.info(f"Loaded {len(self.domain_stopwords)} domain stopwords")
            else:
                logger.warning(f"Domain stopwords file not found: {domain_file}")
        
        # Load custom stopwords
        if self.config.use_custom_stopwords and self.config.custom_stopwords_file:
            custom_file = Path(self.config.custom_stopwords_file)
            if custom_file.exists():
                with open(custom_file, 'r', encoding='utf-8') as f:
                    self.custom_stopwords = set(line.strip().lower() for line in f if line.strip())
                logger.info(f"Loaded {len(self.custom_stopwords)} custom stopwords")
            else:
                logger.warning(f"Custom stopwords file not found: {custom_file}")
    
    def _load_important_terms(self):
        """Load important domain terms that should not be filtered."""
        if not self.config.preserve_important_terms:
            return
        
        # Important vocational education terms
        important_vocational = {
            'nursing', 'medical', 'healthcare', 'clinical', 'patient', 'surgical',
            'therapy', 'diagnostic', 'phlebotomy', 'radiology', 'pharmacy',
            'dental', 'veterinary', 'paramedic', 'cpr', 'bcls', 'acls', 'pals',
            'hipaa', 'osha', 'fda', 'cdc', 'aha', 'certified', 'licensed',
            'registered', 'accredited', 'approved', 'associate', 'bachelor',
            'master', 'doctorate', 'phd', 'bsn', 'rn', 'aos', 'aa', 'as',
            'ba', 'bs', 'ma', 'ms', 'mba', 'diploma', 'certificate',
            'certification', 'license', 'credential', 'vocational', 'technical',
            'career', 'professional', 'computer', 'software', 'hardware',
            'network', 'database', 'programming', 'coding', 'automotive',
            'electrical', 'mechanical', 'construction', 'welding', 'plumbing',
            'culinary', 'hospitality', 'cosmetology', 'massage', 'fitness',
            'wellness', 'communication', 'leadership', 'teamwork',
            'problem-solving', 'critical-thinking', 'time-management',
            'organization', 'attention-to-detail', 'multitasking',
            'customer-service', 'interpersonal', 'analytical', 'creative',
            'technical', 'hands-on', 'state-of-the-art', 'up-to-date',
            'real-world', 'job-ready', 'career-focused', 'industry-standard',
            'cutting-edge', 'high-quality', 'low-cost', 'full-time',
            'part-time', 'on-campus', 'off-campus', 'online', 'in-person',
            'hybrid', 'blended'
        }
        
        self.important_terms = important_vocational
        logger.info(f"Loaded {len(self.important_terms)} important terms")
    
    def _is_important_term(self, word: str) -> bool:
        """Check if a word is an important domain term."""
        if not self.config.preserve_important_terms:
            return False
        
        word_lower = word.lower()
        
        # Check exact match
        if word_lower in self.important_terms:
            return True
        
        # Check if word contains important terms
        for important_term in self.important_terms:
            if important_term in word_lower or word_lower in important_term:
                return True
        
        return False
    
    def _is_stopword(self, word: str) -> bool:
        """Check if a word should be filtered as a stopword."""
        word_lower = word.lower()
        
        # Don't filter important terms
        if self._is_important_term(word):
            return False
        
        # Check general stopwords
        if word_lower in self.general_stopwords:
            return True
        
        # Check domain stopwords
        if word_lower in self.domain_stopwords:
            return True
        
        # Check custom stopwords
        if word_lower in self.custom_stopwords:
            return True
        
        return False
    
    def filter_tokens(self, tokens: List[str]) -> List[str]:
        """
        Filter stopwords from a list of tokens.
        
        Args:
            tokens: List of tokens to filter
            
        Returns:
            List of filtered tokens
        """
        if not tokens:
            return []
        
        filtered_tokens = []
        for token in tokens:
            if not self._is_stopword(token):
                filtered_tokens.append(token)
        
        return filtered_tokens
    
    def filter_text_columns(self, df: pd.DataFrame, token_columns: List[str]) -> pd.DataFrame:
        """
        Filter stopwords from token columns in a DataFrame.
        
        Args:
            df: Input DataFrame
            token_columns: List of column names containing token lists
            
        Returns:
            DataFrame with filtered tokens
        """
        df_filtered = df.copy()
        
        for col in token_columns:
            if col in df.columns:
                logger.info(f"Filtering stopwords from column: {col}")
                
                # Filter tokens
                df_filtered[f'{col}_filtered'] = df_filtered[col].apply(self.filter_tokens)
                
                # Calculate statistics
                df_filtered[f'{col}_original_count'] = df_filtered[col].apply(len)
                df_filtered[f'{col}_filtered_count'] = df_filtered[f'{col}_filtered'].apply(len)
                df_filtered[f'{col}_removed_count'] = (
                    df_filtered[f'{col}_original_count'] - df_filtered[f'{col}_filtered_count']
                )
        
        return df_filtered
    
    def get_stopword_statistics(self, df: pd.DataFrame, token_columns: List[str]) -> Dict[str, Any]:
        """
        Get statistics about stopword filtering.
        
        Args:
            df: DataFrame with token data
            token_columns: List of column names containing token lists
            
        Returns:
            Dictionary with filtering statistics
        """
        stats = {
            'total_stopwords': len(self.general_stopwords | self.domain_stopwords | self.custom_stopwords),
            'general_stopwords': len(self.general_stopwords),
            'domain_stopwords': len(self.domain_stopwords),
            'custom_stopwords': len(self.custom_stopwords),
            'important_terms': len(self.important_terms),
            'column_stats': {}
        }
        
        for col in token_columns:
            if col in df.columns:
                col_stats = {
                    'total_tokens': df[col].apply(len).sum(),
                    'unique_tokens': len(set(token for tokens in df[col] for token in tokens)),
                    'avg_tokens_per_record': df[col].apply(len).mean(),
                    'max_tokens_per_record': df[col].apply(len).max(),
                    'min_tokens_per_record': df[col].apply(len).min()
                }
                
                # If filtered column exists, add filtering stats
                filtered_col = f'{col}_filtered'
                if filtered_col in df.columns:
                    col_stats.update({
                        'total_filtered_tokens': df[filtered_col].apply(len).sum(),
                        'unique_filtered_tokens': len(set(token for tokens in df[filtered_col] for token in tokens)),
                        'avg_filtered_tokens_per_record': df[filtered_col].apply(len).mean(),
                        'removal_rate': (col_stats['total_tokens'] - col_stats['total_filtered_tokens']) / col_stats['total_tokens'] if col_stats['total_tokens'] > 0 else 0
                    })
                
                stats['column_stats'][col] = col_stats
        
        return stats
    
    def create_stopword_report(self, df: pd.DataFrame, token_columns: List[str]) -> str:
        """
        Create a detailed report about stopword filtering.
        
        Args:
            df: DataFrame with token data
            token_columns: List of column names containing token lists
            
        Returns:
            Formatted report string
        """
        stats = self.get_stopword_statistics(df, token_columns)
        
        report = []
        report.append("=== STOPWORD FILTERING REPORT ===")
        report.append(f"Total stopwords loaded: {stats['total_stopwords']}")
        report.append(f"  - General stopwords: {stats['general_stopwords']}")
        report.append(f"  - Domain stopwords: {stats['domain_stopwords']}")
        report.append(f"  - Custom stopwords: {stats['custom_stopwords']}")
        report.append(f"Important terms preserved: {stats['important_terms']}")
        report.append("")
        
        for col, col_stats in stats['column_stats'].items():
            report.append(f"Column: {col}")
            report.append(f"  Total tokens: {col_stats['total_tokens']:,}")
            report.append(f"  Unique tokens: {col_stats['unique_tokens']:,}")
            report.append(f"  Avg tokens per record: {col_stats['avg_tokens_per_record']:.1f}")
            
            if 'total_filtered_tokens' in col_stats:
                report.append(f"  Filtered tokens: {col_stats['total_filtered_tokens']:,}")
                report.append(f"  Removal rate: {col_stats['removal_rate']:.1%}")
            
            report.append("")
        
        return "\n".join(report)


def filter_programs_stopwords(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter stopwords from program data with domain-specific considerations.
    
    Args:
        df: DataFrame containing program data with tokenized text
        
    Returns:
        DataFrame with filtered tokens
    """
    config = StopwordsConfig(
        use_general_stopwords=True,
        use_domain_stopwords=True,
        use_custom_stopwords=False,  # No custom file specified
        preserve_important_terms=True
    )
    
    filterer = DomainStopwordsFilter(config)
    
    # Filter token columns
    token_columns = []
    for col in df.columns:
        if col.endswith('_tokens') or col.endswith('_lemmatized'):
            token_columns.append(col)
    
    if token_columns:
        filtered_df = filterer.filter_text_columns(df, token_columns)
        
        # Generate report
        report = filterer.create_stopword_report(df, token_columns)
        logger.info(f"Stopword filtering report:\n{report}")
        
        return filtered_df
    else:
        logger.warning("No token columns found for stopword filtering")
        return df


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Filter stopwords from tokenized data")
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--output", required=True, help="Output parquet file")
    parser.add_argument("--token-columns", nargs='+', 
                       help="Columns containing token lists to filter")
    parser.add_argument("--domain-stopwords", 
                       default="data/interim/domain_stopwords.txt",
                       help="Path to domain stopwords file")
    parser.add_argument("--custom-stopwords", 
                       help="Path to custom stopwords file")
    parser.add_argument("--no-general", action="store_true",
                       help="Don't use general English stopwords")
    parser.add_argument("--no-domain", action="store_true",
                       help="Don't use domain-specific stopwords")
    parser.add_argument("--no-important", action="store_true",
                       help="Don't preserve important terms")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Load data
    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df)} records")
    
    # Configure stopwords filter
    config = StopwordsConfig(
        use_general_stopwords=not args.no_general,
        use_domain_stopwords=not args.no_domain,
        use_custom_stopwords=bool(args.custom_stopwords),
        domain_stopwords_file=args.domain_stopwords,
        custom_stopwords_file=args.custom_stopwords,
        preserve_important_terms=not args.no_important
    )
    
    filterer = DomainStopwordsFilter(config)
    
    # Determine token columns
    if args.token_columns:
        token_columns = args.token_columns
    else:
        # Auto-detect token columns
        token_columns = [col for col in df.columns if col.endswith('_tokens') or col.endswith('_lemmatized')]
    
    if not token_columns:
        logger.error("No token columns found. Specify with --token-columns")
        return
    
    # Filter stopwords
    filtered_df = filterer.filter_text_columns(df, token_columns)
    
    # Generate and save report
    report = filterer.create_stopword_report(df, token_columns)
    report_file = args.output.replace('.parquet', '_stopwords_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Save filtered data
    filtered_df.to_parquet(args.output, index=False)
    logger.info(f"Saved filtered data to {args.output}")
    logger.info(f"Saved stopwords report to {report_file}")


if __name__ == "__main__":
    main()
