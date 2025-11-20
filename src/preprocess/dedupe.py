"""
Deduplication utilities for job postings.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DeduplicationConfig:
    """Configuration for deduplication."""
    similarity_threshold: float = 0.8
    min_similarity: float = 0.5
    use_title: bool = True
    use_description: bool = True
    use_company: bool = True
    use_location: bool = False
    max_features: int = 10000
    ngram_range: Tuple[int, int] = (1, 2)


class JobDeduplicator:
    """Deduplicate job postings based on content similarity."""
    
    def __init__(self, config: DeduplicationConfig = None):
        self.config = config or DeduplicationConfig()
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            ngram_range=self.config.ngram_range,
            stop_words='english'
        )
        self.similarity_matrix = None
        self.duplicate_groups = []
    
    def _create_text_features(self, df: pd.DataFrame) -> List[str]:
        """Create combined text features for similarity comparison."""
        features = []
        
        for _, row in df.iterrows():
            text_parts = []
            
            if self.config.use_title and pd.notna(row.get('title')):
                text_parts.append(str(row['title']))
            
            if self.config.use_description and pd.notna(row.get('description')):
                text_parts.append(str(row['description']))
            
            if self.config.use_company and pd.notna(row.get('company')):
                text_parts.append(str(row['company']))
            
            if self.config.use_location and pd.notna(row.get('location')):
                text_parts.append(str(row['location']))
            
            features.append(' '.join(text_parts))
        
        return features
    
    def _create_hash_features(self, df: pd.DataFrame) -> List[str]:
        """Create hash-based features for exact duplicate detection."""
        hash_features = []
        
        for _, row in df.iterrows():
            # Create hash from key fields
            key_fields = []
            if pd.notna(row.get('title')):
                key_fields.append(str(row['title']).lower().strip())
            if pd.notna(row.get('company')):
                key_fields.append(str(row['company']).lower().strip())
            if pd.notna(row.get('location')):
                key_fields.append(str(row['location']).lower().strip())
            
            # Create hash
            combined = '|'.join(key_fields)
            hash_value = hashlib.md5(combined.encode()).hexdigest()
            hash_features.append(hash_value)
        
        return hash_features
    
    def find_exact_duplicates(self, df: pd.DataFrame) -> List[List[int]]:
        """Find exact duplicates based on hash comparison."""
        hash_features = self._create_hash_features(df)
        
        # Group by hash
        hash_groups = {}
        for idx, hash_val in enumerate(hash_features):
            if hash_val not in hash_groups:
                hash_groups[hash_val] = []
            hash_groups[hash_val].append(idx)
        
        # Return groups with more than one item
        duplicate_groups = [group for group in hash_groups.values() if len(group) > 1]
        
        logger.info(f"Found {len(duplicate_groups)} exact duplicate groups")
        return duplicate_groups
    
    def find_similar_duplicates(self, df: pd.DataFrame) -> List[List[int]]:
        """Find similar duplicates based on text similarity."""
        text_features = self._create_text_features(df)
        
        # Vectorize text
        tfidf_matrix = self.vectorizer.fit_transform(text_features)
        
        # Calculate similarity matrix
        self.similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Find similar pairs
        similar_pairs = []
        n = len(df)
        
        for i in range(n):
            for j in range(i + 1, n):
                similarity = self.similarity_matrix[i, j]
                if similarity >= self.config.similarity_threshold:
                    similar_pairs.append((i, j, similarity))
        
        # Group similar pairs
        duplicate_groups = self._group_similar_pairs(similar_pairs, n)
        
        logger.info(f"Found {len(duplicate_groups)} similar duplicate groups")
        return duplicate_groups
    
    def _group_similar_pairs(self, similar_pairs: List[Tuple[int, int, float]], n: int) -> List[List[int]]:
        """Group similar pairs into duplicate groups."""
        # Sort by similarity score (descending)
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Union-find data structure
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Union similar pairs
        for i, j, similarity in similar_pairs:
            if similarity >= self.config.similarity_threshold:
                union(i, j)
        
        # Group indices by root parent
        groups = {}
        for i in range(n):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
        
        # Return groups with more than one item
        duplicate_groups = [group for group in groups.values() if len(group) > 1]
        
        return duplicate_groups
    
    def deduplicate(self, df: pd.DataFrame, 
                   remove_exact: bool = True,
                   remove_similar: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Remove duplicates from DataFrame.
        
        Args:
            df: Input DataFrame
            remove_exact: Whether to remove exact duplicates
            remove_similar: Whether to remove similar duplicates
            
        Returns:
            Tuple of (deduplicated_df, stats)
        """
        original_count = len(df)
        df_clean = df.copy()
        
        stats = {
            'original_count': original_count,
            'exact_duplicates_removed': 0,
            'similar_duplicates_removed': 0,
            'final_count': 0
        }
        
        # Remove exact duplicates
        if remove_exact:
            exact_groups = self.find_exact_duplicates(df_clean)
            indices_to_remove = set()
            
            for group in exact_groups:
                # Keep the first item, remove the rest
                indices_to_remove.update(group[1:])
            
            df_clean = df_clean.drop(index=indices_to_remove).reset_index(drop=True)
            stats['exact_duplicates_removed'] = len(indices_to_remove)
            logger.info(f"Removed {len(indices_to_remove)} exact duplicates")
        
        # Remove similar duplicates
        if remove_similar:
            similar_groups = self.find_similar_duplicates(df_clean)
            indices_to_remove = set()
            
            for group in similar_groups:
                # Keep the first item, remove the rest
                indices_to_remove.update(group[1:])
            
            df_clean = df_clean.drop(index=indices_to_remove).reset_index(drop=True)
            stats['similar_duplicates_removed'] = len(indices_to_remove)
            logger.info(f"Removed {len(indices_to_remove)} similar duplicates")
        
        stats['final_count'] = len(df_clean)
        stats['total_removed'] = original_count - len(df_clean)
        
        logger.info(f"Deduplication complete: {original_count} -> {len(df_clean)} records")
        
        return df_clean, stats
    
    def get_duplicate_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get detailed analysis of duplicates."""
        exact_groups = self.find_exact_duplicates(df)
        similar_groups = self.find_similar_duplicates(df)
        
        analysis_data = []
        
        # Analyze exact duplicates
        for group in exact_groups:
            for idx in group:
                analysis_data.append({
                    'index': idx,
                    'duplicate_type': 'exact',
                    'group_size': len(group),
                    'title': df.iloc[idx].get('title', ''),
                    'company': df.iloc[idx].get('company', ''),
                    'similarity_score': 1.0
                })
        
        # Analyze similar duplicates
        for group in similar_groups:
            for i, idx in enumerate(group):
                similarity_scores = []
                for j, other_idx in enumerate(group):
                    if i != j:
                        similarity = self.similarity_matrix[idx, other_idx]
                        similarity_scores.append(similarity)
                
                avg_similarity = np.mean(similarity_scores) if similarity_scores else 0
                
                analysis_data.append({
                    'index': idx,
                    'duplicate_type': 'similar',
                    'group_size': len(group),
                    'title': df.iloc[idx].get('title', ''),
                    'company': df.iloc[idx].get('company', ''),
                    'similarity_score': avg_similarity
                })
        
        return pd.DataFrame(analysis_data)


def main():
    """Main function for command line usage."""
    import argparse
    from utils import load_data, save_data
    
    parser = argparse.ArgumentParser(description="Deduplicate job postings")
    parser.add_argument("--input", default="data/interim/cleaned.parquet", help="Input parquet file")
    parser.add_argument("--output", default="data/interim/deduped.parquet", help="Output parquet file")
    parser.add_argument("--threshold", type=float, default=0.8, 
                       help="Similarity threshold for duplicates")
    parser.add_argument("--no-exact", action="store_true", 
                       help="Skip exact duplicate removal")
    parser.add_argument("--no-similar", action="store_true", 
                       help="Skip similar duplicate removal")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Load data
    df = load_data(args.input)
    logger.info(f"Loaded {len(df)} records")
    
    # Configure deduplicator
    config = DeduplicationConfig(similarity_threshold=args.threshold)
    deduplicator = JobDeduplicator(config)
    
    # Deduplicate
    df_clean, stats = deduplicator.deduplicate(
        df, 
        remove_exact=not args.no_exact,
        remove_similar=not args.no_similar
    )
    
    # Save results
    save_data(df_clean, args.output)
    logger.info(f"Saved deduplicated data to {args.output}")
    logger.info(f"Stats: {stats}")


if __name__ == "__main__":
    main()

