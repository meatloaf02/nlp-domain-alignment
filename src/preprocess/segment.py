"""
Text segmentation utilities for breaking down long documents into meaningful chunks.
"""

import re
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SegmentationConfig:
    """Configuration for text segmentation."""
    max_chunk_size: int = 1000  # Maximum characters per chunk
    min_chunk_size: int = 200   # Minimum characters per chunk
    overlap_size: int = 100     # Overlap between chunks
    split_on_sentences: bool = True
    split_on_paragraphs: bool = True
    preserve_structure: bool = True


class TextSegmenter:
    """Segment long text documents into smaller, meaningful chunks."""
    
    def __init__(self, config: SegmentationConfig = None):
        self.config = config or SegmentationConfig()
        
        # Compile regex patterns for efficient splitting
        self.sentence_endings = re.compile(r'[.!?]+\s+')
        self.paragraph_separators = re.compile(r'\n\s*\n')
        self.section_headers = re.compile(r'\n\s*[A-Z][A-Z\s]+\n', re.MULTILINE)
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        if not text or not isinstance(text, str):
            return []
        
        # Split on sentence endings
        sentences = self.sentence_endings.split(text)
        # Clean up and filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        if not text or not isinstance(text, str):
            return []
        
        # Split on paragraph separators
        paragraphs = self.paragraph_separators.split(text)
        # Clean up and filter empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return paragraphs
    
    def identify_sections(self, text: str) -> List[Tuple[str, int, int]]:
        """Identify section headers and their positions."""
        sections = []
        for match in self.section_headers.finditer(text):
            start = match.start()
            end = match.end()
            header = match.group().strip()
            sections.append((header, start, end))
        return sections
    
    def create_chunks_from_sentences(self, sentences: List[str]) -> List[str]:
        """Create chunks from sentences respecting size constraints."""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If adding this sentence would exceed max size, finalize current chunk
            if current_size + sentence_size > self.config.max_chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) >= self.config.min_chunk_size:
                    chunks.append(chunk_text)
                
                # Start new chunk with overlap
                overlap_sentences = []
                overlap_size = 0
                for sent in reversed(current_chunk):
                    if overlap_size + len(sent) <= self.config.overlap_size:
                        overlap_sentences.insert(0, sent)
                        overlap_size += len(sent)
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_size = overlap_size
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= self.config.min_chunk_size:
                chunks.append(chunk_text)
        
        return chunks
    
    def create_chunks_from_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """Create chunks from paragraphs respecting size constraints."""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for paragraph in paragraphs:
            paragraph_size = len(paragraph)
            
            # If paragraph is too large, split it further
            if paragraph_size > self.config.max_chunk_size:
                # Finalize current chunk if it exists
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    if len(chunk_text) >= self.config.min_chunk_size:
                        chunks.append(chunk_text)
                    current_chunk = []
                    current_size = 0
                
                # Split large paragraph into sentences and create chunks
                sentences = self.split_into_sentences(paragraph)
                sentence_chunks = self.create_chunks_from_sentences(sentences)
                chunks.extend(sentence_chunks)
                continue
            
            # If adding this paragraph would exceed max size, finalize current chunk
            if current_size + paragraph_size > self.config.max_chunk_size and current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                if len(chunk_text) >= self.config.min_chunk_size:
                    chunks.append(chunk_text)
                
                # Start new chunk with overlap
                overlap_paragraphs = []
                overlap_size = 0
                for para in reversed(current_chunk):
                    if overlap_size + len(para) <= self.config.overlap_size:
                        overlap_paragraphs.insert(0, para)
                        overlap_size += len(para)
                    else:
                        break
                
                current_chunk = overlap_paragraphs
                current_size = overlap_size
            
            current_chunk.append(paragraph)
            current_size += paragraph_size
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            if len(chunk_text) >= self.config.min_chunk_size:
                chunks.append(chunk_text)
        
        return chunks
    
    def segment_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Segment text into chunks with metadata.
        
        Args:
            text: Input text to segment
            
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        if not text or not isinstance(text, str):
            return []
        
        chunks = []
        
        if self.config.split_on_paragraphs:
            paragraphs = self.split_into_paragraphs(text)
            chunk_texts = self.create_chunks_from_paragraphs(paragraphs)
        elif self.config.split_on_sentences:
            sentences = self.split_into_sentences(text)
            chunk_texts = self.create_chunks_from_sentences(sentences)
        else:
            # Simple character-based chunking
            chunk_texts = []
            for i in range(0, len(text), self.config.max_chunk_size - self.config.overlap_size):
                chunk = text[i:i + self.config.max_chunk_size]
                if len(chunk) >= self.config.min_chunk_size:
                    chunk_texts.append(chunk)
        
        # Create chunk metadata
        for i, chunk_text in enumerate(chunk_texts):
            chunk_info = {
                'chunk_id': i,
                'text': chunk_text,
                'length': len(chunk_text),
                'word_count': len(chunk_text.split()),
                'char_count': len(chunk_text)
            }
            chunks.append(chunk_info)
        
        return chunks
    
    def segment_dataframe(self, df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
        """
        Segment text columns in a DataFrame.
        
        Args:
            df: Input DataFrame
            text_columns: List of column names containing text to segment
            
        Returns:
            DataFrame with segmented text data
        """
        segmented_data = []
        
        for _, row in df.iterrows():
            base_record = row.to_dict()
            
            for col in text_columns:
                if col in df.columns and pd.notna(row[col]):
                    text = str(row[col])
                    chunks = self.segment_text(text)
                    
                    if chunks:
                        # Create separate records for each chunk
                        for chunk in chunks:
                            record = base_record.copy()
                            record[f'{col}_chunk'] = chunk['text']
                            record[f'{col}_chunk_id'] = chunk['chunk_id']
                            record[f'{col}_chunk_length'] = chunk['length']
                            record[f'{col}_chunk_word_count'] = chunk['word_count']
                            segmented_data.append(record)
                    else:
                        # No chunks created, keep original
                        segmented_data.append(base_record)
                else:
                    # No text in this column, keep original
                    segmented_data.append(base_record)
        
        return pd.DataFrame(segmented_data)


def segment_programs_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Segment program description data.
    
    Args:
        df: DataFrame containing program data
        
    Returns:
        DataFrame with segmented text data
    """
    config = SegmentationConfig(
        max_chunk_size=800,  # Smaller chunks for program descriptions
        min_chunk_size=150,
        overlap_size=50,
        split_on_paragraphs=True
    )
    
    segmenter = TextSegmenter(config)
    
    # Segment description fields
    text_columns = ['description_raw', 'description_text']
    segmented_df = segmenter.segment_dataframe(df, text_columns)
    
    logger.info(f"Segmented {len(df)} records into {len(segmented_df)} chunks")
    
    return segmented_df


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Segment text data into chunks")
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--output", required=True, help="Output parquet file")
    parser.add_argument("--columns", nargs='+', default=['description_raw', 'description_text'], 
                       help="Columns to segment")
    parser.add_argument("--max-chunk-size", type=int, default=800, 
                       help="Maximum chunk size in characters")
    parser.add_argument("--min-chunk-size", type=int, default=150, 
                       help="Minimum chunk size in characters")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Load data
    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df)} records")
    
    # Configure segmenter
    config = SegmentationConfig(
        max_chunk_size=args.max_chunk_size,
        min_chunk_size=args.min_chunk_size
    )
    segmenter = TextSegmenter(config)
    
    # Segment text
    segmented_df = segmenter.segment_dataframe(df, args.columns)
    
    # Save segmented data
    segmented_df.to_parquet(args.output, index=False)
    logger.info(f"Saved segmented data to {args.output}")


if __name__ == "__main__":
    main()
