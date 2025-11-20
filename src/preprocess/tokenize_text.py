"""
Enhanced tokenization and lemmatization utilities with domain-specific considerations.
"""

import re
import string
import pandas as pd
from typing import List, Dict, Any, Optional, Set, Tuple
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import spacy
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')


@dataclass
class TokenizationConfig:
    """Configuration for tokenization and lemmatization."""
    use_spacy: bool = True
    use_lemmatization: bool = True
    use_stemming: bool = False
    preserve_case: bool = False
    min_token_length: int = 2
    max_token_length: int = 50
    remove_numbers: bool = False
    remove_punctuation: bool = True
    preserve_hyphenated: bool = True
    preserve_contractions: bool = True
    domain_specific_patterns: bool = True


class DomainAwareTokenizer:
    """Enhanced tokenizer with domain-specific considerations for vocational programs."""
    
    def __init__(self, config: TokenizationConfig = None):
        self.config = config or TokenizationConfig()
        
        # Initialize tokenizers and lemmatizers
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Try to load spaCy model
        self.nlp = None
        if self.config.use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
                self.config.use_spacy = False
        
        # Domain-specific patterns
        self._compile_domain_patterns()
    
    def _compile_domain_patterns(self):
        """Compile regex patterns for domain-specific tokenization."""
        if not self.config.domain_specific_patterns:
            return
        
        # Educational degree patterns
        self.degree_patterns = [
            r'\b(?:associate|bachelor|master|doctorate|phd|bsn|rn|aos|aa|as|ba|bs|ma|ms|mba)\b',
            r'\b(?:diploma|certificate|certification|license|credential)\b',
            r'\b(?:vocational|technical|career|professional)\b'
        ]
        
        # Medical/healthcare patterns
        self.medical_patterns = [
            r'\b(?:nursing|medical|healthcare|clinical|patient|surgical|therapy|diagnostic)\b',
            r'\b(?:phlebotomy|radiology|pharmacy|dental|veterinary|paramedic)\b',
            r'\b(?:cpr|bcls|acls|pals|hipaa|osha)\b'
        ]
        
        # Technical/skills patterns
        self.technical_patterns = [
            r'\b(?:computer|software|hardware|network|database|programming|coding)\b',
            r'\b(?:automotive|electrical|mechanical|construction|welding|plumbing)\b',
            r'\b(?:culinary|hospitality|cosmetology|massage|fitness|wellness)\b'
        ]
        
        # Compile patterns
        self.degree_regex = re.compile('|'.join(self.degree_patterns), re.IGNORECASE)
        self.medical_regex = re.compile('|'.join(self.medical_patterns), re.IGNORECASE)
        self.technical_regex = re.compile('|'.join(self.technical_patterns), re.IGNORECASE)
    
    def _preserve_hyphenated_words(self, text: str) -> str:
        """Preserve hyphenated words that might be important domain terms."""
        if not self.config.preserve_hyphenated:
            return text
        
        # Common hyphenated terms in vocational education
        hyphenated_terms = [
            'hands-on', 'state-of-the-art', 'up-to-date', 'real-world',
            'job-ready', 'career-focused', 'industry-standard', 'cutting-edge',
            'high-quality', 'low-cost', 'full-time', 'part-time', 'on-campus',
            'off-campus', 'online', 'in-person', 'hybrid', 'blended'
        ]
        
        for term in hyphenated_terms:
            # Replace spaces around hyphenated terms
            pattern = term.replace('-', r'\s*-\s*')
            text = re.sub(pattern, term, text, flags=re.IGNORECASE)
        
        return text
    
    def _preserve_contractions(self, text: str) -> str:
        """Preserve contractions that might be important."""
        if not self.config.preserve_contractions:
            return text
        
        # Common contractions
        contractions = {
            "don't": "do not", "won't": "will not", "can't": "cannot",
            "n't": " not", "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'s": " is", "'m": " am"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        return text
    
    def _extract_domain_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract domain-specific entities from text."""
        entities = {
            'degrees': [],
            'medical_terms': [],
            'technical_terms': [],
            'skills': [],
            'certifications': []
        }
        
        if not self.config.domain_specific_patterns:
            return entities
        
        # Extract degree-related terms
        degree_matches = self.degree_regex.findall(text)
        entities['degrees'] = list(set(degree_matches))
        
        # Extract medical terms
        medical_matches = self.medical_regex.findall(text)
        entities['medical_terms'] = list(set(medical_matches))
        
        # Extract technical terms
        technical_matches = self.technical_regex.findall(text)
        entities['technical_terms'] = list(set(technical_matches))
        
        # Extract skills (common skill patterns)
        skill_patterns = [
            r'\b(?:communication|leadership|teamwork|problem.?solving|critical.?thinking)\b',
            r'\b(?:time.?management|organization|attention.?to.?detail|multitasking)\b',
            r'\b(?:customer.?service|interpersonal|analytical|creative|technical)\b'
        ]
        skill_regex = re.compile('|'.join(skill_patterns), re.IGNORECASE)
        skill_matches = skill_regex.findall(text)
        entities['skills'] = list(set(skill_matches))
        
        # Extract certifications
        cert_patterns = [
            r'\b(?:cpr|bcls|acls|pals|hipaa|osha|fda|cdc|aha)\b',
            r'\b(?:certified|licensed|registered|accredited|approved)\b'
        ]
        cert_regex = re.compile('|'.join(cert_patterns), re.IGNORECASE)
        cert_matches = cert_regex.findall(text)
        entities['certifications'] = list(set(cert_matches))
        
        return entities
    
    def tokenize_with_spacy(self, text: str) -> List[Dict[str, Any]]:
        """Tokenize text using spaCy for better linguistic analysis."""
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        tokens = []
        
        for token in doc:
            if token.is_space or token.is_punct:
                continue
            
            token_info = {
                'text': token.text.lower() if not self.config.preserve_case else token.text,
                'lemma': token.lemma_.lower() if not self.config.preserve_case else token.lemma_,
                'pos': token.pos_,
                'tag': token.tag_,
                'is_alpha': token.is_alpha,
                'is_stop': token.is_stop,
                'length': len(token.text)
            }
            
            # Filter by length
            if (self.config.min_token_length <= token_info['length'] <= self.config.max_token_length):
                tokens.append(token_info)
        
        return tokens
    
    def tokenize_with_nltk(self, text: str) -> List[str]:
        """Tokenize text using NLTK."""
        # Preprocess text
        text = self._preserve_hyphenated_words(text)
        text = self._preserve_contractions(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Filter tokens
        filtered_tokens = []
        for token in tokens:
            # Skip punctuation if configured
            if self.config.remove_punctuation and token in string.punctuation:
                continue
            
            # Skip numbers if configured
            if self.config.remove_numbers and token.isdigit():
                continue
            
            # Filter by length
            if self.config.min_token_length <= len(token) <= self.config.max_token_length:
                # Convert to lowercase if not preserving case
                if not self.config.preserve_case:
                    token = token.lower()
                filtered_tokens.append(token)
        
        return filtered_tokens
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens using WordNet."""
        if not self.config.use_lemmatization:
            return tokens
        
        lemmatized = []
        for token in tokens:
            # Try different POS tags for better lemmatization
            lemma = self.lemmatizer.lemmatize(token, pos='v')  # Try verb first
            if lemma == token:
                lemma = self.lemmatizer.lemmatize(token, pos='n')  # Try noun
            if lemma == token:
                lemma = self.lemmatizer.lemmatize(token, pos='a')  # Try adjective
            if lemma == token:
                lemma = self.lemmatizer.lemmatize(token)  # Default
            
            lemmatized.append(lemma)
        
        return lemmatized
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """Stem tokens using Porter stemmer."""
        if not self.config.use_stemming:
            return tokens
        
        return [self.stemmer.stem(token) for token in tokens]
    
    def tokenize_text(self, text: str) -> Dict[str, Any]:
        """
        Tokenize text with domain-specific considerations.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            Dictionary containing tokens and metadata
        """
        if not text or not isinstance(text, str):
            return {
                'tokens': [],
                'lemmatized_tokens': [],
                'stemmed_tokens': [],
                'unique_tokens': [],
                'unique_lemmatized_tokens': [],
                'unique_stemmed_tokens': [],
                'entities': {},
                'token_count': 0,
                'unique_token_count': 0
            }
        
        # Extract domain entities
        entities = self._extract_domain_entities(text)
        
        # Tokenize
        if self.config.use_spacy and self.nlp:
            spacy_tokens = self.tokenize_with_spacy(text)
            tokens = [t['text'] for t in spacy_tokens]
        else:
            tokens = self.tokenize_with_nltk(text)
        
        # Process tokens
        lemmatized_tokens = self.lemmatize_tokens(tokens)
        stemmed_tokens = self.stem_tokens(tokens)
        
        # Remove duplicates while preserving order
        unique_tokens = list(dict.fromkeys(tokens))
        unique_lemmatized = list(dict.fromkeys(lemmatized_tokens))
        unique_stemmed = list(dict.fromkeys(stemmed_tokens))
        
        return {
            'tokens': tokens,
            'lemmatized_tokens': lemmatized_tokens,
            'stemmed_tokens': stemmed_tokens,
            'unique_tokens': unique_tokens,
            'unique_lemmatized_tokens': unique_lemmatized,
            'unique_stemmed_tokens': unique_stemmed,
            'entities': entities,
            'token_count': len(tokens),
            'unique_token_count': len(unique_tokens)
        }
    
    def tokenize_dataframe(self, df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
        """
        Tokenize text columns in a DataFrame.
        
        Args:
            df: Input DataFrame
            text_columns: List of column names containing text to tokenize
            
        Returns:
            DataFrame with tokenized text data
        """
        df_tokenized = df.copy()
        
        for col in text_columns:
            if col in df.columns:
                logger.info(f"Tokenizing column: {col}")
                
                # Create new columns for tokenized data
                tokenized_data = df_tokenized[col].apply(self.tokenize_text)
                
                # Extract token information
                df_tokenized[f'{col}_tokens'] = tokenized_data.apply(lambda x: x['tokens'])
                df_tokenized[f'{col}_lemmatized'] = tokenized_data.apply(lambda x: x['lemmatized_tokens'])
                df_tokenized[f'{col}_stemmed'] = tokenized_data.apply(lambda x: x['stemmed_tokens'])
                df_tokenized[f'{col}_unique_tokens'] = tokenized_data.apply(lambda x: x['unique_tokens'])
                df_tokenized[f'{col}_token_count'] = tokenized_data.apply(lambda x: x['token_count'])
                df_tokenized[f'{col}_unique_token_count'] = tokenized_data.apply(lambda x: x['unique_token_count'])
                
                # Extract entities
                for entity_type in ['degrees', 'medical_terms', 'technical_terms', 'skills', 'certifications']:
                    df_tokenized[f'{col}_{entity_type}'] = tokenized_data.apply(
                        lambda x: x['entities'].get(entity_type, [])
                    )
        
        return df_tokenized


def tokenize_programs_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tokenize program description data with domain-specific considerations.
    
    Args:
        df: DataFrame containing program data
        
    Returns:
        DataFrame with tokenized text data
    """
    config = TokenizationConfig(
        use_spacy=True,
        use_lemmatization=True,
        use_stemming=False,
        preserve_case=False,
        min_token_length=2,
        max_token_length=50,
        remove_numbers=False,  # Keep numbers for program codes, years, etc.
        remove_punctuation=True,
        preserve_hyphenated=True,
        preserve_contractions=True,
        domain_specific_patterns=True
    )
    
    tokenizer = DomainAwareTokenizer(config)
    
    # Tokenize description fields
    text_columns = ['description_raw', 'description_text', 'program_name']
    tokenized_df = tokenizer.tokenize_dataframe(df, text_columns)
    
    logger.info(f"Tokenized {len(df)} records")
    
    return tokenized_df


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Tokenize text data with domain-specific considerations")
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--output", required=True, help="Output parquet file")
    parser.add_argument("--columns", nargs='+', default=['description_raw', 'description_text'], 
                       help="Columns to tokenize")
    parser.add_argument("--use-spacy", action="store_true", default=True,
                       help="Use spaCy for tokenization")
    parser.add_argument("--use-lemmatization", action="store_true", default=True,
                       help="Use lemmatization")
    parser.add_argument("--use-stemming", action="store_true", default=False,
                       help="Use stemming")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Load data
    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df)} records")
    
    # Configure tokenizer
    config = TokenizationConfig(
        use_spacy=args.use_spacy,
        use_lemmatization=args.use_lemmatization,
        use_stemming=args.use_stemming
    )
    tokenizer = DomainAwareTokenizer(config)
    
    # Tokenize text
    tokenized_df = tokenizer.tokenize_dataframe(df, args.columns)
    
    # Save tokenized data
    tokenized_df.to_parquet(args.output, index=False)
    logger.info(f"Saved tokenized data to {args.output}")


if __name__ == "__main__":
    main()
