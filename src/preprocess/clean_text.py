"""
Text cleaning and preprocessing utilities for job postings.
"""

import re
import string
import html
from typing import List, Optional, Dict, Any
import pandas as pd
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import logging

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


class TextCleaner:
    """Text cleaning and preprocessing class."""
    
    def __init__(self, 
                 remove_html: bool = True,
                 remove_urls: bool = True,
                 remove_emails: bool = True,
                 remove_phone_numbers: bool = True,
                 remove_special_chars: bool = True,
                 lowercase: bool = True,
                 remove_stopwords: bool = True,
                 lemmatize: bool = True,
                 min_length: int = 3):
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_phone_numbers = remove_phone_numbers
        self.remove_special_chars = remove_special_chars
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.min_length = min_length
        
        # Initialize NLTK components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}')
    
    def clean_text(self, text: str) -> str:
        """Clean a single text string."""
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # HTML decoding
        text = html.unescape(text)
        
        # Remove HTML tags
        if self.remove_html:
            soup = BeautifulSoup(text, 'html.parser')
            text = soup.get_text()
        
        # Remove URLs
        if self.remove_urls:
            text = self.url_pattern.sub('', text)
        
        # Remove email addresses
        if self.remove_emails:
            text = self.email_pattern.sub('', text)
        
        # Remove phone numbers
        if self.remove_phone_numbers:
            text = self.phone_pattern.sub('', text)
        
        # Remove special characters
        if self.remove_special_chars:
            # Keep alphanumeric, spaces, and basic punctuation
            text = re.sub(r'[^a-zA-Z0-9\s.,!?;:-]', '', text)
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Tokenize and process
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Lemmatize
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Filter by length
        tokens = [token for token in tokens if len(token) >= self.min_length]
        
        # Join tokens back to string
        return ' '.join(tokens)
    
    def clean_dataframe(self, df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
        """Clean text columns in a DataFrame."""
        df_clean = df.copy()
        
        for column in text_columns:
            if column in df_clean.columns:
                logger.info(f"Cleaning column: {column}")
                df_clean[column] = df_clean[column].apply(self.clean_text)
        
        return df_clean
    
    def extract_skills(self, text: str, skills_list: List[str]) -> List[str]:
        """Extract skills from text using a predefined skills list."""
        text_lower = text.lower()
        found_skills = []
        
        for skill in skills_list:
            if skill.lower() in text_lower:
                found_skills.append(skill)
        
        return found_skills
    
    def extract_requirements(self, text: str) -> Dict[str, Any]:
        """Extract job requirements from text."""
        requirements = {
            'experience_years': None,
            'education_level': None,
            'certifications': [],
            'languages': []
        }
        
        # Extract years of experience
        exp_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*years?\s*in\s*(?:the\s*)?field',
            r'(\d+)\+?\s*years?\s*relevant\s*experience'
        ]
        
        for pattern in exp_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                requirements['experience_years'] = int(match.group(1))
                break
        
        # Extract education level
        education_keywords = {
            'phd': ['phd', 'doctorate', 'doctoral'],
            'masters': ['masters', 'master\'s', 'ms', 'ma', 'mba'],
            'bachelors': ['bachelors', 'bachelor\'s', 'bs', 'ba', 'degree'],
            'associates': ['associates', 'associate\'s', 'aa', 'as'],
            'high_school': ['high school', 'hs', 'diploma']
        }
        
        for level, keywords in education_keywords.items():
            if any(keyword in text.lower() for keyword in keywords):
                requirements['education_level'] = level
                break
        
        return requirements


def clean_job_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """Clean job descriptions in a DataFrame."""
    cleaner = TextCleaner()
    return cleaner.clean_dataframe(df, ['description', 'title', 'requirements'])


def extract_structured_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract structured features from job postings."""
    cleaner = TextCleaner()
    
    # Extract requirements
    requirements_data = df['description'].apply(cleaner.extract_requirements)
    requirements_df = pd.DataFrame(requirements_data.tolist())
    
    # Add to original dataframe
    df_with_features = pd.concat([df, requirements_df], axis=1)
    
    return df_with_features


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean job posting text data")
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--output", required=True, help="Output parquet file")
    parser.add_argument("--columns", nargs='+', default=['description', 'title'], 
                       help="Columns to clean")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Load data
    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df)} records")
    
    # Clean text
    df_clean = clean_job_descriptions(df)
    
    # Save cleaned data
    df_clean.to_parquet(args.output, index=False)
    logger.info(f"Saved cleaned data to {args.output}")


if __name__ == "__main__":
    main()

