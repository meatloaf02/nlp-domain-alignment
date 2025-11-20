"""
Scikit-learn baseline models for job posting classification.

This module provides a comprehensive set of traditional machine learning models
for classifying job postings, including logistic regression, random forest, SVM,
and naive bayes classifiers with proper hyperparameter tuning and evaluation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


class JobClassificationBaseline:
    """Baseline job classification models using scikit-learn.
    
    This class provides a comprehensive set of traditional machine learning
    models for job posting classification, including logistic regression,
    random forest, SVM, and naive bayes classifiers.
    
    Args:
        vectorizer_type: Type of text vectorizer ('tfidf' or 'count')
        max_features: Maximum number of features for vectorizer
        ngram_range: Range of n-grams to extract
        random_state: Random state for reproducibility
        
    Attributes:
        vectorizer: Text vectorizer instance
        models: Dictionary of trained models
        label_encoder: Label encoder for target classes
        best_model: Best performing model
        best_model_name: Name of the best model
        is_fitted: Whether the model has been fitted
    """
    
    def __init__(
        self,
        vectorizer_type: str = "tfidf",
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        random_state: int = 42,
    ) -> None:
        """Initialize the JobClassificationBaseline.
        
        Args:
            vectorizer_type: Type of text vectorizer ('tfidf' or 'count')
            max_features: Maximum number of features for vectorizer
            ngram_range: Range of n-grams to extract
            random_state: Random state for reproducibility
        """
        self.vectorizer_type = vectorizer_type
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.random_state = random_state
        
        # Initialize vectorizer
        if vectorizer_type == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words="english",
            )
        elif vectorizer_type == "count":
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words="english",
            )
        else:
            raise ValueError(f"Unknown vectorizer type: {vectorizer_type}")
        
        # Initialize models
        self.models = {
            "logistic_regression": LogisticRegression(
                random_state=random_state, max_iter=1000
            ),
            "random_forest": RandomForestClassifier(
                random_state=random_state, n_estimators=100
            ),
            "svm": SVC(random_state=random_state, probability=True),
            "naive_bayes": MultinomialNB(),
        }
        
        self.label_encoder = LabelEncoder()
        self.best_model: Optional[Any] = None
        self.best_model_name: Optional[str] = None
        self.is_fitted = False
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        text_column: str = "description",
        target_column: str = "category",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training.
        
        Args:
            df: Input DataFrame containing job postings
            text_column: Name of the text column to use for features
            target_column: Name of the target column for classification
            
        Returns:
            Tuple of (X, y) where X is text data and y is encoded labels
            
        Raises:
            KeyError: If specified columns don't exist in DataFrame
        """
        # Extract text and labels
        X = df[text_column].fillna("").astype(str)
        y = df[target_column].fillna("unknown")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        logger.info(f"Prepared {len(X)} samples with {len(np.unique(y_encoded))} classes")
        return X, y_encoded
    
    def train_models(self, X: np.ndarray, y: np.ndarray, 
                    test_size: float = 0.2) -> Dict[str, Dict[str, float]]:
        """Train all baseline models and return performance metrics."""
        # Check if we have enough samples per class for stratification
        from collections import Counter
        class_counts = Counter(y)
        min_samples_per_class = int(1 / test_size) + 1
        stratify_y = y if all(count >= min_samples_per_class for count in class_counts.values()) else None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=stratify_y
        )
        
        # Vectorize text
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train_vec, y_train)
            
            # Predict
            y_pred = model.predict(X_test_vec)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation score - adjust cv based on sample size
            n_samples = len(y_train)
            cv_folds = min(5, max(2, n_samples // 2))  # Use at most 5 folds, at least 2, and ensure at least 2 samples per fold
            if n_samples < 2:
                # Not enough samples for CV, use training accuracy as fallback
                cv_mean = accuracy
                cv_std = 0.0
            else:
                cv_scores = cross_val_score(model, X_train_vec, y_train, cv=cv_folds)
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
            
            results[name] = {
                'accuracy': accuracy,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'model': model
            }
            
            logger.info(f"{name} - Accuracy: {accuracy:.4f}, CV: {cv_mean:.4f} ± {cv_std:.4f}")
        
        # Select best model
        best_accuracy = max(results.values(), key=lambda x: x['accuracy'])['accuracy']
        self.best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
        self.best_model = results[self.best_model_name]['model']
        self.is_fitted = True
        
        logger.info(f"Best model: {self.best_model_name} with accuracy {best_accuracy:.4f}")
        
        return results
    
    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray, 
                             model_name: str = 'logistic_regression') -> Dict[str, Any]:
        """Perform hyperparameter tuning for a specific model."""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Define parameter grids
        param_grids = {
            'logistic_regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            },
            'naive_bayes': {
                'alpha': [0.1, 0.5, 1.0, 2.0]
            }
        }
        
        if model_name not in param_grids:
            logger.warning(f"No parameter grid defined for {model_name}")
            return {}
        
        # Vectorize text
        X_vec = self.vectorizer.fit_transform(X)
        
        # Create pipeline
        pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.models[model_name])
        ])
        
        # Grid search
        grid_search = GridSearchCV(
            pipeline, 
            param_grids[model_name],
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Update best model
        self.best_model = grid_search.best_estimator_
        self.best_model_name = model_name
        self.is_fitted = True
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_vec = self.vectorizer.transform(X)
        y_pred = self.best_model.predict(X_vec)
        
        return y_pred
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_vec = self.vectorizer.transform(X)
        y_proba = self.best_model.predict_proba(X_vec)
        
        return y_proba
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance for the best model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        if hasattr(self.best_model, 'coef_'):
            # Linear models
            importance = self.best_model.coef_[0]
        elif hasattr(self.best_model, 'feature_importances_'):
            # Tree-based models
            importance = self.best_model.feature_importances_
        else:
            logger.warning("Model does not support feature importance")
            return pd.DataFrame()
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save_model(self, filepath: Path) -> None:
        """Save the trained model and vectorizer."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'vectorizer': self.vectorizer,
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'label_encoder': self.label_encoder,
            'vectorizer_type': self.vectorizer_type,
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: Path) -> None:
        """Load a trained model and vectorizer."""
        model_data = joblib.load(filepath)
        
        self.vectorizer = model_data['vectorizer']
        self.best_model = model_data['best_model']
        self.best_model_name = model_data['best_model_name']
        self.label_encoder = model_data['label_encoder']
        self.vectorizer_type = model_data['vectorizer_type']
        self.max_features = model_data['max_features']
        self.ngram_range = model_data['ngram_range']
        self.random_state = model_data['random_state']
        self.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train baseline job classification models")
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--text-column", default="description", help="Text column name")
    parser.add_argument("--target-column", default="category", help="Target column name")
    parser.add_argument("--output-dir", required=True, help="Output directory for models")
    parser.add_argument("--vectorizer", default="tfidf", choices=["tfidf", "count"], 
                       help="Vectorizer type")
    parser.add_argument("--max-features", type=int, default=10000, 
                       help="Maximum number of features")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Load data
    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df)} records")
    
    # Initialize classifier
    classifier = JobClassificationBaseline(
        vectorizer_type=args.vectorizer,
        max_features=args.max_features
    )
    
    # Prepare data
    X, y = classifier.prepare_data(df, args.text_column, args.target_column)
    
    # Train models
    results = classifier.train_models(X, y)
    
    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "sklearn_baseline.joblib"
    classifier.save_model(model_path)
    
    # Print results
    print("\nModel Performance:")
    for name, metrics in results.items():
        print(f"{name}: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
