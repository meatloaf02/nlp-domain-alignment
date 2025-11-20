"""
Domain Classifier Module

This module assigns domain labels to programs and jobs.
For programs, uses manual mapping based on program names.
For jobs, uses keyword-based classification.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class DomainClassifier:
    """Classify programs and jobs into domain categories."""
    
    # Domain categories
    DOMAINS = ['healthcare', 'technical', 'business', 'it/technology', 'other']
    
    def __init__(self):
        """Initialize the domain classifier."""
        # Manual mapping for program names (can be expanded)
        self.program_mappings = {
            # Healthcare
            'nursing': 'healthcare',
            'medical': 'healthcare',
            'healthcare': 'healthcare',
            'health': 'healthcare',
            'patient': 'healthcare',
            'clinical': 'healthcare',
            'dental': 'healthcare',
            'pharmacy': 'healthcare',
            'veterinary': 'healthcare',
            'massage': 'healthcare',
            'therapist': 'healthcare',
            'assistant': 'healthcare',  # Medical assistant
            
            # IT/Technology
            'computer': 'it/technology',
            'software': 'it/technology',
            'programming': 'it/technology',
            'network': 'it/technology',
            'cyber': 'it/technology',
            'information': 'it/technology',
            'it': 'it/technology',
            'data': 'it/technology',
            'web': 'it/technology',
            'developer': 'it/technology',
            'database': 'it/technology',
            'security': 'it/technology',  # IT security
            
            # Technical/Trades
            'automotive': 'technical',
            'welding': 'technical',
            'electrical': 'technical',
            'plumbing': 'technical',
            'hvac': 'technical',
            'construction': 'technical',
            'carpentry': 'technical',
            'machinist': 'technical',
            'mechanic': 'technical',
            'diesel': 'technical',
            'collision': 'technical',
            'repair': 'technical',
            'maintenance': 'technical',
            'technician': 'technical',
            'engineering': 'technical',
            
            # Business
            'business': 'business',
            'accounting': 'business',
            'finance': 'business',
            'management': 'business',
            'marketing': 'business',
            'sales': 'business',
            'administration': 'business',
            'office': 'business',
            'administrative': 'business',
            'human resources': 'business',
            'hr': 'business',
        }
        
        # Keyword lists for job classification
        self.healthcare_keywords = [
            'nurse', 'medical', 'healthcare', 'health', 'patient', 'clinical',
            'hospital', 'clinic', 'doctor', 'physician', 'dental', 'pharmacy',
            'therapy', 'therapist', 'assistant', 'aide', 'caregiver'
        ]
        
        self.technical_keywords = [
            'automotive', 'mechanic', 'technician', 'welding', 'electrical',
            'plumbing', 'hvac', 'construction', 'carpentry', 'machinist',
            'diesel', 'repair', 'maintenance', 'installer', 'fitter'
        ]
        
        self.business_keywords = [
            'business', 'accounting', 'finance', 'management', 'manager',
            'administrative', 'office', 'clerk', 'secretary', 'coordinator',
            'human resources', 'hr', 'marketing', 'sales', 'accountant'
        ]
        
        self.it_keywords = [
            'software', 'developer', 'programmer', 'computer', 'it', 'network',
            'cyber', 'security', 'database', 'web', 'data', 'analyst',
            'engineer', 'system', 'programming', 'information technology'
        ]
    
    def label_program(self, program_name: str) -> str:
        """Label a program based on its name (manual mapping).
        
        Args:
            program_name: Name of the program
            
        Returns:
            Domain label
        """
        if not program_name:
            return 'other'
        
        program_name_lower = program_name.lower()
        
        # Check manual mappings
        for keyword, domain in self.program_mappings.items():
            if keyword in program_name_lower:
                return domain
        
        # Default to other
        return 'other'
    
    def label_job(self, title: str, description: str = '') -> str:
        """Label a job based on title and description (keyword-based).
        
        Args:
            title: Job title
            description: Job description (optional)
            
        Returns:
            Domain label
        """
        text = f"{title} {description}".lower()
        
        # Count keyword matches
        healthcare_score = sum(1 for keyword in self.healthcare_keywords if keyword in text)
        technical_score = sum(1 for keyword in self.technical_keywords if keyword in text)
        business_score = sum(1 for keyword in self.business_keywords if keyword in text)
        it_score = sum(1 for keyword in self.it_keywords if keyword in text)
        
        # Get domain with highest score
        scores = {
            'healthcare': healthcare_score,
            'technical': technical_score,
            'business': business_score,
            'it/technology': it_score
        }
        
        max_score = max(scores.values())
        
        if max_score == 0:
            return 'other'
        
        # Return domain with highest score
        for domain, score in scores.items():
            if score == max_score:
                return domain
        
        return 'other'

