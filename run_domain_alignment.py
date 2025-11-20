#!/usr/bin/env python3
"""
Domain Alignment Pipeline Main Script

This script runs the domain alignment analysis pipeline to measure alignment
between vocational programs and job postings.

Usage:
    python run_domain_alignment.py [--config CONFIG_PATH] [--verbose]
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.domain_alignment import DomainAlignmentPipeline
from src.utils import setup_logging


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Flatten nested config for easier access
    flat_config = {}
    
    # Data paths
    if 'data' in config:
        flat_config.update(config['data'])
    
    # Embeddings
    if 'embeddings' in config:
        flat_config['embeddings_path'] = config['embeddings'].get('embeddings_path')
    
    # Output
    if 'output' in config:
        flat_config['output_dir'] = config['output'].get('output_dir')
    
    # Cleaning parameters
    if 'cleaning' in config:
        flat_config['use_cleaned_data'] = config['cleaning'].get('use_cleaned_data', True)
        flat_config['min_word_count'] = config['cleaning'].get('min_word_count', 50)
    
    return flat_config


def main():
    """Main entry point for domain alignment pipeline."""
    parser = argparse.ArgumentParser(
        description="Run domain alignment analysis between programs and jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/domain_alignment_config.yaml",
        help="Path to configuration file (default: config/domain_alignment_config.yaml)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = "DEBUG" if args.verbose else args.log_level
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Validate configuration file
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            sys.exit(1)
        
        logger.info("=" * 80)
        logger.info("DOMAIN ALIGNMENT PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Configuration: {config_path}")
        logger.info(f"Log level: {log_level}")
        logger.info("=" * 80)
        
        # Load configuration
        config = load_config(str(config_path))
        
        # Initialize pipeline
        pipeline = DomainAlignmentPipeline(config)
        
        # Run pipeline
        pipeline.run()
        
        logger.info("=" * 80)
        logger.info("DOMAIN ALIGNMENT PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"\nResults saved to: {pipeline.output_dir}")
        
    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

