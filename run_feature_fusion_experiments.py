#!/usr/bin/env python3
"""
Feature Fusion Experiment Main Script.

This script runs feature fusion experiments combining word (1-2) and 
character (3-5) n-grams in TF-IDF with optional phrase modeling.

Experiments:
- Word n-grams (1-2) + Char n-grams (3-5)
- Fusion strategies: additive vs weighted concatenation
- Phrase modeling with thresholds: {5, 7, 10, 15}
- Evaluation: P@5, MRR, Silhouette

Usage:
    python run_feature_fusion_experiments.py [--config CONFIG_PATH] [--output OUTPUT_DIR]
"""

import argparse
import logging
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.experiment_runner import NgramExperimentRunner
from src.utils import setup_logging


def main():
    """Main entry point for feature fusion experiments."""
    parser = argparse.ArgumentParser(
        description="Run feature fusion experiments with word+char n-grams and phrase modeling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/ngram_experiment_config.yaml",
        help="Path to configuration file (default: config/ngram_experiment_config.yaml)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for results (overrides config)"
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
    
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline experiment (use existing baseline metrics)"
    )
    
    parser.add_argument(
        "--fusion-only",
        action="store_true",
        help="Run only feature fusion experiments (skip other experiment types)"
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
        logger.info("FEATURE FUSION EXPERIMENT PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Configuration: {config_path}")
        logger.info(f"Log level: {log_level}")
        logger.info("=" * 80)
        
        # Initialize experiment runner
        runner = NgramExperimentRunner(str(config_path))
        
        # Override output directory if specified
        if args.output:
            runner.output_dir = Path(args.output)
            runner.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory: {runner.output_dir}")
        
        # Load data
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Loading Data")
        logger.info("=" * 80)
        runner.load_data()
        
        # Create fixed splits
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Creating Fixed Train/Test Splits")
        logger.info("=" * 80)
        runner.create_fixed_splits()
        
        # Run baseline
        if not args.skip_baseline:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 3: Running Baseline Experiment")
            logger.info("=" * 80)
            runner.run_baseline()
        else:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 3: Skipping Baseline (using existing)")
            logger.info("=" * 80)
            # Try to load existing baseline
            baseline_path = runner.output_dir / "baseline" / "metrics.json"
            if baseline_path.exists():
                import json
                with open(baseline_path, 'r') as f:
                    runner.baseline_metrics = json.load(f)
                logger.info(f"Loaded baseline metrics from {baseline_path}")
            else:
                logger.warning("No existing baseline found, running baseline anyway")
                runner.run_baseline()
        
        if args.fusion_only:
            # Run only feature fusion experiments
            logger.info("\n" + "=" * 80)
            logger.info("STEP 4: Running Feature Fusion Experiments Only")
            logger.info("=" * 80)
            
            from src.models.feature_fusion_experiments import FeatureFusionExperiment
            
            fusion_dir = runner.output_dir / "feature_fusion"
            fusion_experiment = FeatureFusionExperiment(runner.config)
            fusion_summary = fusion_experiment.run_all_experiments(
                runner.all_texts, runner.tokenized_docs, runner.all_doc_types,
                fusion_dir,
                train_indices=runner.train_indices, test_indices=runner.test_indices
            )
            
            # Evaluate each feature fusion experiment
            all_results = []
            for _, row in fusion_summary.iterrows():
                features = np.load(row['features_path'])
                metrics = runner.evaluate_features(features, row['experiment_id'])
                
                # Calculate relative gains
                rel_gains = {}
                if runner.baseline_metrics:
                    baseline_retrieval = runner.baseline_metrics.get('retrieval', {})
                    baseline_f1 = runner.baseline_metrics.get('classification', {}).get('f1_weighted', 0)
                    baseline_silhouette = runner.baseline_metrics.get('clustering', {}).get('silhouette_score', 0)
                    
                    retrieval_p5 = metrics['retrieval']['precision_at_k']
                    baseline_p5 = baseline_retrieval.get('precision_at_k', 0)
                    if baseline_p5 > 0:
                        rel_gains['retrieval_p5'] = ((retrieval_p5 - baseline_p5) / baseline_p5) * 100
                    
                    retrieval_mrr = metrics['retrieval']['mean_reciprocal_rank']
                    baseline_mrr = baseline_retrieval.get('mean_reciprocal_rank', 0)
                    if baseline_mrr > 0:
                        rel_gains['retrieval_mrr'] = ((retrieval_mrr - baseline_mrr) / baseline_mrr) * 100
                    
                    f1 = metrics['classification']['f1_weighted']
                    if baseline_f1 > 0:
                        rel_gains['classification_f1'] = ((f1 - baseline_f1) / baseline_f1) * 100
                    
                    silhouette = metrics['clustering']['silhouette_score']
                    if baseline_silhouette > 0:
                        rel_gains['clustering_silhouette'] = ((silhouette - baseline_silhouette) / baseline_silhouette) * 100
                
                result = {
                    'experiment_type': 'feature_fusion',
                    'experiment_id': row['experiment_id'],
                    **row.to_dict(),
                    **metrics,
                    'relative_gains': rel_gains
                }
                all_results.append(result)
                
                # Save metrics
                exp_dir = Path(row['features_path']).parent
                metrics_path = exp_dir / "metrics.json"
                with open(metrics_path, 'w') as f:
                    json.dump({**metrics, 'relative_gains': rel_gains}, f, indent=2)
            
            results_df = pd.DataFrame(all_results)
            
        else:
            # Run all experiments (including feature fusion)
            logger.info("\n" + "=" * 80)
            logger.info("STEP 4: Running All Experiments")
            logger.info("=" * 80)
            results_df = runner.run_all_experiments()
        
        # Generate report
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: Generating Report")
        logger.info("=" * 80)
        report = runner.generate_report(results_df)
        
        # Save report
        report_path = runner.output_dir / "feature_fusion_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Print summary
        logger.info("=" * 80)
        logger.info("FEATURE FUSION EXPERIMENTS COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        logger.info(f"\nResults saved to: {runner.output_dir}")
        logger.info(f"Summary CSV: {runner.output_dir / 'experiment_summary.csv'}")
        logger.info(f"Report: {report_path}")
        
        # Print feature fusion results summary
        if len(results_df) > 0:
            fusion_results = results_df[results_df['experiment_type'] == 'feature_fusion']
            if len(fusion_results) > 0:
                logger.info("\nFeature Fusion Results Summary:")
                
                # Best P@5
                best_p5 = fusion_results.loc[
                    fusion_results['retrieval'].apply(lambda x: x.get('precision_at_k', 0)).idxmax()
                ]
                logger.info(f"  Best P@5: {best_p5['experiment_id']} "
                           f"({best_p5['retrieval']['precision_at_k']:.4f})")
                
                # Best MRR
                best_mrr = fusion_results.loc[
                    fusion_results['retrieval'].apply(lambda x: x.get('mean_reciprocal_rank', 0)).idxmax()
                ]
                logger.info(f"  Best MRR: {best_mrr['experiment_id']} "
                           f"({best_mrr['retrieval']['mean_reciprocal_rank']:.4f})")
                
                # Best Silhouette
                best_sil = fusion_results.loc[
                    fusion_results['clustering'].apply(lambda x: x.get('silhouette_score', -1)).idxmax()
                ]
                logger.info(f"  Best Silhouette: {best_sil['experiment_id']} "
                           f"({best_sil['clustering']['silhouette_score']:.4f})")
                
                # Print phrase statistics if available
                if 'merge_rate' in fusion_results.columns:
                    logger.info("\nPhrase Statistics:")
                    for _, row in fusion_results.iterrows():
                        if pd.notna(row.get('merge_rate')):
                            logger.info(f"  {row['experiment_id']}: "
                                       f"merge_rate={row['merge_rate']:.4f}, "
                                       f"vocab: {row.get('vocab_size_before', 'N/A')} -> "
                                       f"{row.get('vocab_size_after', 'N/A')}")
        
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.info("\nExperiment pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Experiment pipeline failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

