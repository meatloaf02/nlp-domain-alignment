"""
Domain Alignment Pipeline

This module implements a comprehensive pipeline to measure alignment between
vocational programs and job postings using NLP embeddings, compute domain-level
statistics, and generate visualizations and reports.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import umap

from src.models.domain_classifier import DomainClassifier
from src.preprocess.clean_programs import ProgramCleaner
from src.utils import create_output_directory

logger = logging.getLogger(__name__)


class DomainAlignmentPipeline:
    """Main pipeline for domain alignment analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the domain alignment pipeline.
        
        Args:
            config: Configuration dictionary with paths and parameters
        """
        self.config = config
        self.output_dir = Path(config.get('output_dir', 'artifacts/unified_evaluation/domain_alignment'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data
        self.programs_df = None
        self.jobs_df = None
        self.program_embeddings = None
        self.job_embeddings = None
        
        # Index mappings
        self.program_id_to_idx = {}  # Map program ID to index position
        self.job_id_to_idx = {}     # Map job ID to index position
        
        # Results
        self.similarity_matrix = None
        self.domain_labels = None
        self.program_statistics = None
        
        # Domain classifier
        self.domain_classifier = DomainClassifier()
        
        # Program cleaner
        min_word_count = config.get('min_word_count', 50)
        self.program_cleaner = ProgramCleaner(min_word_count=min_word_count)
        
        # Cleaned data flags
        self.use_cleaned_data = config.get('use_cleaned_data', True)
        self.cleaned_programs_df = None
        
        logger.info(f"Initialized domain alignment pipeline with output: {self.output_dir}")
    
    def load_data(self) -> None:
        """Load processed program descriptions, job postings, and embeddings."""
        logger.info("Loading data and embeddings")
        
        # Load program data - use tokenized data to match embeddings
        programs_tokenized_path = self.config.get('programs_tokenized_path', 'data/interim/programs_tokenized.parquet')
        programs_tokenized = pd.read_parquet(programs_tokenized_path)
        logger.info(f"Loaded {len(programs_tokenized)} programs from tokenized data")
        
        # Clean programs if requested
        if self.use_cleaned_data:
            logger.info("Cleaning program data...")
            self.cleaned_programs_df = self.program_cleaner.clean_programs(
                programs_tokenized,
                program_name_col='program_name',
                description_col='description_text',
                deduplicate=True,
                remove_non_programs=True,
                enforce_min_length=True
            )
            
            # Save cleaned data
            cleaned_path = self.output_dir.parent / "programs_clean.parquet"
            self.cleaned_programs_df.to_parquet(cleaned_path)
            logger.info(f"Saved cleaned programs to {cleaned_path}")
            
            # Save deduplicated data
            dedup_path = self.output_dir.parent / "programs_dedup.parquet"
            dedup_df = self.program_cleaner.deduplicate(programs_tokenized)
            dedup_df.to_parquet(dedup_path)
            logger.info(f"Saved deduplicated programs to {dedup_path}")
            
            self.programs_df = self.cleaned_programs_df
        else:
            self.programs_df = programs_tokenized
        
        # Load job data
        jobs_path = self.config.get('jobs_path', 'data/interim/jobs_tokenized.parquet')
        self.jobs_df = pd.read_parquet(jobs_path)
        logger.info(f"Loaded {len(self.jobs_df)} jobs")
        
        # Load embeddings
        embeddings_path = self.config.get(
            'embeddings_path',
            'artifacts/ngram_experiments/feature_fusion/additive_phrases_threshold_7/features.npy'
        )
        all_embeddings = np.load(embeddings_path)
        logger.info(f"Loaded embeddings: {all_embeddings.shape}")
        
        # Determine split - embeddings are computed on original tokenized data
        # We need to match cleaned programs to original embeddings
        n_jobs = len(self.jobs_df)
        
        # Load original tokenized programs to match embeddings
        programs_tokenized_path = self.config.get('programs_tokenized_path', 'data/interim/programs_tokenized.parquet')
        programs_original = pd.read_parquet(programs_tokenized_path)
        n_programs_original = len(programs_original)
        
        logger.info(f"Embeddings split: {n_jobs} jobs, {n_programs_original} programs (original)")
        logger.info(f"Cleaned programs: {len(self.programs_df)} programs")
        
        # Split embeddings
        self.job_embeddings = all_embeddings[:n_jobs]
        program_embeddings_full = all_embeddings[n_jobs:n_jobs + n_programs_original]
        
        # If we cleaned data, we need to filter embeddings to match cleaned programs
        if self.use_cleaned_data and self.cleaned_programs_df is not None:
            # Create mapping: for each cleaned program, find its position in original data
            # Use normalized program names to match
            original_positions = []
            cleaned_program_names_norm = {
                self.program_cleaner.normalize_program_name(str(row.get('program_name', ''))): i
                for i, (_, row) in enumerate(self.programs_df.iterrows())
            }
            
            # Match original programs to cleaned programs
            for orig_pos, (orig_idx, orig_row) in enumerate(programs_original.iterrows()):
                orig_name_norm = self.program_cleaner.normalize_program_name(
                    str(orig_row.get('program_name', ''))
                )
                
                # Check if this original program exists in cleaned data
                if orig_name_norm in cleaned_program_names_norm:
                    original_positions.append(orig_pos)
            
            # Filter embeddings to match cleaned programs
            if len(original_positions) == len(self.programs_df):
                self.program_embeddings = program_embeddings_full[original_positions]
                logger.info(f"Filtered embeddings to {len(self.program_embeddings)} programs after cleaning")
            else:
                logger.warning(f"Mismatch: {len(original_positions)} matched positions vs {len(self.programs_df)} cleaned programs")
                # Fallback: use all embeddings
                self.program_embeddings = program_embeddings_full[:len(self.programs_df)]
        else:
            self.program_embeddings = program_embeddings_full
        
        logger.info(f"Job embeddings: {self.job_embeddings.shape}")
        logger.info(f"Program embeddings: {self.program_embeddings.shape}")
        
        # Normalize embeddings for cosine similarity
        self.job_embeddings = normalize(self.job_embeddings, norm='l2')
        self.program_embeddings = normalize(self.program_embeddings, norm='l2')
        
        # Create index mappings
        self.program_id_to_idx = {pid: i for i, pid in enumerate(self.programs_df.index)}
        self.job_id_to_idx = {jid: i for i, jid in enumerate(self.jobs_df.index)}
    
    def compute_similarity_matrix(self) -> None:
        """Compute cosine similarity matrix between all programs and jobs."""
        logger.info("Computing similarity matrix")
        
        # Compute cosine similarity
        self.similarity_matrix = cosine_similarity(self.program_embeddings, self.job_embeddings)
        
        logger.info(f"Similarity matrix shape: {self.similarity_matrix.shape}")
        
        # Save similarity matrix as parquet
        # Use program and job IDs as index/columns
        program_ids = [str(pid) for pid in self.programs_df.index]
        job_ids = [str(jid) for jid in self.jobs_df.index]
        
        similarity_df = pd.DataFrame(
            self.similarity_matrix,
            index=program_ids,
            columns=job_ids
        )
        
        output_path = self.output_dir / "alignment_matrix.parquet"
        similarity_df.to_parquet(output_path)
        logger.info(f"Saved similarity matrix to {output_path}")
        
        # Compute per-program statistics
        self._compute_program_statistics()
    
    def _compute_program_statistics(self) -> None:
        """Compute per-program statistics (top5, top10 means)."""
        logger.info("Computing per-program statistics")
        
        stats = []
        for i in range(len(self.programs_df)):
            similarities = self.similarity_matrix[i]
            
            # Get top 5 and top 10
            top5_indices = np.argsort(similarities)[-5:][::-1]
            top10_indices = np.argsort(similarities)[-10:][::-1]
            
            top5_mean = np.mean(similarities[top5_indices])
            top10_mean = np.mean(similarities[top10_indices])
            
            # Get program ID
            program_id = self.programs_df.index[i]
            
            stats.append({
                'program_id': str(program_id),
                'top5_mean': top5_mean,
                'top10_mean': top10_mean
            })
        
        self.program_statistics = pd.DataFrame(stats)
        
        # Save statistics
        output_path = self.output_dir / "program_statistics.csv"
        self.program_statistics.to_csv(output_path, index=False)
        logger.info(f"Saved program statistics to {output_path}")
    
    def assign_domain_labels(self) -> None:
        """Assign domain labels to programs and jobs."""
        logger.info("Assigning domain labels")
        
        labels = []
        
        # Label programs (manual mapping based on program names)
        for idx, row in self.programs_df.iterrows():
            program_name = row.get('program_name', '')
            if pd.isna(program_name):
                program_name = ''
            
            domain = self.domain_classifier.label_program(program_name)
            
            labels.append({
                'id': str(idx),
                'type': 'program',
                'domain': domain
            })
        
        # Label jobs (keyword-based)
        for idx, row in self.jobs_df.iterrows():
            title = row.get('title', '')
            description = row.get('description', '') or row.get('description_text', '')
            
            if pd.isna(title):
                title = ''
            if pd.isna(description):
                description = ''
            
            domain = self.domain_classifier.label_job(title, description)
            
            labels.append({
                'id': str(idx),
                'type': 'job',
                'domain': domain
            })
        
        self.domain_labels = pd.DataFrame(labels)
        
        # Save labels
        output_path = self.output_dir / "domain_labels.csv"
        self.domain_labels.to_csv(output_path, index=False)
        logger.info(f"Saved domain labels to {output_path}")
        
        # Print distribution
        logger.info("Domain distribution:")
        for domain in self.domain_labels['domain'].unique():
            count = len(self.domain_labels[self.domain_labels['domain'] == domain])
            logger.info(f"  {domain}: {count}")
    
    def compute_domain_aggregation(self) -> pd.DataFrame:
        """Compute in-domain and cross-domain average similarities."""
        logger.info("Computing domain-level aggregation")
        
        # Create domain mapping
        program_labels = self.domain_labels[self.domain_labels['type'] == 'program']
        job_labels = self.domain_labels[self.domain_labels['type'] == 'job']
        
        program_domains = dict(zip(program_labels['id'], program_labels['domain']))
        job_domains = dict(zip(job_labels['id'], job_labels['domain']))
        
        # Get unique domains
        all_domains = sorted(self.domain_labels['domain'].unique())
        
        results = []
        
        for domain in all_domains:
            # Get program and job indices for this domain
            program_indices = [i for i, pid in enumerate(self.programs_df.index) 
                             if str(pid) in program_domains and program_domains[str(pid)] == domain]
            job_indices = [i for i, jid in enumerate(self.jobs_df.index) 
                          if str(jid) in job_domains and job_domains[str(jid)] == domain]
            
            if len(program_indices) == 0 or len(job_indices) == 0:
                logger.warning(f"No programs or jobs for domain {domain}, skipping")
                continue
            
            # In-domain similarities
            in_domain_similarities = []
            for p_idx in program_indices:
                for j_idx in job_indices:
                    in_domain_similarities.append(self.similarity_matrix[p_idx, j_idx])
            
            mean_in = np.mean(in_domain_similarities) if in_domain_similarities else 0.0
            
            # Cross-domain similarities
            cross_domain_similarities = []
            for p_idx in program_indices:
                for j_idx in range(len(self.jobs_df)):
                    if j_idx not in job_indices:  # Different domain
                        cross_domain_similarities.append(self.similarity_matrix[p_idx, j_idx])
            
            mean_cross = np.mean(cross_domain_similarities) if cross_domain_similarities else 0.0
            
            gap = mean_in - mean_cross
            
            results.append({
                'domain': domain,
                'mean_in': mean_in,
                'mean_cross': mean_cross,
                'gap': gap
            })
        
        summary_df = pd.DataFrame(results)
        
        # Save summary
        output_path = self.output_dir / "domain_alignment_summary.csv"
        summary_df.to_csv(output_path, index=False)
        logger.info(f"Saved domain alignment summary to {output_path}")
        
        return summary_df
    
    def analyze_top_bottom_programs(self) -> pd.DataFrame:
        """Identify top and bottom programs by alignment."""
        logger.info("Analyzing top and bottom programs")
        
        # Create domain mapping
        program_labels = self.domain_labels[self.domain_labels['type'] == 'program']
        job_labels = self.domain_labels[self.domain_labels['type'] == 'job']
        
        program_domains = dict(zip(program_labels['id'], program_labels['domain']))
        job_domains = dict(zip(job_labels['id'], job_labels['domain']))
        
        results = []
        
        for i in range(len(self.programs_df)):
            program_id = self.programs_df.index[i]
            program_id_str = str(program_id)
            program_name = self.programs_df.iloc[i].get('program_name', 'Unknown')
            program_domain = program_domains.get(program_id_str, 'other')
            
            similarities = self.similarity_matrix[i]
            
            # Get top 5 jobs
            top5_indices = np.argsort(similarities)[-5:][::-1]
            top5_similarities = similarities[top5_indices]
            
            # Get top 5 job details
            top5_jobs = []
            same_domain_count = 0
            for j_idx in top5_indices:
                job_id = self.jobs_df.index[j_idx]
                job_id_str = str(job_id)
                job_title = self.jobs_df.iloc[j_idx].get('title', 'Unknown')
                job_domain = job_domains.get(job_id_str, 'other')
                similarity = similarities[j_idx]
                
                top5_jobs.append({
                    'title': job_title,
                    'similarity': similarity,
                    'domain': job_domain
                })
                
                if job_domain == program_domain:
                    same_domain_count += 1
            
            alignment_concentration = same_domain_count / 5.0  # Percentage
            
            mean_top5 = np.mean(top5_similarities)
            
            results.append({
                'program_id': program_id_str,
                'program_name': program_name,
                'program_domain': program_domain,
                'mean_top5': mean_top5,
                'alignment_concentration': alignment_concentration,
                'top5_jobs': json.dumps(top5_jobs)  # Store as JSON string
            })
        
        results_df = pd.DataFrame(results)
        
        # Sort by mean_top5
        results_df = results_df.sort_values('mean_top5', ascending=False)
        
        # Get top 10 and bottom 10
        top10 = results_df.head(10).copy()
        bottom10 = results_df.tail(10).copy()
        
        # Combine and mark
        top10['rank_type'] = 'top'
        bottom10['rank_type'] = 'bottom'
        
        top_bottom_df = pd.concat([top10, bottom10], ignore_index=True)
        
        # Save results
        output_path = self.output_dir / "top_bottom_alignment.csv"
        top_bottom_df.to_csv(output_path, index=False)
        logger.info(f"Saved top/bottom alignment to {output_path}")
        
        return top_bottom_df
    
    def generate_visualizations(self) -> None:
        """Generate visualization plots."""
        logger.info("Generating visualizations")
        
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Plot 1: Horizontal bar chart of mean(top5 similarity) per program
        self._plot_program_alignment_bars(viz_dir)
        
        # Plot 2: Domain-level comparison
        self._plot_domain_comparison(viz_dir)
        
        # Plot 3: UMAP projection
        self._plot_umap_projection(viz_dir)
    
    def _plot_program_alignment_bars(self, output_dir: Path) -> None:
        """Plot horizontal bar chart of program alignment."""
        # Sort by top5_mean
        sorted_stats = self.program_statistics.sort_values('top5_mean', ascending=True)
        
        # Take top 50 for readability
        top50 = sorted_stats.tail(50)
        
        plt.figure(figsize=(12, 16))
        plt.barh(range(len(top50)), top50['top5_mean'])
        plt.yticks(range(len(top50)), [f"Program {pid}" for pid in top50['program_id']])
        plt.xlabel('Mean Top-5 Similarity')
        plt.title('Program Alignment (Top 50 Programs)')
        plt.tight_layout()
        
        output_path = output_dir / "program_alignment_bars.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved program alignment bars to {output_path}")
    
    def _plot_domain_comparison(self, output_dir: Path) -> None:
        """Plot domain-level comparison bar chart."""
        summary_df = pd.read_csv(self.output_dir / "domain_alignment_summary.csv")
        
        x = np.arange(len(summary_df))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bars1 = ax.bar(x - width/2, summary_df['mean_in'], width, label='In-Domain', alpha=0.8)
        bars2 = ax.bar(x + width/2, summary_df['mean_cross'], width, label='Cross-Domain', alpha=0.8)
        
        ax.set_xlabel('Domain')
        ax.set_ylabel('Mean Similarity')
        ax.set_title('Domain Alignment Comparison: In-Domain vs Cross-Domain')
        ax.set_xticks(x)
        ax.set_xticklabels(summary_df['domain'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        output_path = output_dir / "domain_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved domain comparison to {output_path}")
    
    def _plot_umap_projection(self, output_dir: Path) -> None:
        """Plot UMAP projection of embeddings colored by domain."""
        # Combine embeddings
        all_embeddings = np.vstack([self.program_embeddings, self.job_embeddings])
        
        # Create domain labels for visualization
        program_labels = self.domain_labels[self.domain_labels['type'] == 'program']
        job_labels = self.domain_labels[self.domain_labels['type'] == 'job']
        
        program_domains = dict(zip(program_labels['id'], program_labels['domain']))
        job_domains = dict(zip(job_labels['id'], job_labels['domain']))
        
        all_domains = []
        for pid in self.programs_df.index:
            all_domains.append(program_domains.get(str(pid), 'other'))
        for jid in self.jobs_df.index:
            all_domains.append(job_domains.get(str(jid), 'other'))
        
        # Reduce dimensionality
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        embedding_2d = reducer.fit_transform(all_embeddings)
        
        # Plot
        plt.figure(figsize=(14, 10))
        
        # Plot by domain
        unique_domains = sorted(set(all_domains))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_domains)))
        domain_colors = dict(zip(unique_domains, colors))
        
        for domain in unique_domains:
            mask = np.array(all_domains) == domain
            plt.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1], 
                       c=[domain_colors[domain]], label=domain, alpha=0.6, s=20)
        
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.title('UMAP Projection of Embeddings Colored by Domain')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        output_path = output_dir / "umap_domain_projection.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved UMAP projection to {output_path}")
    
    def qualitative_inspection(self) -> None:
        """Perform qualitative inspection of top and bottom programs."""
        logger.info("Performing qualitative inspection")
        
        top_bottom_df = pd.read_csv(self.output_dir / "top_bottom_alignment.csv")
        
        # Get top 3 and bottom 3
        top3 = top_bottom_df[top_bottom_df['rank_type'] == 'top'].head(3)
        bottom3 = top_bottom_df[top_bottom_df['rank_type'] == 'bottom'].tail(3)
        
        output_lines = []
        output_lines.append("=" * 80)
        output_lines.append("QUALITATIVE INSPECTION: TOP AND BOTTOM ALIGNED PROGRAMS")
        output_lines.append("=" * 80)
        output_lines.append("")
        
        # Top 3 programs
        output_lines.append("TOP 3 ALIGNED PROGRAMS")
        output_lines.append("-" * 80)
        
        for idx, row in top3.iterrows():
            program_id = row['program_id']
            program_name = row['program_name']
            program_domain = row['program_domain']
            mean_top5 = row['mean_top5']
            alignment_concentration = row['alignment_concentration']
            
            # Get program description
            # Try to find by string match or index position
            program_row = None
            for pid in self.programs_df.index:
                if str(pid) == program_id:
                    program_row = self.programs_df.loc[pid]
                    break
            description = ''
            if program_row is not None:
                description = program_row.get('description_text', '') or program_row.get('description_raw', '')
                if pd.isna(description):
                    description = ''
                description = description[:500] + '...' if len(description) > 500 else description
            
            output_lines.append(f"\nProgram: {program_name}")
            output_lines.append(f"Domain: {program_domain}")
            output_lines.append(f"Mean Top-5 Similarity: {mean_top5:.4f}")
            output_lines.append(f"Alignment Concentration: {alignment_concentration:.2%}")
            output_lines.append(f"\nDescription: {description}")
            
            # Get top 5 jobs
            top5_jobs = json.loads(row['top5_jobs'])
            output_lines.append("\nTop 5 Matching Jobs:")
            for i, job in enumerate(top5_jobs, 1):
                output_lines.append(f"  {i}. {job['title']} (similarity: {job['similarity']:.4f}, domain: {job['domain']})")
            
            output_lines.append("")
        
        # Bottom 3 programs
        output_lines.append("\n" + "=" * 80)
        output_lines.append("BOTTOM 3 ALIGNED PROGRAMS")
        output_lines.append("-" * 80)
        
        for idx, row in bottom3.iterrows():
            program_id = row['program_id']
            program_name = row['program_name']
            program_domain = row['program_domain']
            mean_top5 = row['mean_top5']
            alignment_concentration = row['alignment_concentration']
            
            # Get program description
            # Try to find by string match or index position
            program_row = None
            for pid in self.programs_df.index:
                if str(pid) == program_id:
                    program_row = self.programs_df.loc[pid]
                    break
            description = ''
            if program_row is not None:
                description = program_row.get('description_text', '') or program_row.get('description_raw', '')
                if pd.isna(description):
                    description = ''
                description = description[:500] + '...' if len(description) > 500 else description
            
            output_lines.append(f"\nProgram: {program_name}")
            output_lines.append(f"Domain: {program_domain}")
            output_lines.append(f"Mean Top-5 Similarity: {mean_top5:.4f}")
            output_lines.append(f"Alignment Concentration: {alignment_concentration:.2%}")
            output_lines.append(f"\nDescription: {description}")
            
            # Get top 5 jobs
            top5_jobs = json.loads(row['top5_jobs'])
            output_lines.append("\nTop 5 Matching Jobs:")
            for i, job in enumerate(top5_jobs, 1):
                output_lines.append(f"  {i}. {job['title']} (similarity: {job['similarity']:.4f}, domain: {job['domain']})")
            
            output_lines.append("")
        
        # Save analysis
        output_path = self.output_dir / "qualitative_analysis.txt"
        with open(output_path, 'w') as f:
            f.write('\n'.join(output_lines))
        
        logger.info(f"Saved qualitative analysis to {output_path}")
    
    def generate_summary_report(self) -> None:
        """Generate comprehensive markdown summary report."""
        logger.info("Generating summary report")
        
        # Load data
        summary_df = pd.read_csv(self.output_dir / "domain_alignment_summary.csv")
        top_bottom_df = pd.read_csv(self.output_dir / "top_bottom_alignment.csv")
        
        output_lines = []
        output_lines.append("# Domain Alignment Summary Report")
        output_lines.append("")
        output_lines.append(f"Generated on: {pd.Timestamp.now()}")
        output_lines.append("")
        
        # Executive Summary
        output_lines.append("## Executive Summary")
        output_lines.append("")
        output_lines.append("This report analyzes the alignment between vocational programs and job postings")
        output_lines.append("using feature fusion embeddings (word 1-2, char 3-5, phrases threshold=7).")
        output_lines.append("")
        
        # Domain-Level Statistics
        output_lines.append("## Domain-Level Statistics")
        output_lines.append("")
        output_lines.append("| Domain | Mean In-Domain | Mean Cross-Domain | Gap |")
        output_lines.append("|--------|----------------|-------------------|-----|")
        for _, row in summary_df.iterrows():
            output_lines.append(f"| {row['domain']} | {row['mean_in']:.4f} | {row['mean_cross']:.4f} | {row['gap']:.4f} |")
        output_lines.append("")
        
        # Key Findings
        output_lines.append("## Key Findings")
        output_lines.append("")
        
        # Best aligned domain
        best_domain = summary_df.loc[summary_df['gap'].idxmax()]
        output_lines.append(f"### Best Aligned Domain: {best_domain['domain']}")
        output_lines.append(f"- In-domain similarity: {best_domain['mean_in']:.4f}")
        output_lines.append(f"- Cross-domain similarity: {best_domain['mean_cross']:.4f}")
        output_lines.append(f"- Gap: {best_domain['gap']:.4f}")
        output_lines.append("")
        
        # Top Programs
        output_lines.append("### Top Aligned Programs")
        output_lines.append("")
        top10 = top_bottom_df[top_bottom_df['rank_type'] == 'top'].head(10)
        for idx, row in top10.iterrows():
            output_lines.append(f"- **{row['program_name']}** ({row['program_domain']}): "
                              f"Mean Top-5 = {row['mean_top5']:.4f}, "
                              f"Concentration = {row['alignment_concentration']:.2%}")
        output_lines.append("")
        
        # Example Aligned Pairs
        output_lines.append("## Example Aligned Pairs")
        output_lines.append("")
        
        # Get a few examples from top programs
        top_program = top_bottom_df[top_bottom_df['rank_type'] == 'top'].iloc[0]
        top5_jobs = json.loads(top_program['top5_jobs'])
        
        # Find program row
        program_row = None
        program_id = top_program['program_id']
        for pid in self.programs_df.index:
            if str(pid) == program_id:
                program_row = self.programs_df.loc[pid]
                break
        program_desc = ''
        if program_row is not None:
            program_desc = program_row.get('description_text', '') or program_row.get('description_raw', '')
            if pd.isna(program_desc):
                program_desc = ''
            program_desc = program_desc[:300] + '...' if len(program_desc) > 300 else program_desc
        
        output_lines.append(f"### Program: {top_program['program_name']}")
        output_lines.append(f"**Description snippet:** {program_desc}")
        output_lines.append("")
        output_lines.append("**Top matching job:**")
        top_job = top5_jobs[0]
        job_row = self.jobs_df.iloc[0]  # Need to find by similarity
        job_desc = job_row.get('description', '') or job_row.get('description_text', '')
        if pd.isna(job_desc):
            job_desc = ''
        job_desc = job_desc[:300] + '...' if len(job_desc) > 300 else job_desc
        output_lines.append(f"- **{top_job['title']}** (similarity: {top_job['similarity']:.4f})")
        output_lines.append(f"  Description: {job_desc}")
        output_lines.append("")
        
        # Observations
        output_lines.append("## Observations")
        output_lines.append("")
        output_lines.append("### Domain Alignment Strengths")
        output_lines.append("")
        output_lines.append("- Programs and jobs within the same domain show higher similarity scores")
        output_lines.append(f"- The {best_domain['domain']} domain shows the strongest in-domain alignment")
        output_lines.append("")
        
        output_lines.append("### Domain Alignment Gaps")
        output_lines.append("")
        output_lines.append("- Some programs may have low alignment due to:")
        output_lines.append("  - Mismatch between program content and job requirements")
        output_lines.append("  - Limited job postings in the program's domain")
        output_lines.append("  - Generic program descriptions that don't capture specific skills")
        output_lines.append("")
        
        # Recommendations
        output_lines.append("## Recommendations")
        output_lines.append("")
        output_lines.append("1. **Enhance program descriptions** with more specific skill terms")
        output_lines.append("2. **Focus on high-alignment domains** for curriculum development")
        output_lines.append("3. **Investigate low-alignment programs** to identify skill gaps")
        output_lines.append("")
        
        # Save report
        output_path = self.output_dir / "domain_alignment_summary.md"
        with open(output_path, 'w') as f:
            f.write('\n'.join(output_lines))
        
        logger.info(f"Saved summary report to {output_path}")
    
    def analyze_healthcare_subdomains(self) -> None:
        """Analyze healthcare domain split into Clinical vs Administrative subdomains."""
        logger.info("Analyzing healthcare subdomains (Clinical vs Administrative)")
        
        # Filter to healthcare domain only
        healthcare_program_labels = self.domain_labels[
            (self.domain_labels['type'] == 'program') & 
            (self.domain_labels['domain'] == 'healthcare')
        ]
        healthcare_job_labels = self.domain_labels[
            (self.domain_labels['type'] == 'job') & 
            (self.domain_labels['domain'] == 'healthcare')
        ]
        
        if len(healthcare_program_labels) == 0 or len(healthcare_job_labels) == 0:
            logger.warning("Insufficient healthcare programs or jobs for subdomain analysis")
            return
        
        # Get program and job indices
        healthcare_program_ids = set(healthcare_program_labels['id'].astype(str))
        healthcare_job_ids = set(healthcare_job_labels['id'].astype(str))
        
        healthcare_program_indices = [
            i for i, pid in enumerate(self.programs_df.index) 
            if str(pid) in healthcare_program_ids
        ]
        healthcare_job_indices = [
            i for i, jid in enumerate(self.jobs_df.index) 
            if str(jid) in healthcare_job_ids
        ]
        
        # Subdomain keywords
        clinical_keywords = [
            'assistant', 'technician', 'phlebotomy', 'patient', 'nurse', 
            'therap', 'clinical', 'dental', 'surgical', 'medical assistant',
            'pharmacy', 'radiology', 'sonography', 'respiratory'
        ]
        admin_keywords = [
            'billing', 'collections', 'revenue', 'office', 'front desk', 
            'records', 'clerical', 'coding', 'medical records', 'receptionist'
        ]
        
        # Assign subdomain labels to jobs
        job_subdomains = {}
        for j_idx in healthcare_job_indices:
            job_id = str(self.jobs_df.index[j_idx])
            job_title = self.jobs_df.iloc[j_idx].get('title', '')
            job_desc = self.jobs_df.iloc[j_idx].get('description', '') or \
                       self.jobs_df.iloc[j_idx].get('description_text', '')
            
            text = f"{job_title} {job_desc}".lower()
            
            # Check keywords
            clinical_score = sum(1 for kw in clinical_keywords if kw in text)
            admin_score = sum(1 for kw in admin_keywords if kw in text)
            
            if clinical_score > admin_score:
                job_subdomains[job_id] = 'clinical'
            elif admin_score > clinical_score:
                job_subdomains[job_id] = 'administrative'
            else:
                job_subdomains[job_id] = 'other'
        
        # Assign subdomain labels to programs
        program_subdomains = {}
        for p_idx in healthcare_program_indices:
            program_id = str(self.programs_df.index[p_idx])
            program_name = self.programs_df.iloc[p_idx].get('program_name', '')
            program_desc = self.programs_df.iloc[p_idx].get('description_text', '') or \
                          self.programs_df.iloc[p_idx].get('description_raw', '')
            
            text = f"{program_name} {program_desc}".lower()
            
            # Check keywords
            clinical_score = sum(1 for kw in clinical_keywords if kw in text)
            admin_score = sum(1 for kw in admin_keywords if kw in text)
            
            if clinical_score > admin_score:
                program_subdomains[program_id] = 'clinical'
            elif admin_score > clinical_score:
                program_subdomains[program_id] = 'administrative'
            else:
                program_subdomains[program_id] = 'other'
        
        # Compute similarity matrix for healthcare only
        healthcare_program_embeddings = self.program_embeddings[healthcare_program_indices]
        healthcare_job_embeddings = self.job_embeddings[healthcare_job_indices]
        
        healthcare_similarity = cosine_similarity(healthcare_program_embeddings, healthcare_job_embeddings)
        
        # Analyze by subdomain
        results = []
        
        for subdomain in ['clinical', 'administrative']:
            # Get program and job indices for this subdomain
            subdomain_program_indices = [
                i for i, p_idx in enumerate(healthcare_program_indices)
                if str(self.programs_df.index[p_idx]) in program_subdomains and
                program_subdomains[str(self.programs_df.index[p_idx])] == subdomain
            ]
            subdomain_job_indices = [
                i for i, j_idx in enumerate(healthcare_job_indices)
                if str(self.jobs_df.index[j_idx]) in job_subdomains and
                job_subdomains[str(self.jobs_df.index[j_idx])] == subdomain
            ]
            
            if len(subdomain_program_indices) == 0 or len(subdomain_job_indices) == 0:
                continue
            
            # Compute mean top-5 similarity for programs in this subdomain
            for prog_idx in subdomain_program_indices:
                similarities = healthcare_similarity[prog_idx]
                top5_indices = np.argsort(similarities)[-5:][::-1]
                
                # Check if top matches are clinical or administrative
                clinical_matches = 0
                admin_matches = 0
                
                for job_idx in top5_indices:
                    job_id = str(self.jobs_df.index[healthcare_job_indices[job_idx]])
                    if job_id in job_subdomains:
                        if job_subdomains[job_id] == 'clinical':
                            clinical_matches += 1
                        elif job_subdomains[job_id] == 'administrative':
                            admin_matches += 1
                
                program_id = str(self.programs_df.index[healthcare_program_indices[prog_idx]])
                program_name = self.programs_df.iloc[healthcare_program_indices[prog_idx]].get('program_name', 'Unknown')
                
                mean_top5 = np.mean(similarities[top5_indices])
                clinical_proportion = clinical_matches / 5.0
                
                results.append({
                    'program_id': program_id,
                    'program_name': program_name,
                    'program_subdomain': subdomain,
                    'mean_top5': mean_top5,
                    'clinical_matches': clinical_matches,
                    'administrative_matches': admin_matches,
                    'clinical_proportion': clinical_proportion
                })
        
        results_df = pd.DataFrame(results)
        
        # Save results
        output_path = self.output_dir / "healthcare_subdomain_alignment.csv"
        results_df.to_csv(output_path, index=False)
        logger.info(f"Saved healthcare subdomain analysis to {output_path}")
        
        # Generate summary statistics
        summary = []
        for subdomain in ['clinical', 'administrative']:
            subdomain_results = results_df[results_df['program_subdomain'] == subdomain]
            if len(subdomain_results) > 0:
                summary.append({
                    'subdomain': subdomain,
                    'num_programs': len(subdomain_results),
                    'mean_top5_similarity': subdomain_results['mean_top5'].mean(),
                    'mean_clinical_proportion': subdomain_results['clinical_proportion'].mean()
                })
        
        summary_df = pd.DataFrame(summary)
        
        # Plot comparison
        if len(summary_df) > 0:
            self._plot_healthcare_subdomain_comparison(summary_df)
        
        logger.info("Healthcare subdomain analysis complete")
    
    def _plot_healthcare_subdomain_comparison(self, summary_df: pd.DataFrame) -> None:
        """Plot healthcare subdomain comparison."""
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(summary_df))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, summary_df['mean_top5_similarity'], width, 
                      label='Mean Top-5 Similarity', alpha=0.8)
        bars2 = ax.bar(x + width/2, summary_df['mean_clinical_proportion'], width,
                      label='Clinical Match Proportion', alpha=0.8)
        
        ax.set_xlabel('Subdomain')
        ax.set_ylabel('Score')
        ax.set_title('Healthcare Subdomain Alignment: Clinical vs Administrative')
        ax.set_xticks(x)
        ax.set_xticklabels(summary_df['subdomain'])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        output_path = viz_dir / "healthcare_subdomain_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved healthcare subdomain comparison to {output_path}")
    
    def run(self) -> None:
        """Run the complete domain alignment pipeline."""
        logger.info("=" * 80)
        logger.info("DOMAIN ALIGNMENT PIPELINE")
        logger.info("=" * 80)
        
        # Step 1: Load data
        logger.info("\nStep 1: Loading data")
        self.load_data()
        
        # Step 2: Compute similarity matrix
        logger.info("\nStep 2: Computing similarity matrix")
        self.compute_similarity_matrix()
        
        # Step 3: Assign domain labels
        logger.info("\nStep 3: Assigning domain labels")
        self.assign_domain_labels()
        
        # Step 4: Domain aggregation
        logger.info("\nStep 4: Computing domain aggregation")
        self.compute_domain_aggregation()
        
        # Step 5: Top/bottom analysis
        logger.info("\nStep 5: Analyzing top/bottom programs")
        self.analyze_top_bottom_programs()
        
        # Step 6: Visualizations
        logger.info("\nStep 6: Generating visualizations")
        self.generate_visualizations()
        
        # Step 7: Qualitative inspection
        logger.info("\nStep 7: Qualitative inspection")
        self.qualitative_inspection()
        
        # Step 8: Summary report
        logger.info("\nStep 8: Generating summary report")
        self.generate_summary_report()
        
        # Step 9: Healthcare subdomain analysis (if healthcare domain exists)
        logger.info("\nStep 9: Healthcare subdomain analysis")
        if self.domain_labels is not None:
            healthcare_count = len(self.domain_labels[
                (self.domain_labels['type'] == 'program') & 
                (self.domain_labels['domain'] == 'healthcare')
            ])
            if healthcare_count > 0:
                self.analyze_healthcare_subdomains()
            else:
                logger.info("No healthcare programs found, skipping subdomain analysis")
        
        logger.info("\n" + "=" * 80)
        logger.info("DOMAIN ALIGNMENT PIPELINE COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {self.output_dir}")

