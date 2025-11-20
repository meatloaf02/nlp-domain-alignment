# Domain Alignment Pipeline

A minimal repository for reproducing the domain alignment analysis between vocational programs and job postings using NLP embeddings.

## Overview

This repository contains the essential code to:
1. Preprocess raw program and job data
2. Generate feature fusion embeddings (word + character n-grams with phrase modeling)
3. Analyze domain alignment between programs and jobs
4. Generate visualizations and reports

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Setup

1. **Clone or download this repository**
   ```bash
   cd nlp-domain-alignment
   ```

2. **Install dependencies**
   ```bash
   pip install -e .
   ```

3. **Download spaCy language model** (required for tokenization)
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Download NLTK data** (required for preprocessing)
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

## Data Requirements

### Input Data Format

#### Programs Data
- **Format**: JSONL file (`programs.jsonl`)
- **Location**: `data/processed/programs.jsonl`
- **Required fields**:
  - `program_name`: Name of the program
  - `description_raw` or `description_text`: Program description
  - Additional fields are optional

Example:
```json
{
  "program_name": "Medical Assistant",
  "description_text": "This program prepares students for careers as medical assistants...",
  "school": "Example College"
}
```

#### Jobs Data
- **Format**: Parquet file
- **Location**: `data/raw/jobs.parquet` (or similar)
- **Required fields**:
  - `title`: Job title
  - `description` or `description_text`: Job description
  - Additional fields are optional

## Workflow

The complete pipeline consists of three main steps:

### Step 1: Data Preprocessing

#### Preprocess Programs Data

```bash
python -m src.preprocess.pipeline \
    --input data/processed/programs.jsonl \
    --output-dir data/interim
```

This will:
- Clean text (remove HTML, URLs, etc.)
- Segment long documents
- Tokenize and lemmatize
- Filter stopwords
- Deduplicate
- Generate tokenized output: `data/interim/programs_tokenized.parquet`

#### Preprocess Jobs Data

```bash
python -m src.preprocess.preprocess_jobs \
    --input data/raw/jobs.parquet \
    --output data/interim/jobs_tokenized.parquet
```

This will generate: `data/interim/jobs_tokenized.parquet`

**Note**: If you already have tokenized data, you can skip this step and place your files directly in `data/interim/`.

### Step 2: Generate Embeddings

Run feature fusion experiments to generate embeddings:

```bash
python run_feature_fusion_experiments.py \
    --config config/ngram_experiment_config.yaml \
    --fusion-only
```

This will:
- Load tokenized programs and jobs data
- Generate feature fusion embeddings (word 1-2, char 3-5, phrases threshold=7)
- Save embeddings to: `artifacts/ngram_experiments/feature_fusion/additive_phrases_threshold_7/features.npy`

**Expected output**: `artifacts/ngram_experiments/feature_fusion/additive_phrases_threshold_7/features.npy`

### Step 3: Run Domain Alignment Analysis

```bash
python run_domain_alignment.py \
    --config config/domain_alignment_config.yaml
```

This will:
- Load tokenized programs and jobs
- Load embeddings from Step 2
- Compute similarity matrix between programs and jobs
- Assign domain labels (healthcare, technical, business, IT/technology, other)
- Generate domain-level statistics
- Create visualizations
- Generate summary report

**Output location**: `artifacts/unified_evaluation/domain_alignment/`

## Configuration

### Domain Alignment Config

Edit `config/domain_alignment_config.yaml` to customize:

- **Data paths**: Update paths to your preprocessed data
- **Embeddings path**: Path to embeddings from Step 2
- **Output directory**: Where to save results
- **Analysis parameters**: Top-K values for analysis
- **Cleaning parameters**: Minimum word count, etc.

### Feature Fusion Config

Edit `config/ngram_experiment_config.yaml` to customize:

- **Data paths**: Paths to tokenized programs and jobs
- **Feature fusion parameters**: Word/char n-gram ranges, phrase thresholds
- **Output directory**: Where to save embeddings

## Output Structure

After running the complete pipeline, you'll have:

```
artifacts/
├── ngram_experiments/
│   └── feature_fusion/
│       └── additive_phrases_threshold_7/
│           └── features.npy          # Embeddings file
└── unified_evaluation/
    └── domain_alignment/
        ├── alignment_matrix.parquet  # Similarity matrix
        ├── program_statistics.csv    # Per-program stats
        ├── domain_labels.csv         # Domain assignments
        ├── domain_alignment_summary.csv
        ├── top_bottom_alignment.csv
        ├── domain_alignment_summary.md  # Main report
        ├── qualitative_analysis.txt
        └── visualizations/
            ├── program_alignment_bars.png
            ├── domain_comparison.png
            └── umap_domain_projection.png
```

## Understanding the Results

### Domain Alignment Summary

The `domain_alignment_summary.csv` contains:
- **mean_in**: Average similarity within the same domain
- **mean_cross**: Average similarity across different domains
- **gap**: Difference (mean_in - mean_cross)

Higher gaps indicate better domain alignment.

### Program Statistics

The `program_statistics.csv` contains per-program metrics:
- **top5_mean**: Average similarity to top 5 matching jobs
- **top10_mean**: Average similarity to top 10 matching jobs

### Top/Bottom Alignment

The `top_bottom_alignment.csv` identifies:
- Programs with highest alignment to jobs
- Programs with lowest alignment to jobs
- Alignment concentration (fraction of top matches in same domain)

## Troubleshooting

### Common Issues

1. **Missing data files**
   - Ensure programs.jsonl and jobs.parquet exist
   - Check file paths in configuration files

2. **Import errors**
   - Verify all dependencies are installed: `pip install -e .`
   - Check that spaCy model is downloaded: `python -m spacy download en_core_web_sm`

3. **Memory errors**
   - Reduce dataset size for testing
   - Process data in batches if needed

4. **Embeddings not found**
   - Ensure Step 2 (feature fusion) completed successfully
   - Check that `features.npy` exists at the path specified in config

5. **Preprocessing errors**
   - Verify input data format matches expected schema
   - Check that required columns exist in input files

## File Structure

```
nlp-domain-alignment/
├── README.md                          # This file
├── .gitignore                         # Git ignore rules
├── pyproject.toml                     # Dependencies
├── run_domain_alignment.py            # Main domain alignment script
├── run_feature_fusion_experiments.py  # Embedding generation script
├── config/
│   ├── domain_alignment_config.yaml   # Domain alignment config
│   └── ngram_experiment_config.yaml   # Feature fusion config
├── src/
│   ├── models/
│   │   ├── domain_alignment.py        # Main pipeline
│   │   ├── domain_classifier.py       # Domain labeling
│   │   ├── feature_fusion_experiments.py
│   │   ├── experiment_runner.py
│   │   ├── ngram_experiments.py
│   │   ├── char_ngram_experiments.py
│   │   ├── phrase_experiments.py
│   │   └── ... (other model modules)
│   ├── preprocess/
│   │   ├── pipeline.py                 # Programs preprocessing
│   │   ├── preprocess_jobs.py         # Jobs preprocessing
│   │   ├── clean_text.py
│   │   ├── tokenize_text.py
│   │   ├── segment.py
│   │   ├── stopwords.py
│   │   ├── dedupe.py
│   │   ├── stats.py
│   │   ├── tfidf.py
│   │   └── clean_programs.py
│   └── utils.py                        # Utility functions
└── data/
    ├── raw/                            # Place raw data here
    ├── interim/                        # Preprocessed data
    └── processed/                      # Final processed data
```

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{domain_alignment_pipeline,
  title={Domain Alignment Pipeline for Vocational Programs and Job Postings},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/nlp-domain-alignment}
}
```

## License

This project is licensed under the MIT License.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review configuration files for correct paths
3. Verify all dependencies are installed correctly

