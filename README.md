# Zipf's Law & Hate Speech Analysis

A comprehensive linguistic analysis of hate speech using corpus linguistics, Zipf's law, and NLP embeddings on the [Kaggle Hate Speech Dataset](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset).

## Overview

This project investigates linguistic patterns in hate speech, offensive language, and neutral tweets through:

- **Statistical Analysis**: Vocabulary statistics, token frequency, POS tagging
- **Power-Law Distributions**: Zipf's law & Heap's law fitting with confidence bounds
- **Similarity Metrics**: Jaccard, sentiment (VADER), and categorical embeddings
- **Embeddings**: Empath, Doc2Vec, DistilBERT for category discrimination
- **Linguistic Quality**: Spelling error analysis and edit distance patterns

## Dataset

**Source**: Twitter dataset (24,802 tweets) annotated into 3 categories:

- `hate_speech` (1,430 tweets)
- `offensive_language` (19,190 tweets)
- `neither` (4,163 tweets)

## Key Analyses

### 1. Vocabulary & Statistical Analysis

- Top 30 frequent words per category
- Token/pronoun statistics
- WordCloud visualizations
- Category balance assessment

### 2. TF-IDF Analysis

- Top 30 TF-IDF tokens per category
- Comparison with frequency-based rankings
- Discrimination power evaluation

### 3. Zipf's Law (Word Frequency)

- Power-law distribution fitting
- R² and Adjusted R² evaluation
- 90% confidence bounds
- Log-log scale visualization

### 4. Zipf's Law (POS Tags)

- Part-of-speech frequency distributions
- Penn Treebank POS tagging
- Comparison with word-level Zipf's law

### 5. Heap's Law (Vocabulary Growth)

- V = K × N^β fitting
- Vocabulary expansion analysis
- Growth curve visualization

### 6. Jaccard Similarity

- Vocabulary overlap between categories
- Pairwise similarity matrix
- Discrimination assessment

### 7. VADER Sentiment Analysis

- Sentiment score distributions
- 10-bin histograms
- Euclidean distance comparison

### 8. Empath Categorization

- Psychological category embeddings (~194 dimensions)
- Cosine similarity analysis
- Top discriminative categories

### 9. Embedding Comparison

- **Doc2Vec** (Word2Vec 300d)
- **DistilBERT** (768d)
- **Empath** (194d lexicon)
- PCA visualization & discrimination ranking

### 10. Linguistic Quality (WordNet)

- Spelling error detection
- Error rate by category
- Top error patterns

### 11. Edit Distance Analysis

- Levenshtein distance calculation
- Correction cost distribution
- Error type categorization

## Installation

```bash
# Create virtual environment
python -m pip install virtualenv
python -m virtualenv .venv

# Activate environment
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Usage

Open and run `zipf_analysis.ipynb` in Jupyter Notebook in VS code

**Note**: First run will download pretrained models (~1.6GB for Word2Vec, ~250MB for DistilBERT).

## Project Structure

```
hate-speech-zipf-analysis/
├── zipf_analysis.ipynb          # Main analysis notebook
├── reddit_hate_speech.csv       # Dataset
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## License

This project is for educational and research purposes.
