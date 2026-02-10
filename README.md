# Scientific Information Retrieval 'Chatbot'

Hybrid TF-IDF/Word2Vec retrieval engine for scientific literature search. Evaluated on RELISH dataset.

## Data Requirements
```
data/relish/
├── relish_documents.tsv    # ~163k articles
└── relevance_matrix.tsv    # ~189k query-doc pairs (ground truth)
```

## Architecture

**Preprocessing**
- Corpus cleaning, column standardization
- Query-to-relevant-documents mapping construction

**Models**
- TF-IDF: lexical similarity
- Word2Vec: semantic similarity  
- Hybrid scorer: weighted fusion

**Evaluation**
- Metrics: P@10, MAP, NDCG@10, Recall@10
- Output: `results/*.csv`

## Execution Flow

1. Initialize directory structure (`data/`, `models/`, `results/`)
2. Load and preprocess corpus
3. Train retrieval models
4. Run queries (article abstracts)
5. Compute evaluation metrics on query subset

## Purpose

Demonstrate reproducible retrieval pipeline with established ground truth evaluation.