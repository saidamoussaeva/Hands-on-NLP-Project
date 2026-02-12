# Scientific Information Retrieval 'Chatbot'

Sparse vs Dense Embeddings for Scientific Article Recommendation  

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
- Transformer-Based Model (SPECTER2)

**Evaluation**
- Metrics: P@10, MAP, NDCG@10, Recall@10

## Task Definition

Input: PubMed article (PMID or abstract)   
Output: Top-k scientifically relevant articles   
Objective: Rank relevant documents as high as possible   

## Purpose

Demonstrate reproducible retrieval pipeline with established ground truth evaluation.
