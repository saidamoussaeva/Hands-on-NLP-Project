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

#### --> Use with Google Colab & Transformers Model
*Why a Transformers model?*
In order to go further and evaluate the results of our initial method (TF-IDF), we wanted to compare them with a Transformers model. We specifically chose to use **SPECTER2** and its **"proximity"** adapter. This model is specially trained to generate high-performance embeddings for scientific tasks such as article recommendation and proximity. 
*To learn more about this model: [https://huggingface.co/allenai/specter2](https://huggingface.co/allenai/specter2).*

**Launch the notebook on Google Colab**
Everything is prepared in the `OptimusPrime_for_GoogleColab.ipynb` notebook. Here are the steps to launch it correctly with the necessary hardware acceleration:

1. Go to your Google Drive.
2. Create a folder named `Project_NLP`.
3. Drag the notebook `OptimusPrime_for_GoogleColab.ipynb` into it and double-click on it to open it in Google Colab.
4. **Selecting the kernel:**
   - In the top right corner of the Colab interface, click on the small arrow next to *'Connect'* (or *'RAM/Disk'*).
   - Select **Change runtime type**.
   - Choose the **T4 GPU** hardware accelerator and confirm.

**Evaluation**
- Metrics: P@10, MAP, NDCG@10, Recall@10

## Task Definition

Input: PubMed article (PMID or abstract)   
Output: Top-k scientifically relevant articles   
Objective: Rank relevant documents as high as possible   

## Purpose

Demonstrate reproducible retrieval pipeline with established ground truth evaluation.
