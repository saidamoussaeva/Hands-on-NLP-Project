# Hands-on NLP Project – Scientific Information Retrieval (OHSUMED)

Ce projet implémente un **chatbot de recherche d’information scientifique** basé sur le corpus **OHSUMED**, restreint aux documents cardiovasculaires (MeSH C14).  
L’objectif est de comparer un **modèle de retrieval lexical simple** à un **modèle hybride plus avancé**.

---

## 1. Mise en place des données

### 1.1. Extraction du corpus brut

Depuis la racine du projet :

```bash
tar -xzf data/data_raw/ohsumed-all-docs.tar.gz -C data/data_raw
```
### 1.2 Préparation du dataset

Toujours depuis la racine du projet :

```bash
python src/download_ohsumed.py
```

```bash
python src/prepare_ohsumed.py
```

```bash
python src/preprocess_ohsumed.py
```

## Modèles  

Une comparaison de modèles sera réalisée dans ce projet.  

- TF-IDF + cosinus similarity (modèle simple)
- Modèle hybride TF-IDF + Word2Vec (alpha)
