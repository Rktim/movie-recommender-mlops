# 🎬 Movie Recommendation System — End-to-End MLOps Project

## Overview

This repository contains a **production-style Movie Recommendation System** built using modern **Machine Learning and MLOps practices**.

The project started as a content-based recommender using cosine similarity and evolved into a **complete ML lifecycle system** that supports:

* reproducible ML pipelines
* online inference via API
* runtime monitoring and metrics
* automated model retraining
* safe model versioning and promotion
* containerized deployment

The focus of this project is not just *building a model*, but **operating it reliably over time**.

---

## What This Project Demonstrates

* Feature engineering and similarity modeling
* Scikit-learn pipelines and model serialization
* Online inference with FastAPI
* Monitoring real usage metrics
* Automated retraining based on drift signals
* Model governance (champion / challenger strategy)
* Reproducible, containerized ML systems

This makes the project a **complete MLOps reference**, not just a recommendation demo.

---

## Core Recommendation System

### Data Processing

* Ingests a CSV dataset containing:

  ```
  id, title, genre, overview
  ```
* Combines textual features into a single semantic representation
* Produces a clean feature set optimized for similarity modeling

### Feature Engineering

* Text vectorization using **TF-IDF**
* Vocabulary learned from movie metadata
* Dimensionality reduction handled implicitly by TF-IDF weighting

### Similarity Modeling

* Uses **cosine similarity** to measure semantic closeness between movies
* Precomputes similarity vectors for fast inference
* Supports partial and noisy user input

### Example (Conceptual)

```python
recommend("Inception", top_k=5)
```

---

## Machine Learning Pipeline

* All preprocessing and modeling steps are encapsulated in a **scikit-learn Pipeline**
* The pipeline is serialized as a reusable artifact:

  ```
  pipeline.joblib
  ```
* Guarantees:

  * reproducibility
  * consistent inference
  * environment-independent deployment

---

## Online Inference API

* Built using **FastAPI**
* Exposes a REST endpoint:

  ```
  GET /recommend?movie=<name>&k=<n>
  ```
* Handles:

  * input normalization
  * cold-start queries
  * ranking and similarity scoring
* Designed for low-latency inference

---

## Runtime Monitoring & Metrics

During live inference, the system continuously records:

* request volume
* cold-start frequency
* similarity quality
* time since last retraining

Metrics are persisted to disk and used as **signals for retraining decisions**.

---

## Automated Retraining Workflow

* The system periodically evaluates runtime metrics
* Retraining is triggered only when:

  * data quality degrades
  * model performance weakens
  * the model becomes outdated
* New models are trained using a reusable ML training interface
* Training outputs include:

  * serialized models
  * evaluation reports

This ensures the system stays **up-to-date without manual intervention**.

---

## Model Governance & Safety

To avoid regressions:

* Newly trained models are treated as **challengers**
* The currently deployed model is the **champion**
* Promotion rule:

  ```
  challenger_score > champion_score
  ```
* If no improvement is observed, the challenger is rejected

This guarantees:

* stability in production
* controlled model evolution
* zero-downtime updates

---

## Containerized Deployment

* Fully containerized using **Docker**
* Separate services for:

  * inference
  * retraining
* Shared storage for:

  * models
  * metrics
  * reports
* Containers are **stateless and reproducible**

---

## Project Structure

```
movie_r/
├── api/        # Inference API
├── src/        # ML pipeline & inference logic
├── agents/     # Retraining & promotion logic
├── models/     # Model artifacts
├── metrics/    # Runtime metrics
├── reports/    # Evaluation reports
├── docker/     # Docker configuration
├── runtime/    # Shared runtime storage
└── requirements.txt
```

---

## Why This Project Matters

Most ML projects stop at training a model.

This project covers:

* training
* deployment
* monitoring
* retraining
* governance

It reflects how machine learning systems are **actually built and maintained in real applications**.

---

## Potential Extensions

* Hybrid recommendation (content + collaborative)
* Advanced ranking metrics (NDCG, MAP)
* Online A/B testing
* Model explainability
* Cloud deployment

---

## Author

**Raktim Kalita**
AI & Machine Learning Engineer
📧 [raktmxx@gmail.com](mailto:raktmxx@gmail.com)

---

## Final Note

This repository is intentionally designed as a **learning-plus-production project**, demonstrating how a machine learning system evolves from a simple idea into a robust, maintainable service.
