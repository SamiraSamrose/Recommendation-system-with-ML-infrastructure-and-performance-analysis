## System Design Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            DATA INGESTION LAYER                         │
├─────────────────────────────────────────────────────────────────────────┤
│  Goodbooks-10k Dataset (GitHub)                                         │
│  ├── ratings.csv (user_id, book_id, rating)                             │
│  └── books.csv (book_id, metadata)                                      │
│                              ↓                                          │
│  Data Loading (pandas.read_csv)                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                       PREPROCESSING PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────┤
│  1. Deduplication (drop_duplicates)                                     │
│  2. Filtering (min_user_ratings=5, min_item_ratings=5)                  │
│  3. Label Encoding (user_id→user_idx, book_id→item_idx)                 │
│  4. Feature Engineering                                                 │
│     ├── User Aggregates: avg_rating, rating_std, rating_count           │
│     └── Item Aggregates: avg_rating, rating_std, rating_count           │
│  5. Train/Val/Test Split (80/12/8)                                      │
│  6. Sparse Matrix Construction (CSR format)                             │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                          MODEL TRAINING LAYER                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │              RETRIEVAL MODELS (Candidate Generation)        │        │
│  ├─────────────────────────────────────────────────────────────┤        │
│  │  • SVD (n_factors=100, lr=0.005, reg=0.02)                  │        │
│  │    Input: Sparse matrix → Output: User/Item latent factors  │        │
│  │                                                             │        │
│  │  • NMF (n_factors=50, non-negative constraint)              │        │
│  │    Input: Sparse matrix → Output: Non-negative factors      │        │
│  │                                                             │        │
│  │  • Item-Item CF (k=40, cosine similarity)                   │        │
│  │    Computes: Item similarity matrix (n_items × n_items)     │        │
│  │                                                             │        │
│  │  • User-User CF (k=40, mean-centered)                       │        │
│  │    Computes: User similarity matrix (n_users × n_users)     │        │
│  │                                                             │        │
│  │  • TruncatedSVD (n_components=50)                           │        │
│  │    Output: User embeddings (n_users,50)                     │        │
│  │            Item embeddings (n_items,50)                     │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                              ↓                                          │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │               RANKING MODELS (Score Prediction)             │        │
│  ├─────────────────────────────────────────────────────────────┤        │
│  │  • LightGBM (num_leaves=31, lr=0.05, n_est=100)             │        │
│  │    Input: [user_feats, item_feats] → Output: Rating score   │        │
│  │                                                             │        │
│  │  • XGBoost (max_depth=6, lr=0.05, n_est=100)                │        │
│  │    Input: [user_feats, item_feats] → Output: Rating score   │        │
│  │                                                             │        │
│  │  • LambdaRank LTR (query groups per user)                   │        │
│  │    Objective: Optimize NDCG directly                        │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                              ↓                                          │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │              PERSONALIZATION LAYER                          │        │
│  ├─────────────────────────────────────────────────────────────┤        │
│  │  • K-Means Clustering (n_clusters=5)                        │        │
│  │    Input: User embeddings → Output: User segments           │        │
│  │  • Cluster Preferences (cluster-item rating aggregates)     │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                      INFERENCE PIPELINE (API LAYER)                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  RecommendationSystem.get_recommendations(user_id, n=10, method)        │
│                              ↓                                          │
│  ┌───────────────────────────────────────────────────────────┐          │
│  │ STAGE 1: RETRIEVAL (Candidate Generation)                 │          │
│  │ ──────────────────────────────────────────────────────────│          │
│  │ 1. Get user_idx from encoder                              │          │
│  │ 2. Retrieve user_embedding[user_idx]                      │          │
│  │ 3. Compute cosine_similarity(user_emb, all_item_embs)     │          │
│  │ 4. Select top-100 candidates                              │          │
│  └───────────────────────────────────────────────────────────┘          │
│                              ↓                                          │
│  ┌───────────────────────────────────────────────────────────┐          │
│  │ STAGE 2: RANKING (Score Prediction)                       │          │
│  │ ──────────────────────────────────────────────────────────│          │
│  │ For each candidate:                                       │          │
│  │   IF method=='svd':                                       │          │
│  │     score = svd_model.predict(user_id, item_id).est       │          │
│  │   ELIF method=='lgb':                                     │          │
│  │     features = [user_feats, item_feats]                   │          │
│  │     score = lgb_model.predict(features)                   │          │
│  │   ELIF method=='hybrid':                                  │          │
│  │     score = 0.5*svd_score + 0.5*lgb_score                 │          │
│  └───────────────────────────────────────────────────────────┘          │
│                              ↓                                          │
│  ┌───────────────────────────────────────────────────────────┐          │
│  │ STAGE 3: POST-PROCESSING                                  │          │
│  │ ──────────────────────────────────────────────────────────│          │
│  │ 1. Sort candidates by score (descending)                  │          │
│  │ 2. Select top-N items                                     │          │
│  │ 3. Return: {recommendations, scores, latency_ms}          │          │
│  └───────────────────────────────────────────────────────────┘          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    BATCH PROCESSING & MONITORING                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  BatchProcessor                      SystemMonitor                      │
│  ├── process_batch()                 ├── log_error()                    │
│  │   - Iterate user_ids              │   - Error type tracking          │
│  │   - Call get_recommendations()    │   - Context logging              │
│  │   - Aggregate results             ├── log_performance()              │
│  │   - Compute throughput            │   - Metric collection            │
│  └── process_all_users()             │   - Timestamp tracking           │
│      - Split into batches            ├── get_error_summary()            │
│      - Sequential processing         ├── get_performance_summary()      │
│      - Statistics aggregation        └── health_check()                 │
│                                          - Error rate < 5%              │
│                                          - Uptime tracking              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         EVALUATION FRAMEWORK                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐       │
│  │ Accuracy Metrics │  │ Ranking Metrics  │  │ Search Quality   │       │
│  ├──────────────────┤  ├──────────────────┤  ├──────────────────┤       │
│  │ • RMSE           │  │ • Precision@K    │  │ • MRR            │       │
│  │ • MAE            │  │ • Recall@K       │  │ • MAP            │       │
│  │ • R²             │  │ • NDCG@K         │  │ • Hit Rate@K     │       │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘       │
│                                                                         │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐       │
│  │ Diversity        │  │ Statistical Tests│  │ Advanced Analysis│       │
│  ├──────────────────┤  ├──────────────────┤  ├──────────────────┤       │
│  │ • Coverage       │  │ • T-test         │  │ • Cold Start     │       │
│  │ • Gini Coeff     │  │ • Mann-Whitney   │  │ • Long Tail      │       │
│  │ • ILD            │  │ • Cohen's d      │  │ • Pop Bias       │       │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                      OPTIMIZATION & TESTING LAYER                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Hyperparameter Optimization         Testing Framework                  │
│  ├── Grid Search                     ├── Unit Tests                     │
│  │   - Parameter combinations        │   - Valid/invalid inputs         │
│  │   - Validation RMSE scoring       │   - Edge cases                   │
│  │   - Best param selection          ├── Benchmark Tests                │
│  ├── Learning-to-Rank                │   - Latency SLAs                 │
│  │   - LambdaRank objective          │   - Throughput targets           │
│  │   - NDCG optimization             │   - Accuracy thresholds          │
│  └── Query Expansion                 ├── Data Quality Tests             │
│      - Embedding similarity          │   - Null checks                  │
│      - Relevance feedback            │   - Range validation             │
│                                      └── Consistency Tests              │
│                                          - Determinism                  │
│                                          - Output validity              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                        OUTPUT & VISUALIZATION                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Generated Artifacts:                                                   │
│  ├── 11 Visualization Figures (.png)                                    │
│  │   - Model comparisons, ranking curves, feature importance            │
│  │   - Data distributions, system performance, clustering               │
│  │   - Statistical tests, trade-offs, search quality                    │
│  ├── Performance Report (.json)                                         │
│  │   - All metrics aggregated by category                               │
│  ├── Executive Summary (.txt)                                           │
│  │   - High-level findings and recommendations                          │
│  └── Test Report (.txt)                                                 │
│      - Test results, pass/fail counts, diagnostics                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA FLOW SUMMARY                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Raw Data → Preprocessing → Feature Engineering → Model Training        │
│      ↓                                                                  │
│  Retrieval (top-100) → Ranking (scoring) → Top-N Selection              │
│      ↓                                                                  │
│  API Response + Monitoring + Batch Processing                           │
│      ↓                                                                  │
│  Evaluation + Optimization + Testing                                    │
│      ↓                                                                  │
│  Reports + Visualizations                                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```