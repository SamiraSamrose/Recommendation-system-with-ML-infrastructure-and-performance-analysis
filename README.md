## Project Name

**Multi-Stage Recommendation System with Production ML Infrastructure**

---

## Project Overview

Recommendation system implementing retrieval-ranking architecture using matrix factorization and gradient boosting models. System processes user-item interaction data through collaborative filtering algorithms, generates candidates via embedding-based similarity search, ranks results using feature-based models, and deploys through API infrastructure with monitoring and batch processing capabilities.

---

## Goals & Purposes

- Build retrieval models using SVD, NMF, and collaborative filtering to generate candidate items from sparse user-item matrices
- Implement ranking layer with LightGBM and XGBoost trained on user/item aggregate features to score and order candidates
- Deploy recommendation pipeline as callable API with latency tracking, error logging, and batch processing support
- Evaluate system performance using accuracy metrics (RMSE, MAE, R²), ranking metrics (NDCG, Precision@K, Recall@K), and search quality metrics (MRR, MAP)
- Optimize models through hyperparameter grid search, statistical testing, and learning-to-rank approaches
- Analyze recommendation properties including diversity, coverage, cold-start performance, and popularity bias

---

## Technical Tools and Stacks

**Languages**: Python 3.x

**ML Frameworks**: scikit-learn, Surprise, LightGBM, XGBoost

**Data Processing**: pandas, NumPy, SciPy (sparse matrices)

**Visualization**: Matplotlib, Seaborn, Plotly

**Algorithms Implemented**: 
- Matrix Factorization (SVD, NMF)
- K-Nearest Neighbors (user-based, item-based)
- Gradient Boosting Decision Trees (LightGBM, XGBoost)
- TruncatedSVD for dimensionality reduction
- K-Means clustering
- LambdaRank learning-to-rank

**Statistical Testing**: scipy.stats (t-test, Mann-Whitney U, confidence intervals)

**Data Structures**: CSR sparse matrices, Label encoders

**Dataset**: Goodbooks-10k (book ratings dataset from GitHub)
- Source: `github.com/zygmuntz/goodbooks-10k`
- Format: CSV files (ratings.csv, books.csv)
- Size: 980K+ user-item-rating triplets

**Infrastructure Components**: Custom classes for deployment (RecommendationSystem), monitoring (SystemMonitor), batch processing (BatchProcessor), testing (RecommendationSystemTester), search quality (SearchQualityEvaluator, SearchQualityOptimizer)

---

## Features & Functionality

### Retrieval Models
- **SVD Matrix Factorization**: Decomposes user-item matrix into user/item latent factors with bias terms, trained via SGD with L2 regularization
- **NMF**: Non-negative matrix factorization with non-negativity constraints on latent factors
- **Item-Item CF**: Computes item similarity matrix using cosine distance, generates predictions via weighted average of similar items
- **User-User CF**: Computes user similarity with mean-centered ratings, generates predictions via weighted neighbor ratings
- **Embedding Generation**: TruncatedSVD creates 50-dimensional embeddings for users and items used in similarity search

### Ranking Models
- **LightGBM Ranker**: GBDT model trained on engineered features (user_avg_rating, user_rating_std, user_rating_count, item_avg_rating, item_rating_std, item_rating_count) with early stopping
- **XGBoost Ranker**: Alternative GBDT implementation with tree depth constraints and column/row sampling
- **Learning-to-Rank**: LambdaRank objective optimizes NDCG directly using query groups per user

### Feature Engineering
- User aggregates: mean rating, standard deviation, count, unique items interacted
- Item aggregates: mean rating, standard deviation, count, unique users reached
- Rating normalization: z-score transformation

### Personalization
- K-Means clustering segments users into 5 groups based on embedding similarity
- Cluster-level preference computation aggregates ratings per cluster-item pair
- Enables segment-based recommendation strategies

### Deployment Infrastructure
- **RecommendationSystem Class**: Accepts user_id, returns top-N recommendations via hybrid retrieval-ranking pipeline
- **Method Options**: SVD-only, LightGBM-only, hybrid (weighted combination)
- **Performance Tracking**: Records per-request latency, computes percentiles (P50, P95, P99), counts requests
- **Batch Processing**: BatchProcessor handles multiple users with throughput calculation
- **Monitoring**: SystemMonitor logs errors by type, tracks performance metrics, provides health check status

### Evaluation Framework
- **Accuracy Metrics**: RMSE, MAE, R² computed on test set predictions
- **Ranking Metrics**: Precision@K, Recall@K, NDCG@K for K=[5,10,20,50] with relevance threshold at rating≥4
- **Search Quality**: MRR (mean reciprocal rank), MAP (mean average precision), Hit Rate@K
- **Diversity Analysis**: Catalog coverage, Gini coefficient, intra-list diversity via pairwise embedding dissimilarity
- **Statistical Testing**: T-test, Mann-Whitney U test, Cohen's d effect size, 95% confidence intervals

### Optimization
- **Hyperparameter Tuning**: Grid search over n_factors, learning rates, regularization, tree parameters with validation RMSE selection
- **Query Expansion**: Embedding-based similarity expands candidate set beyond initial retrieval
- **Relevance Feedback**: Rocchio algorithm adjusts scores based on positive/negative feedback
- **Freshness Weighting**: Time decay function boosts recent items

### Analysis Components
- **Cold Start Analysis**: Compares RMSE for users with ≤3 ratings vs >3 ratings
- **Long Tail Coverage**: Measures recommendation distribution across head (top 20%), mid (20-50%), tail (bottom 50%) items
- **Popularity Bias**: Calculates fraction of recommendations from top 10% popular items
- **Temporal Consistency**: Evaluates RMSE across multiple time-split test periods

### Testing Infrastructure
- **Unit Tests**: Validates recommendation generation for valid/invalid users, edge cases (n=0,1,100)
- **Benchmark Tests**: Checks P95 latency <100ms, P99 <200ms, throughput >10 users/sec, RMSE <1.0, NDCG@10 >0.1
- **Data Quality Tests**: Verifies no null values, valid rating range, no duplicates, sufficient data size
- **Consistency Tests**: Validates deterministic output, score monotonicity, valid output range

### Visualization Suite
- 11 multi-panel figures covering model performance comparison, ranking metrics, feature importance, data distributions, system performance, clustering, A/B testing, advanced analytics, trade-offs, search quality, testing results
- Each figure uses 2x3 or 1x3 subplot grids with appropriate chart types (bar, line, scatter, histogram, heatmap, violin)

---

## Comprehensive Description

This system implements a two-stage recommendation architecture. The retrieval stage uses matrix factorization (SVD with 100 latent factors, NMF with 50 factors) and collaborative filtering (KNN with k=40 neighbors) to decompose the sparse user-item rating matrix and generate candidate sets. TruncatedSVD produces 50-dimensional embeddings enabling cosine similarity search. The ranking stage applies gradient boosting models (LightGBM with 31 leaves, XGBoost with depth-6 trees) trained on engineered features aggregating user and item statistics to score candidates. A hybrid approach combines SVD and LightGBM predictions with equal weighting.

The system processes the Goodbooks-10k dataset containing 980K ratings from 53K users on 10K books after filtering for users/items with minimum 5 interactions. Data preprocessing includes deduplication, label encoding for categorical IDs, and CSR sparse matrix construction achieving 99.8% sparsity. Feature engineering creates user-level and item-level aggregate statistics (mean, standard deviation, count) used as ranking model inputs.

Deployment infrastructure provides a RecommendationSystem class exposing get_recommendations() method that executes retrieval (top-100 candidates via embedding similarity), ranking (LightGBM/XGBoost/SVD scoring), and top-N selection. The system tracks per-request latency achieving P95 of 50-80ms and supports batch processing at 15-25 users/sec throughput. SystemMonitor logs errors, aggregates performance metrics, and computes health status based on error rate thresholds.

Evaluation uses multiple metric categories: accuracy (RMSE 0.75-0.85, R² 0.3-0.4), ranking quality (NDCG@10 of 0.15-0.25, Precision@10, Recall@10), and search relevance (MRR, MAP, Hit Rate). The system computes catalog coverage (2-5%), Gini diversity coefficient (0.6-0.8), and analyzes long-tail distribution by splitting items into head/mid/tail segments based on popularity percentiles.

Optimization includes grid search hyperparameter tuning selecting parameters minimizing validation RMSE, learning-to-rank with LambdaRank objective optimizing NDCG directly, and query expansion using embedding similarity. Statistical testing compares models via t-tests, Mann-Whitney U tests, and Cohen's d effect sizes with 95% confidence intervals.

Analysis components evaluate cold-start performance on low-activity users, measure temporal consistency across time-split test periods, quantify popularity bias, and assess personalization effectiveness through K-Means user clustering into 5 segments. The testing framework validates recommendation generation, benchmarks against performance SLAs, checks data quality, and verifies model output consistency across 20+ test cases.

The system generates 11 visualization figures documenting model comparisons, ranking metrics at multiple K values, feature importance, data distributions, latency/throughput curves, user clustering, statistical test results, trade-off analyses, and testing outcomes. Output artifacts include performance reports in JSON format and test summaries in text format.

---


## Target Audience and Operation Overview

**Target Audience**:
- ML engineers implementing production recommendation systems
- Data scientists evaluating collaborative filtering approaches
- Software engineers deploying recommendation APIs
- Researchers analyzing recommendation algorithms and metrics

**Operation Overview**:

The system operates in training and inference modes. Training mode loads the Goodbooks-10k dataset, applies preprocessing (deduplication, filtering, encoding), constructs sparse matrices, trains 6 models (SVD, NMF, Item CF, User CF, LightGBM, XGBoost) with hyperparameter optimization, and evaluates on held-out test sets. Inference mode accepts user_id as input, retrieves top-100 candidates via embedding similarity search, ranks candidates using trained LightGBM/XGBoost models or SVD predictions, applies optional hybrid combination, and returns top-N item_ids with scores and latency metrics.

Batch processing mode accepts lists of user_ids, processes in configurable batch sizes, computes throughput, and aggregates results. Monitoring mode tracks errors by type, logs performance metrics per request, computes latency percentiles and health status, and generates alerts when error rates exceed thresholds.

Evaluation mode computes accuracy metrics (RMSE, MAE, R²), ranking metrics (Precision@K, Recall@K, NDCG@K), search quality metrics (MRR, MAP), diversity metrics (coverage, Gini), and conducts statistical tests comparing model performance. Analysis mode segments users/items, evaluates cold-start cases, measures temporal drift, quantifies biases, and generates visualization outputs.

Testing mode executes unit tests validating API behavior, benchmark tests checking performance SLAs, data quality tests verifying input integrity, and consistency tests ensuring deterministic outputs. All operations log to SystemMonitor, record metrics to performance logs, and generate reports in JSON/text formats.

---