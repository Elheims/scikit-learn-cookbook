# scikit-learn Cookbook

Code reproduction and theoretical deep-dive of **scikit-learn Cookbook, Third Edition** (John Sukup, Packt) — *Over 80 recipes for machine learning in Python with scikit-learn*.

This is an individual assignment (*Tugas 2 — Enrichment for Machine Learning Classes*). For every chapter I:

1. **Reproduce** the book's code in a Jupyter notebook and execute it (outputs are saved).
2. **Summarize** each section of the chapter.
3. **Explain the theory** behind every concept in Markdown cells next to the code.

Cells labelled **Extension** are *not* in the book's code; they are small demos I added so that a concept the book only describes can actually be seen working. Where my output differs from the book's printed output, the notebook says why.

## Progress

| # | Chapter | Status |
|---|---|---|
| 1 | Common Conventions and API Elements of scikit-learn | Done |
| 2 | Pre-Model Workflow and Data Preprocessing | Done |
| 3 | Dimensionality Reduction Techniques | Done |
| 4 | Building Models with Distance Metrics and Nearest Neighbors | Done |
| 5 | Linear Models and Regularization | Done |
| 6 | Advanced Logistic Regression and Extensions | Done |
| 7 | Support Vector Machines and Kernel Methods | Done |
| 8 | Tree-Based Algorithms and Ensemble Methods | Done |
| 9 | Text Processing and Multiclass Classification | Done |
| 10 | Clustering Techniques | Done |
| 11 | Novelty and Outlier Detection | Done |
| 12 | Cross-Validation and Model Evaluation Techniques | Done |
| 13 | Deploying scikit-learn Models in Production | Done |

(Assignment scope: Chapters 1–5 were due 10 Oct 2026, 23:59; Chapters 9–13 are due 24 Oct 2026, 23:59. Chapters 6–8 are included as well.)

## Chapter overviews

### Chapter 1 — Common Conventions and API Elements of scikit-learn
Notebook: [`ch01_sklearn_api.ipynb`](01.%20API%20Elements%20of%20scikit-learn/ch01_sklearn_api.ipynb)

The conceptual foundation of the book. scikit-learn is designed around *consistency, simplicity, modularity and reusability*: almost every object exposes the same verbs. The chapter covers:

- **Design philosophy** — one uniform interface (`fit`, `predict`, `transform`) so models are interchangeable.
- **Estimators** — objects that learn from data (`fit`, `predict`, `fit_predict`); shown with `LinearRegression` and `KMeans`.
- **Transformers** — objects that change data using learned parameters (`fit`, `transform`, `fit_transform`); shown with `StandardScaler`, including why you never fit on test data (data leakage).
- **Custom estimators and transformers** — subclass `BaseEstimator` + mixins so your own code works in pipelines and searches.
- **Pipelines and MLOps** — chain steps into a single object; leak-free cross-validation; model persistence with `joblib`.
- **Common attributes and methods** — learned attributes end in `_` (`coef_`, `intercept_`); `score()` gives accuracy or R²; R² vs adjusted R².
- **Hyperparameter tuning** — `get_params`/`set_params`, `GridSearchCV`, `RandomizedSearchCV`, successive halving.
- **Metadata** — estimator *tags* and *metadata routing* (routers vs consumers, e.g. `sample_weight`).
- **Best practices** — uniform API, preprocessing, pipelines, cross-validation, tuning.

### Chapter 2 — Pre-Model Workflow and Data Preprocessing
Notebook: [`ch02_data_preprocessing.ipynb`](02.%20Data%20Preprocessing/ch02_data_preprocessing.ipynb)

"Garbage in, garbage out": data quality decides model quality, and preprocessing is an ongoing task, not a one-off. The chapter covers:

- **Impact of raw data** — missing values, outliers, categorical variables, unscaled features and data leakage, each with its scikit-learn remedy.
- **Missing data** — `SimpleImputer` (column statistic), `KNNImputer` (average of similar rows), `IterativeImputer` (regress each feature on the others); a hidden-value experiment compares their accuracy.
- **Scaling** — `StandardScaler` (z-score), `MinMaxScaler` (to [0, 1]), `Normalizer` (unit-norm rows); plus `RobustScaler`, `PowerTransformer` and `QuantileTransformer` from the book's decision guide, and which models need scaling at all.
- **Encoding categoricals** — `OneHotEncoder` (nominal), `LabelEncoder` / `OrdinalEncoder` (ordinal), `ColumnTransformer` for mixed tables.
- **Pipelines** — chain steps, split *before* transforming to avoid leakage (demonstrated numerically), visualise with `set_config(display="diagram")`.
- **Feature engineering** — creating features (`PolynomialFeatures`, `KBinsDiscretizer`) and selecting them (`RFE`, `SelectFromModel`; filter vs wrapper methods).
- **Practical exercise** — a comprehensive pipeline on California Housing, built step by step with an ablation table (R² 0.60 → 0.69).

The notebook also lists a few inaccuracies found in the book's text (e.g. how `KNNImputer` works).

### Chapter 3 — Dimensionality Reduction Techniques
Notebook: [`ch03_dimensionality_reduction.ipynb`](03.%20Dimensionality%20Reduction/ch03_dimensionality_reduction.ipynb)

Finding "the right ingredients": fewer, better features give simpler, faster, less over-fitted models and make high-dimensional data visible. The chapter covers:

- **Why reduce** — simplicity, cost, generalisation, visualisation; feature *selection* vs feature *extraction*; the curse of dimensionality (demonstrated numerically).
- **PCA** (unsupervised, linear) — orthogonal directions of maximum variance on the Wine data; verified by hand with an eigendecomposition; scree/cumulative variance, loadings and a biplot, reconstruction error.
- **LDA** (supervised, linear) — maximises between-class vs within-class variance (Fisher criterion); side-by-side comparison with PCA and a leak-free cross-validated comparison.
- **t-SNE** (unsupervised, non-linear) — visualising the 10 handwritten-digit classes; effect of perplexity and initialisation; why it has no `transform` and what its plots do and don't mean.
- **Choosing a technique** — the book's decision flow and comparison table, plus a measured speed/capability comparison.
- **Impact on model performance** — accuracy and time vs number of PCA components (underfitting, plateau, diminishing returns).
- **Practical exercises** — PCA + logistic regression on Iris, and t-SNE + K-Means on digits (ARI 0.46 on raw pixels → 0.77 on the t-SNE map).

Book inaccuracies found here include the PCA arrow plot, the "MNIST" label for the small 8×8 digits set, and t-SNE being described like a supervised method.

### Chapter 4 — Building Models with Distance Metrics and Nearest Neighbors
Notebook: [`ch04_distance_metrics_knn.ipynb`](04.%20Models%20with%20Distance%20Metrics%20and%20Nearest%20Neighbors/ch04_distance_metrics_knn.ipynb)

Classification from similarity: a new point is labelled by its closest known neighbours. The chapter covers:

- **Distance metrics** — Euclidean (L2), Manhattan (L1) and Minkowski (Lp), plus cosine, Hamming, Jaccard and Chebyshev, computed by hand and with scikit-learn.
- **KNN** — voting among the `k` nearest points (mean for regression); built from scratch in NumPy and checked against `KNeighborsClassifier`; decision boundaries and the elbow plot for choosing `k` (bias vs variance).
- **Metric comparison** — the book's circles/checkerboard experiment, repeated over many datasets and splits; the effect of feature scaling, of irrelevant features, and of the neighbour-search algorithm.
- **Hyperparameter tuning** — `GridSearchCV` over `n_neighbors`, `weights` and `metric` (reproduces the book's result), `RandomizedSearchCV`, and why the "best" parameters often tie.
- **Evaluation** — cross-validation, learning curves, confusion matrix, precision / recall / F1 (recomputed by hand), and why accuracy misleads on imbalanced data (with over/undersampling and precision–recall curves compared).
- **Practical exercises** — breast cancer (imputation + scaling + KNN), wine (tuned pipeline), handwritten digits (10-class confusion matrix), and KNN regression on California Housing.

Book inaccuracies found here include `minkowski` being identical to Euclidean in the metric comparison (default `p=2`), the Manhattan "win" being noise, and a wrong description of what cross-validation does.

### Chapter 5 — Linear Models and Regularization
Notebook: [`ch05_linear_models.ipynb`](05.%20Linear%20Models%20and%20Regularization/ch05_linear_models.ipynb)

From a straight line to penalised and non-linear fits. The chapter covers:

- **Linear regression (OLS)** — the normal equation (verified by hand), residual diagnostics, R² and MSE, and how multicollinearity makes coefficients unstable while predictions stay fine.
- **Ridge (L2) and Lasso (L1)** — shrinkage vs exact zeros, coefficient-path plots on real data, the closed-form Ridge solution, why features must be scaled, and choosing `alpha` by cross-validation (`RidgeCV`, `LassoCV`).
- **ElasticNet** — mixing the penalties with `l1_ratio`, readable coefficient paths, the grouping effect on correlated features, and `ElasticNetCV`.
- **Regularization theory** — bias–variance trade-off shown as a train/test error curve, Bayesian and geometric views of L1 vs L2.
- **Polynomial and spline regression** — choosing the degree / number of knots by cross-validation against the noise floor, interpolation vs regression splines, and extrapolation behaviour.
- **Practical exercises** — Ridge, Lasso and ElasticNet on synthetic data, compared over 10 datasets on test error, sparsity and coefficient recovery.

This chapter has the most book corrections: the first synthetic dataset overwrites 7 of its 10 informative columns (so the models' R² of about 0.1 is not "multicollinearity"), the coefficient-path plot shows no shrinkage at the book's data scale, and the polynomial/spline conclusions are drawn from test-set selection and training error.

### Chapter 6 — Advanced Logistic Regression and Extensions
Notebook: [`ch06_advanced_logistic_regression.ipynb`](06.%20Advanced%20Logistic%20Regression/ch06_advanced_logistic_regression.ipynb)

Classification with regression machinery. The chapter covers:

- **Logistic regression** — the sigmoid and log-odds, maximum likelihood (log-loss checked by hand), probabilities computed from the coefficients, odds ratios, and why the 0.5 threshold is a choice.
- **Multiclass** — one-vs-rest vs multinomial (softmax), verified by hand; decision boundaries for both. The book's code for this recipe no longer runs on current scikit-learn and is adapted.
- **Regularization** — L2 vs L1 with `C` (inverse strength), sparsity (7 of 30 features survive at `C=0.1`), coefficient paths against `C`, `LogisticRegressionCV`, and the modern `l1_ratio` API.
- **Multilabel classification** — one classifier per label, `MultiOutputClassifier`, classifier chains, hamming loss vs exact-match accuracy, and the label co-occurrence analysis (reproduces the book's counts exactly).
- **Evaluation** — accuracy, precision, recall, F1, ROC and AUC (AUC verified by brute force), and how the decision threshold trades missed cancers against false alarms.
- **Practical exercises** — tuned L1/L2 models, multinomial evaluation on Iris, and a mesh-grid decision-boundary plot.

Book inaccuracies found here include code that no longer runs (`multi_class=`, `liblinear` for multiclass), a `SyntaxError` in the co-occurrence function, a wrong "high kurtosis" claim, and a Ridge-vs-Lasso conclusion that holds for only one class.

### Chapter 7 — Support Vector Machines and Kernel Methods
Notebook: [`ch07_svm_kernel_methods.ipynb`](07.%20Support%20Vector%20Machines%20and%20Kernel%20Methods/ch07_svm_kernel_methods.ipynb)

Maximum-margin classification and the kernel trick. The chapter covers:

- **SVM basics** — the maximum-margin hyperplane, support vectors, the soft margin and `C` (inspected directly: weight vector, margin width, `decision_function = w·x + b`), one-vs-one multiclass, and `SVR` on a real continuous target.
- **Kernels** — linear, polynomial and RBF; the kernel trick verified numerically (kernel value = dot product of an explicit feature map), a "lift to a higher dimension" demo, the effect of `gamma` on the two-moons data, and custom/precomputed kernels.
- **Tuning** — grid search over `C`, `kernel`, `degree` (reproduces the book's result and shows that 8 configurations tie), a `C` × `gamma` heat-map, randomized search, and a correctly structured grid.
- **High-dimensional data** — default SVMs overfit wide data; regularisation, RBF, L1 and `PCA(50)` → RBF (0.95) fix it, and standardising turns out to hurt because the feature scale carries signal.
- **Evaluation** — report, ROC and AUC; why a ROC needs only `decision_function`; calibrated probabilities with `CalibratedClassifierCV` (the book's `probability=True` is deprecated).
- **Practical exercises** — a simple SVM, a tuned SVM on breast cancer, and decision-function plots with margins and support vectors.

Book inaccuracies found here include a hyperplane described as "a curved line", a "decision boundary" figure that is actually an SVR output, a `NameError` in the evaluation recipe, and a high-dimensional recipe that promises a fix it never shows.

### Chapter 8 — Tree-Based Algorithms and Ensemble Methods
Notebook: [`ch08_tree_based_ensembles.ipynb`](08.%20Tree-Based%20Algorithms%20and%20Ensemble%20Methods/ch08_tree_based_ensembles.ipynb)

Recursive splits and the ensembles built from them. The chapter covers:

- **Decision trees** — Gini impurity and entropy computed by hand, the best split found by brute force (and the exact tie that scikit-learn breaks randomly), reading a fitted tree, a flower's path down it, the effect of each complexity hyper-parameter, validation curve, cost-complexity pruning, regression trees, and scale/NaN/categorical behaviour.
- **Random forests and bagging** — a forest built from scratch, soft voting verified, out-of-bag accuracy, repeated-CV comparison with a single tree on two datasets, and impurity vs permutation importance (with noise columns and correlated features).
- **Gradient boosting** — boosting written from scratch and matched exactly to `GradientBoostingRegressor`, staged curves, the learning-rate / number-of-trees trade-off, real early stopping, and `HistGradientBoosting`.
- **Tuning** — the book's grid search (20 of 27 configurations tie), `RandomizedSearchCV` (non-adaptive), the heat-map sliced by learning rate, and validation curves for a forest.
- **Comparing ensembles** — bagging vs boosting vs stacking over three datasets with repeated CV, and stacking rebuilt by hand from out-of-fold predictions.
- **Practical exercises** — a readable depth-3 tree, a tuned forest on Wine, and boosting vs forest on the digits.

Book inaccuracies found here include a Gini range of "up to 1.0", `RandomizedSearchCV` described as adaptive, "early stopping" confused with pre-pruning, and a forest-vs-tree "improvement" on Iris that vanishes under cross-validation.

### Chapter 9 — Text Processing and Multiclass Classification
Notebook: [`ch09_text_processing_multiclass.ipynb`](09.%20Text%20Processing%20and%20Multiclass%20Classification/ch09_text_processing_multiclass.ipynb)

From raw text to a classifier, on the NLTK `movie_reviews`, `reuters` and `brown` corpora. The chapter covers:

- **Text processing** — `CountVectorizer`, sparse document-term matrices, stop words (including why they remove negations), a Porter-stemming analyzer, and whether any of it changes accuracy (cross-validated).
- **Vectorization** — Bag of Words vs TF-IDF (TF-IDF reproduced by hand), per-class characteristic terms, BoW vs TF-IDF with a linear SVM, sparse vs dense memory, `HashingVectorizer`.
- **Feature extraction** — n-grams, POS tagging, a proper-noun proxy for NER, the vocabulary explosion of n-grams and their (lack of) effect on movie-review accuracy.
- **Classification models** — Naive Bayes, SVM and logistic regression on 15 Brown genres; the book's accuracies (0.228 / 0.332 / 0.324) are reproduced and then diagnosed: defaults under-fit TF-IDF, and tuned models tie at about 0.5.
- **Multiclass strategies** — OvR vs OvO (15 vs 105 models), ten repeated splits, and a multilabel example on Reuters.
- **Evaluation** — the book's report and confusion matrix (0.803) recomputed by hand, a confidence interval, ROC/AUC, cross-validation and a learning curve.
- **Practical exercises** — cleaning and vectorising, n-gram features, and an 8-topic Reuters classifier.

Book inaccuracies found here include a plot that draws 18,881 bars instead of 20, a split in the printed code that cannot produce Figures 9.8/9.9, an `except nltk.downloader.DownloadError` that does not exist, a Naive Bayes "performs well" claim contradicted by its own results, and an NER section without any NER code.

### Chapter 10 — Clustering Techniques
Notebook: [`ch10_clustering.ipynb`](10.%20Clustering%20Techniques/ch10_clustering.ipynb)

Unsupervised grouping of unlabelled data. The chapter covers:

- **Introduction** — what clustering needs (a distance), and a demo of why scaling matters (ARI 0.81 → 0.50 after a change of units).
- **K-means** — Lloyd's algorithm written by hand and matched exactly, initialisation and local minima (K-means++ and `n_init`), and the elbow method compared with silhouette, Davies-Bouldin, Calinski-Harabasz and ARI.
- **Hierarchical clustering** — dendrogram, four linkages (single and average chain), merge heights and cutting the tree.
- **DBSCAN** — core/border/noise points, the k-distance plot, and a sensitivity grid over `eps` and `min_samples`.
- **Evaluation metrics** — the book's silhouette / Davies-Bouldin / ARI (0.534 / 0.702 / 0.814) checked by hand, and a case where internal metrics prefer the wrong clustering.
- **Choosing an algorithm** — a 3 × 3 grid of algorithms and datasets, measured scalability and distance concentration in high dimensions.
- **Spectral clustering and GMMs** — soft assignments, BIC, covariance types and over-confidence.
- **Practical exercises** — K-means on Iris, DBSCAN vs K-means on moons, PCA + GMM on the digits.

Book inaccuracies found here include an elbow that says 2–3 for data generated with 4 centres, a `CalibratedClassifierCV` suggestion that cannot work for a GMM, a spectral-scalability claim that holds only conditionally, and `eps=0.2` presented without noting that it breaks after standardising.

### Chapter 11 — Novelty and Outlier Detection
Notebook: [`ch11_outlier_novelty_detection.ipynb`](11.%20Novelty%20and%20Outlier%20Detection/ch11_outlier_novelty_detection.ipynb)

Finding anomalies in training data (outliers) and in new data (novelties). The chapter covers:

- **Introduction** — outliers vs novelties, `contamination` (it fixes *how many* points are flagged, not which), and the `novelty=False/True` API of LOF.
- **Isolation Forest** — the anomaly score recomputed by hand, the cross-shaped decision surface, and the missing `feature_importances_`.
- **One-Class SVM** — what `nu` and `gamma` really do (measured), the effect of scaling, and the book's "failure" traced to a data bug.
- **LOF** — the algorithm written by hand, a local outlier ranked 16th by LOF vs 39th (Isolation Forest) and 133rd (One-Class SVM).
- **Evaluation** — the book's confusion matrix reproduced exactly, and its ROC-AUC of 0.005 explained as a sign error (0.995 once negated); threshold tuning and precision-recall.
- **Handling outliers** — removal, median replacement, capping and flagging, their side effects, and a regression experiment showing a robust model beating detect-and-remove.
- **Choosing a technique** — the book's comparison table tested on three labelled scenarios, timings and distance concentration.
- **Practical exercises** — Isolation Forest, LOF and One-Class SVM on simulated data.

Book inaccuracies found here include an inverted ROC-AUC, a One-Class SVM "failure" caused by test data drawn around a different centre, `nu` described backwards, `feature_importances_` that does not exist for Isolation Forest, and LOF wrongly described as unable to predict.

### Chapter 12 — Cross-Validation and Model Evaluation Techniques
Notebook: [`ch12_cross_validation.ipynb`](12.%20Cross-Validation%20and%20Model%20Evaluation%20Techniques/ch12_cross_validation.ipynb)

How to estimate generalisation honestly. The chapter covers:

- **Cross-validation** — k-fold vs stratified k-fold, fold composition on rare classes, and 300 repeated shuffles showing the book's 0.756 vs 0.766 is noise.
- **Advanced methods** — LOOCV (with the closed-form shortcut of `RidgeCV`), nested CV, selection bias on pure-noise data, data leakage through feature selection, `GroupKFold` and `TimeSeriesSplit`.
- **Implementing CV** — `cross_val_score`, `cross_validate`, `GridSearchCV`, a metrics overview (the recipe the book announces but omits), and out-of-fold predictions.
- **Model selection** — grid vs random search, what each actually evaluated, and how much the search itself varies across seeds.
- **Generalizability** — learning and validation curves with error bands, a model that really over-fits, and why a converged curve is not the same as a good model.
- **Practical exercises** — multi-metric CV, grid search with nested CV, and curves on the digits data.

Book inaccuracies found here include a "0.593 MSE" that appears nowhere (the nested-CV 0.480 is an R²), the claim that `cv=5` did not stratify (it does), random search described as using fewer evaluations (10 candidates vs 6), and a learning-curve reading that contradicts the book's own figure.

### Chapter 13 — Deploying scikit-learn Models in Production
Notebook: [`ch13_deploying_models.ipynb`](13.%20Deploying%20scikit-learn%20Models%20in%20Production/ch13_deploying_models.ipynb)

From a fitted model to a maintained service. The chapter covers:

- **Deployment overview** — round-trip verification, a simulated prediction endpoint with input validation, and single vs batch latency (about 700× per sample).
- **Serialization** — `pickle` vs `joblib` measured, a harmless demonstration that unpickling runs code, version-mismatch warnings and checksums.
- **Scaling** — `n_jobs` measured for single rows, batches, fitting and cross-validation (parallelism can be 3× slower for small requests).
- **Monitoring** — the book's stream reproduced exactly (accuracy 0.30–0.38, worse than chance), then a real drift simulation with static, online and trigger-based models and a label-free KS drift test.
- **Life cycle** — a working artifact with metadata, SHA-256 and a golden prediction snapshot, with upgrade, corruption and rollback scenarios.
- **Deployment pipelines** — training–serving skew, and a multi-check deployment gate compared with the book's.
- **Practical exercises** — reloading a pipeline, monitoring with retraining, and threshold-based deployment checks.

Book inaccuracies found here include a monitoring "simulation" on an unrelated dataset, a validation snapshot that crashes (Iris through a 20-feature model), a hard-coded library version, and a CI check on random noise that can never pass. All artifacts are written to a temporary folder, not to the repository.

## Repository structure

```
scikit-learn-cookbook/
├── 01. API Elements of scikit-learn/
│   └── ch01_sklearn_api.ipynb
├── 02. Data Preprocessing/
│   └── ch02_data_preprocessing.ipynb
├── 03. Dimensionality Reduction/
│   └── ch03_dimensionality_reduction.ipynb
├── 04. Models with Distance Metrics and Nearest Neighbors/
│   └── ch04_distance_metrics_knn.ipynb
├── 05. Linear Models and Regularization/
│   └── ch05_linear_models.ipynb
├── 06. Advanced Logistic Regression/
│   └── ch06_advanced_logistic_regression.ipynb
├── 07. Support Vector Machines and Kernel Methods/
│   └── ch07_svm_kernel_methods.ipynb
├── 08. Tree-Based Algorithms and Ensemble Methods/
│   └── ch08_tree_based_ensembles.ipynb
├── 09. Text Processing and Multiclass Classification/
│   └── ch09_text_processing_multiclass.ipynb
├── 10. Clustering Techniques/
│   └── ch10_clustering.ipynb
├── 11. Novelty and Outlier Detection/
│   └── ch11_outlier_novelty_detection.ipynb
├── 12. Cross-Validation and Model Evaluation Techniques/
│   └── ch12_cross_validation.ipynb
├── 13. Deploying scikit-learn Models in Production/
│   └── ch13_deploying_models.ipynb
└── README.md
```

## Getting started

Tested with Python 3.14, scikit-learn 1.9.1, NumPy 2.5.3.

```bash
pip install scikit-learn numpy pandas matplotlib seaborn scipy joblib nltk jupyter
jupyter lab
```

> The book was written for an earlier scikit-learn release; a few printed outputs differ slightly (e.g. NumPy arrays print without commas, `get_params()` lists depend on the version). These differences are noted in the notebooks.

## Reference

Sukup, J. *scikit-learn Cookbook, Third Edition*. Packt Publishing.

