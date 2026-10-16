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

(Assignment scope: Chapters 1–5. Deadline: 10 Oct 2026, 23:59.)

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
└── README.md
```

## Getting started

Tested with Python 3.14, scikit-learn 1.9.1, NumPy 2.5.3.

```bash
pip install scikit-learn numpy pandas matplotlib seaborn scipy joblib jupyter
jupyter lab
```

> The book was written for an earlier scikit-learn release; a few printed outputs differ slightly (e.g. NumPy arrays print without commas, `get_params()` lists depend on the version). These differences are noted in the notebooks.

## Reference

Sukup, J. *scikit-learn Cookbook, Third Edition*. Packt Publishing.

