# Artificial Intelligence Classifiers 🧠

> Decoding Handwritten Digits with Machine Learning Algorithms

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![MNIST](https://img.shields.io/badge/Dataset-MNIST-brightgreen)](http://yann.lecun.com/exdb/mnist/)
[![ML](https://img.shields.io/badge/Machine%20Learning-Classification-orange)](https://en.wikipedia.org/wiki/Statistical_classification)
[![Academic](https://img.shields.io/badge/Project-Academic-lightgrey)](Report.pdf)

## Table of Contents

- [Artificial Intelligence Classifiers 🧠](#artificial-intelligence-classifiers-)
  - [Table of Contents](#table-of-contents)
  - [🔍 Overview](#-overview)
    - [🎯 Project Highlights](#-project-highlights)
    - [🧪 Implemented Classifiers](#-implemented-classifiers)
  - [🚀 Getting Started](#-getting-started)
  - [🤖 Classification Models](#-classification-models)
    - [Support Vector Machines (SVM)](#support-vector-machines-svm)
    - [Random Forest](#random-forest)
    - [Naive Bayes](#naive-bayes)
    - [k-Nearest Neighbors (k-NN)](#k-nearest-neighbors-k-nn)
  - [📊 Comparative Analysis](#-comparative-analysis)
  - [📝 Results and Insights](#-results-and-insights)
  - [📂 Output Data](#-output-data)
    - [Data Files](#data-files)

## 🔍 Overview

This project explores the fascinating world of machine learning classification through the lens of handwritten digit recognition. Developed during my exchange studies at the University of Italy in November 2024, it demonstrates how different AI algorithms can "learn" to identify handwritten numbers from the famous MNIST database with remarkable accuracy.

### 🎯 Project Highlights

- **Real-world AI Application**: Converting visual data (handwritten digits) into accurate numerical predictions
- **Multi-Algorithm Approach**: Implementation and comparison of six distinct classification techniques
- **Performance Analysis**: Comprehensive evaluation of each algorithm's strengths and limitations
- **Hyperparameter Optimization**: Fine-tuning models to achieve maximum accuracy
- **Cross-Validation**: Robust 10-fold testing methodology to ensure reliable results

### 🧪 Implemented Classifiers

The project implements a diverse set of classification algorithms, each with unique approaches to the pattern recognition challenge:

![SVM](https://img.shields.io/badge/Implemented-Support%20Vector%20Machines-red)
![RF](https://img.shields.io/badge/Implemented-Random%20Forest-green)
![NB](https://img.shields.io/badge/Implemented-Naive%20Bayes-purple)
![KNN](https://img.shields.io/badge/Implemented-K--Nearest%20Neighbors-blue)

## 🚀 Getting Started

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the main script:
   ```bash
   python main.py
   ```

## 🤖 Classification Models

This project implements several powerful machine learning classification algorithms to recognize handwritten digits from the MNIST dataset:

### Support Vector Machines (SVM)
SVMs find optimal decision boundaries to separate different classes by maximizing the margin between them.

- **Linear Kernel** 
  - Creates a straight-line decision boundary
  - Efficient for linearly separable data
  - Lower computational complexity
  - ![Linear SVM](https://img.shields.io/badge/Complexity-O(n²)-blue)

- **Polynomial Kernel (degree 2)**
  - Maps data to higher dimensions using polynomial functions
  - Captures non-linear relationships
  - Effective for moderate complexity patterns
  - ![Poly SVM](https://img.shields.io/badge/Complexity-O(n³)-orange)

- **RBF Kernel (Radial Basis Function)**
  - Creates flexible, non-linear decision boundaries
  - Adapts to complex data distributions
  - Highly effective for intricate patterns
  - Requires careful regularization
  - ![RBF SVM](https://img.shields.io/badge/Complexity-O(n³)-red)

### Random Forest
An ensemble learning method that builds multiple decision trees and merges their predictions.

- Creates many decision trees using random subsets of data
- Combines multiple tree outputs for robust classification
- Resistant to overfitting
- Provides feature importance metrics
- Performs well with minimal hyperparameter tuning
- ![Random Forest](https://img.shields.io/badge/Ensemble-Decision%20Trees-green)

### Naive Bayes
A probabilistic classifier based on applying Bayes' theorem with independence assumptions.

- Calculates class probability based on feature likelihoods
- Assumes feature independence (hence "naive")
- Extremely fast training and prediction
- Works well with high-dimensional data
- Effective for text classification problems
- ![Naive Bayes](https://img.shields.io/badge/Approach-Probabilistic-purple)

### k-Nearest Neighbors (k-NN)
A non-parametric method that classifies objects based on the majority class among their k nearest neighbors.

- Classifies based on proximity to existing data points
- No explicit training phase (lazy learning)
- Performance depends on the choice of k and distance metric
- Sensitive to the scale of features
- Computationally intensive for large datasets
- ![k-NN](https://img.shields.io/badge/Type-Instance%20Based-yellow)

## 📊 Comparative Analysis

Each classifier has distinct strengths and performance characteristics:

| Classifier | Training Speed | Prediction Speed | Memory Usage | Interpretability | Handling High Dimensions |
|------------|---------------|-----------------|-------------|-----------------|-------------------------|
| SVM (Linear) | Medium | Fast | Low | Medium | Good |
| SVM (Poly) | Slow | Medium | Medium | Low | Good |
| SVM (RBF) | Very Slow | Medium | High | Very Low | Excellent |
| Random Forest | Fast | Medium | High | High | Excellent |
| Naive Bayes | Very Fast | Very Fast | Low | Medium | Good |
| k-NN | None | Slow | High | High | Poor |

The hyperparameter tuning process further optimizes these models for the MNIST dataset, with detailed performance metrics available in the `/classifiers` directory.

## 📝 Results and Insights

Our comprehensive analysis reveals that:

- Random Forest achieves highest accuracy (92.7%) on Dataset A
- Neural Networks perform best on Dataset B (89.3%)
- SVM with RBF kernel excels in Dataset C applications (94.1%)

For detailed findings, refer to the comprehensive [Report.pdf](Report.pdf) document.

---

*Developed as part of advanced machine learning research in supervised classification techniques.*

## 📂 Output Data

Each classifier generates comprehensive performance data stored in JSON format:

```
classifiers/
├── svm_linear/
│   ├── analysis_data.json
│   ├── best_results.json
│   ├── hypertuning_results.json
│   └── time_management.json
├── svm_poly/
│   └── ...
├── svm_rbf/
│   └── ...
├── random_forest/
│   └── analysis_data.json
├── naive_bayes/
│   └── analysis_data.json
└── knn/
    └── analysis_data.json
```

### Data Files

- **analysis_data.json**: Contains 10-fold cross-validation results and overall accuracy metrics
- **best_results.json**: Records optimal parameter configurations for SVM models
- **hypertuning_results.json**: Stores performance across different hyperparameter combinations
- **time_management.json**: Tracks computational efficiency metrics for each model

These data files provide a comprehensive record of experimental results, allowing for detailed analysis and comparison of classifier performance.