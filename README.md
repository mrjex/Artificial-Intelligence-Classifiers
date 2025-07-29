# Artificial Intelligence Classifiers

> Decoding Handwritten Digits with **Support Vector Machines**, **Random Forests**, **Naive Bayes** and **K-NN**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![MNIST](https://img.shields.io/badge/Dataset-MNIST-brightgreen)](http://yann.lecun.com/exdb/mnist/)
[![ML](https://img.shields.io/badge/Machine%20Learning-Classification-orange)](https://en.wikipedia.org/wiki/Statistical_classification)
[![Academic](https://img.shields.io/badge/Project-Academic-lightgrey)](Report.pdf)

## Table of Contents

- [Artificial Intelligence Classifiers](#artificial-intelligence-classifiers)
  - [Table of Contents](#table-of-contents)
    - [Project Highlights](#project-highlights)
  - [Getting Started](#getting-started)
  - [Classification Models](#classification-models)
    - [Support Vector Machines (SVM)](#support-vector-machines-svm)
    - [Random Forest](#random-forest)
    - [Naive Bayes](#naive-bayes)
    - [k-Nearest Neighbors (k-NN)](#k-nearest-neighbors-k-nn)


### Project Highlights

- **Real-world AI Application**: Converting visual data (handwritten digits) into accurate numerical predictions
- **Multi-Algorithm Approach**: Implementation and comparison of six distinct classification techniques
- **Performance Analysis**: Comprehensive evaluation of each algorithm's strengths and limitations
- **Hyperparameter Optimization**: Fine-tuning models to achieve maximum accuracy
- **Cross-Validation**: Robust 10-fold testing methodology to ensure reliable results


## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the main script:
   ```bash
   python main.py
   ```

## Classification Models

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
