# AI Project - Joel Mattsson's Classifier Implementations

This system was developed as an individual assignment in 2024 November during my exchange studies in Italy. It covers a classification problem in machine learning that fetches hand-written images from MNIST database. Using supervised learning, it is up to the classifiers to correctly categorize the data instances. This is the second assignment of `Foundations of Artificial Intelligence` and it emphasized focus on the writing part, which can be found in `Report.pdf`. However, the coding part was also essential, since the results derived from the performances were used to obtain insights and make conclusions. Below, are the classifiers implemented in this assignment:

- Support Vector Machines (SVM)
  - **Linear Kernel**
  - Non-Linear Kernels
    - **Polynomial (degree 2) Kernel**
    - **RBF Kernel**
- **Random Forest**
- **Naive Bayes**
- **k-NN**


## Table of Contents

- [AI Project - Joel Mattsson's Classifier Implementations](#ai-project---joel-mattssons-classifier-implementations)
  - [Table of Contents](#table-of-contents)
  - [Getting Started](#getting-started)
  - [Output Data](#output-data)


## Getting Started

Navigate to `main.py` and uncomment the selected classifier. Don't forget to comment the previous classifier unless you wish to run multiple classifiers at once.


## Output Data

Each classifier generates output in `.json` files that store information about the execution such as the 10 way cross validation, the accuracies achieved and the total time required. Inside the selected classifier in `/classifiers` directory, *analysis_data.json* contains all associated information, except for the three support vector machine classifiers, that use **hypertuning** as well. To inituitively display this date and their produced accuracies, three separate JSON files are used: *best_results.json*, *hypertuning_results* and *time_management.json*