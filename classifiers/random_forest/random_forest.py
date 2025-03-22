from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import cross_val_score
import numpy as np

import json
import time

pathFromMain = "classifiers/random_forest"


def fetchDataset():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target

    return X, y


# Write the results from the classifier in JSON format
def reportResults(cv_scores, start_time):
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Store data in dictionary structure
    jsonObj = {
        "totalTimeElapsed": elapsed_time,
        "Cross-validation accuracy": str(cv_scores),
        "Mean accuracy:": str(np.mean(cv_scores) * 100),
        "Standard deviation of accuracy": str(np.std(cv_scores) * 100)
    }

    # Write the dictionary structured variable in JSON file
    with open(f"{pathFromMain}/analysis_data.json", "w") as write_file:
        json.dump(jsonObj, write_file, indent=4)


def runRandomForest():
    X, y = fetchDataset()
    start_time = time.time()

    randomForestClassifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Enable cross validation and set it to 10 folds, while prioritizing accuracy as the scoring metric
    cv_scores = cross_val_score(randomForestClassifier, X, y, cv=10, scoring='accuracy')

    reportResults(cv_scores, start_time)