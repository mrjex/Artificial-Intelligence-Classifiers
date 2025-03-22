from sklearn.datasets import fetch_openml
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import json
import time

pathFromMain = "classifiers/svm/linear" # Since "main.py" is the entrypoint of this script


# Retrieve MNIST images
def fetchDataset():
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist.data, mnist.target

    return X, y


# SVM with linear kernel
def getLinearKernelPipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='linear'))
    ])

# Get the predefined hyperparameters
def getParameterC():
    return {
        'svc__C': [0.001, 0.004, 0.005, 0.01, 0.02, 0.015, 0.05, 0.1, 1, 10]
    }


# Report and store the results of the execution
def reportResults(grid_search, start_time, X, y):
    jsonObj1 = {
        "Best Parameters": grid_search.best_params_,
        "Best Cross-Validation Accuracy": grid_search.best_score_
    }

    with open(f"{pathFromMain}/besult_results.json", "w") as write_file:
        json.dump(jsonObj1, write_file, indent=4)


    # Declare JSON object in a more global scope before assigning key-value pairs
    jsonObj2 = {

    }
    cv_results = grid_search.cv_results_

    # Store data into the dictionary object
    for mean_score, std_dev, params in zip(cv_results["mean_test_score"], cv_results["std_test_score"], cv_results["params"]):
        currentCVal = params['svc__C']
        keyName = f"C={currentCVal}"

        jsonObj2[keyName] = {
            "Mean Accuracy": f"{mean_score:.4f}",
            "other ratio": f"+/-{std_dev:.4f}"
        }

    # Write hypertuning output to the corresponding JSON file
    with open(f"{pathFromMain}/hypertuning_results.json", "w") as write_file:
        json.dump(jsonObj2, write_file, indent=4)

    # Declare skeleton of output JSON object for 'time_management.json'
    jsonObj3 = {
        "totalTimeElapsed": 0,
        "mean_fit_time": [],
        "std_fit_time": [],
        "mean_score_time": [],
        "std_score_time": []
    }

    for mean_fit_time, std_fit_time, mean_score_time, std_score_time in zip(cv_results["mean_fit_time"], cv_results["std_fit_time"], cv_results["mean_score_time"], cv_results["std_score_time"]):
        jsonObj3["mean_fit_time"].append(mean_fit_time)
        jsonObj3["std_fit_time"].append(std_fit_time)
        jsonObj3["mean_score_time"].append(mean_score_time)
        jsonObj3["std_score_time"].append(std_score_time)

    end_time = time.time()
    elapsed_time = end_time - start_time
    jsonObj3["totalTimeElapsed"] = elapsed_time

    with open(f"{pathFromMain}/time_management.json", "w") as write_file:
        json.dump(jsonObj3, write_file, indent=4)

    # Print the best model to the console output
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X)
    print(classification_report(y, y_pred))


def runLinearKernel():
    start_time = time.time()
    X, y = fetchDataset()

    X, _, y, _ = train_test_split(X, y, train_size=10000, stratify=y, random_state=42) # For quick experimentation, we run on 10000 samples as opposed to the entire dataset
    # X, _, y, _ = train_test_split(X, y, train_size=len(X) - 10, stratify=y, random_state=42) # NOTE: Run this line if you want to train all data (this takes a lot of time)

    pipelineObj = getLinearKernelPipeline()
    param_grid = getParameterC()

    # Set cross validation folds to 10 and focus on the attained accuracies
    grid_search = GridSearchCV(pipelineObj, param_grid, cv=10, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)

    reportResults(grid_search, start_time, X, y)