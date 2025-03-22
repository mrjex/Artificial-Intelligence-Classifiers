from sklearn.datasets import fetch_openml
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import json
import time

pathFromMain = "classifiers/svm/rbf"


def fetchDataset():
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist.data, mnist.target
    
    return X, y


def reportResults(grid_search, start_time):
    jsonObj1 = {
        "Best Parameters": grid_search.best_params_,
        "Best Cross-Validation Accuracy": grid_search.best_score_
    }

    with open(f"{pathFromMain}/besult_results.json", "w") as write_file:
        json.dump(jsonObj1, write_file, indent=4)

    # Declare JSON object more globally to access it multiple times in this function before writing it to a file
    jsonObj2 = {

    }

    cv_results = grid_search.cv_results_
    for mean_score, std_dev, params in zip(cv_results["mean_test_score"], cv_results["std_test_score"], cv_results["params"]):

        # Define the name of the key that represents the current combination of the two hyperparameters 'C' and 'gamma'
        currentCVal = params['svc__C']
        currentGVal = params['svc__gamma']
        key = f"C={currentCVal}, gamma={currentGVal}"

        # For each combination of 'C' and 'gamma', store the values in the specified key as a nested object
        jsonObj2[key] = {
            "Mean Accuracy": f"{mean_score:.4f}",
            "ratio": f"+/-{std_dev:.4f}"
        }

    with open(f"{pathFromMain}/hypertuning_results.json", "w") as write_file:
        json.dump(jsonObj2, write_file, indent=4)

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


# Set the support vector machine's kernel to 'RBF'
def getPipelineRBF():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf'))     
    ])


# Returns the two hyperparameters
#   C: Regularization
#   Gamma: Coefficient for the RBF kernel
def getHyperparameters():
    return {
        'svc__C': [0.01, 0.2, 1],
        'svc__gamma': [0.001, 0.1, 1]
    }


def runKernelRBF():
    start_time = time.time()
    X, y = fetchDataset()

    # NOTE: To efficiently debug and execute, the training size is set to 10000. If you wish to perform
    #       the entire training set, then set 'train_size=60000'.
    X, _, y, _ = train_test_split(X, y, train_size=10000, stratify=y, random_state=42)

    pipelineObj = getPipelineRBF()
    param_grid = getHyperparameters()

    grid_search = GridSearchCV(estimator=pipelineObj, param_grid=param_grid, cv=10, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)

    reportResults(grid_search, start_time)