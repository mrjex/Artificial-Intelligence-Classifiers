from sklearn.datasets import fetch_openml
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

import json
import time

pathFromMain = "classifiers/svm/polynomial"

def fetchDataset():
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist.data, mnist.target

    return X, y


# Write to the corresponding output JSON files to store the information tied to the latest execution
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

    # Systematically iterate through each hypertuning value for 'C' and store all tries
    cv_results = grid_search.cv_results_
    for mean_score, std_dev, params in zip(cv_results["mean_test_score"], cv_results["std_test_score"], cv_results["params"]):

        # Set the key-name to the current value of C
        currentCVal = params['svc__C']
        keyName = f"C={currentCVal}"

        jsonObj2[keyName] = {
            "Mean Accuracy": f"{mean_score:.4f}",
            "other ratio": f"+/-{std_dev:.4f}"
        }
    
    # Write output to the hypertuning file
    with open(f"{pathFromMain}/hypertuning_results.json", "w") as write_file:
        json.dump(jsonObj2, write_file, indent=4)
    
    # Initialize JSON object key-structure
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

    # Write output to the time-management file
    with open(f"{pathFromMain}/time_management.json", "w") as write_file:
        json.dump(jsonObj3, write_file, indent=4)


# SVM with polynomial kernel
def getPolyPipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='poly'))
    ])


def getParameterC():
    return {
        'svc__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    }


def runPolynomialKernel():
    start_time = time.time()
    X, y = fetchDataset()

    # NOTE: To efficiently debug and execute, the training size is set to 10000. If you wish to perform
    #       the entire training set, then set 'train_size=60000'.
    X, _, y, _ = train_test_split(X, y, train_size=10000, stratify=y, random_state=42)

    pipelineObj = getPolyPipeline()
    param_grid = getParameterC()

    grid_search = GridSearchCV(estimator=pipelineObj, param_grid=param_grid, cv=10, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)

    reportResults(grid_search, start_time)