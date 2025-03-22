import numpy as np
from tensorflow.keras.datasets import mnist
from collections import Counter

import json
import time


'''
--  NOTE FOR DEVELOPER  --

Running this on Windows may cause issues when fetching the dataset due to the UTF8 encoding. The
error "UnicodeEncodeError: 'charmap' codec can't encode characters" usually occurs due to issues
with character encoding, especially on Windows. When TensorFlow fetches the MNIST dataset this
error arises. 

SOLUTIONS:
    - Run on Ubuntu WSL terminal or using Virtualbox
    - Run on other operating system than Windows
'''

kValue = 3

pathFromMain = "classifiers/kNN"
start_time = time.time()

def fetchDataset():
    (X, y), _ = mnist.load_data()
    X = X.reshape(X.shape[0], -1) / 255.0
    
    return X, y

# Returns the 'K' closest neighbors for the specified data instance
def getKNearestNeighbors(X_train, y_train, X_test, k=3):
    distances = np.sqrt(((X_train - X_test) ** 2).sum(axis=1))
    nearest_indices = distances.argsort()[:k]
    nearest_labels = y_train[nearest_indices]
    most_common = Counter(nearest_labels).most_common(1)[0][0]
    return most_common

# Implement cross validaiton with default folds set to 10 and k=3
def cross_validate_knn(X, y, k=3, num_folds=10):
    fold_size = len(X) // num_folds
    accuracies = []

    # Initialize JSON object skeleton
    jsonObj1 = {
        "Total Time Elapsed (seconds)": -1,
        "Average Accuracy": -1,
        "Standard Deviation of Accuracy": -1
    }

    for fold in range(num_folds):
        currentFoldStartTime = time.time()

        start, end = fold * fold_size, (fold + 1) * fold_size
        X_val, y_val = X[start:end], y[start:end]
        X_train = np.concatenate([X[:start], X[end:]], axis=0)
        y_train = np.concatenate([y[:start], y[end:]], axis=0)

        correct_predictions = 0
        for i in range(len(X_val)):
            prediction = getKNearestNeighbors(X_train, y_train, X_val[i], k)
            if prediction == y_val[i]:
                correct_predictions += 1

        accuracy = correct_predictions / len(X_val)
        accuracies.append(accuracy)

        currentFoldEndTime = time.time()
        currentFoldElapsedTime = currentFoldEndTime - currentFoldStartTime

        # Map each fold from the cross validation to the assocaited performance of accuracy and total time consumed
        jsonObj1[f"Fold{fold + 1}"] = {
            "Accuracy": accuracy,
            "Time Consumed": currentFoldElapsedTime,
        }
    
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    end_time = time.time()
    elapsed_time = end_time - start_time

    jsonObj1["Total Time Elapsed (seconds)"] = elapsed_time
    jsonObj1["Average Accuracy"] = mean_accuracy
    jsonObj1["Standard Deviation of Accuracy"] = std_accuracy

    with open(f"{pathFromMain}/analysis_data.json", "w") as write_file:
        json.dump(jsonObj1, write_file, indent=4)


def runKNN():
    X, y = fetchDataset()
    cross_validate_knn(X, y, k=kValue, num_folds=10)