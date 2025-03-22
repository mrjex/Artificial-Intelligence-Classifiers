import numpy as np
from sklearn.datasets import fetch_openml
from scipy.special import gammaln
from scipy.optimize import minimize

import json
import time

pathFromMain = "classifiers/naive_bayes"
start_time = time.time()

def fetchDataset():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target

    X = X / 255.0
    y = y.astype(int)

    return X, y


# Returns an estimation of both parameters
def getParametersEstimation(data):
    mean = np.mean(data)
    variance = np.var(data)

    # Set initial values for both parameters
    alpha_0 = mean * ((mean * (1 - mean)) / variance - 1)
    beta_0 = alpha_0 * (1 / mean - 1)

    # Refine Alpha and Beta
    def loss(params):
        alpha, beta = params
        log_likelihood = np.sum((alpha - 1) * np.log(data) + (beta - 1) * np.log(1 - data))
        log_likelihood -= np.sum(gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta))
        return -log_likelihood

    bounds = [(1e-6, None), (1e-6, None)]
    result = minimize(loss, [alpha_0, beta_0], bounds=bounds)
    return result.x


# As the Beta parameter concerns the influence of the prior knowledge, this function
# calculates the probability density
def calculateBetaProbability(x, alpha, beta):
    B = np.exp(gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta))
    return (x ** (alpha - 1)) * ((1 - x) ** (beta - 1)) / B



# Naive Bayes Classifier using Beta distribution
class NaiveBayesBetaDistribution:
    def __init__(self):
        self.alpha = {}
        self.beta = {}
        self.priors = {}

    # Calculate or estimate the parameters for the pixels
    def calculateParams(self, X, y):
        self.classes = np.unique(y)
        for c in self.classes:
            X_c = X[y == c]
            self.alpha[c] = []
            self.beta[c] = []
            self.priors[c] = X_c.shape[0] / X.shape[0]

            for pixel in range(X_c.shape[1]):
                pixel_data = X_c[:, pixel]
                alpha, beta = getParametersEstimation(pixel_data)
                self.alpha[c].append(alpha)
                self.beta[c].append(beta)

            self.alpha[c] = np.array(self.alpha[c])
            self.beta[c] = np.array(self.beta[c])
    
    # Predict the class for all instances
    def predictClass(self, X):
        predictions = [self._getProbability(x) for x in X]
        return np.array(predictions)
    
    # Get probability for a specified instance
    def _getProbability(self, x):
        posteriors = {}
        for c in self.classes:
            prior = np.log(self.priors[c])
            likelihood = np.sum(
                np.log(calculateBetaProbability(x, self.alpha[c], self.beta[c]) + 1e-9)  # Avoid log(0)
            )
            posteriors[c] = prior + likelihood
        return max(posteriors, key=posteriors.get)


# Run 10 way cross validation
def cross_val_score(model, X, y, jsonObj, folds=10):
    fold_size = len(X) // folds
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    accuracies = []
    for fold in range(folds):
        currentFoldStartTime = time.time()

        test_idx = indices[fold * fold_size:(fold + 1) * fold_size]
        train_idx = np.setdiff1d(indices, test_idx)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.calculateParams(X_train, y_train)
        predictions = model.predictClass(X_test)

        accuracy = np.mean(predictions == y_test)
        accuracies.append(accuracy)

        currentFoldEndTime = time.time()
        currentFoldElapsedTime = currentFoldEndTime - currentFoldStartTime

        jsonObj[f"Fold{fold + 1}"] = {
            "Accuracy": accuracy,
            "Time Consumed": currentFoldElapsedTime
        }
        print(f"Fold {fold + 1} done, accuracy: {accuracy}, time: {currentFoldElapsedTime}")

    return accuracies


# Write output to JSON files
def reportResults(accuracies, jsonObj):
    end_time = time.time()
    elapsed_time = end_time - start_time
    jsonObj["Total Time Elapsed (seconds)"] = elapsed_time

    # Print results
    print("Accuracies for each fold:", accuracies)
    print("Mean accuracy:", np.mean(accuracies))
    print("Standard deviation:", np.std(accuracies))

    jsonObj["Mean Accuracies"] = np.mean(accuracies)
    jsonObj["Standard Deviation"] = np.std(accuracies)

    with open(f"{pathFromMain}/analysis_data.json", "w") as write_file:
        json.dump(jsonObj, write_file, indent=4)


def runNaiveBayes():
    jsonObj = {"Total Time Elapsed (seconds)": -1}

    naiveBayesObj = NaiveBayesBetaDistribution()
    X, y = fetchDataset()

    accuracies = cross_val_score(naiveBayesObj, X, y, jsonObj, folds=10) # Perform the 10-way cross validation on the model
    reportResults(accuracies, jsonObj) # Write and store the results in stdout-console and files
