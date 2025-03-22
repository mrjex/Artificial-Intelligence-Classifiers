import classifiers.svm.linear.linear_kernel as linearKernel
import classifiers.svm.polynomial.polynomial_kernel as polynomialKernel
import classifiers.svm.rbf.rbf_kernel as rbfKernel
import classifiers.random_forest.random_forest as randomForest
import classifiers.naive_bayes.naive_bayes as naiveBayes
import classifiers.kNN.kNN as kNN


# NOTE: Uncomment the classifier that you wish to run below


linearKernel.runLinearKernel()

# polynomialKernel.runPolynomialKernel()

# rbfKernel.runKernelRBF()

# randomForest.runRandomForest()

# naiveBayes.runNaiveBayes()

# kNN.runKNN()