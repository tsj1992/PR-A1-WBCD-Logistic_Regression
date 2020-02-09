import numpy as np


class LogisticRegression:

    def __init__(self, learning_rate=0.001, iterations=1000, threshold=0.5):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.threshold = threshold

    # X = N-dimensional vector (N - no.of features) - size M (M - no.of samples)
    # Y = 1-dimensional vector - size M (M - no.of samples)
    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)  # vector of only zeros
        self.bias = 0

        # gradient descent
        for _ in range(self.iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_classes = [1 if i >= self.threshold else 0 for i in y_predicted]
        return np.asarray(y_predicted_classes)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
