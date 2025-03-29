import numpy as np
import random

class RANSAC:
    def __init__(self, data, model, min_sample_size=10, distance_threshold=0.05, max_iterations=100):
        """
        Initializes the RANSAC algorithm.
        :param data: Input dataset (numpy array of shape (N,2) for 2D data)
        :param model: Model to be fitted (e.g., LinearModel)
        :param min_sample_size: Minimum number of samples required to estimate a model
        :param distance_threshold: Distance threshold to determine inliers
        :param max_iterations: Number of iterations for RANSAC
        """
        self.data = data
        self.model = model
        self.min_sample_size = min_sample_size
        self.distance_threshold = distance_threshold
        self.max_iterations = max_iterations

    def run(self):
        """
        Runs the RANSAC algorithm to find the best model fitting the data.
        :return: Best model and list of inliers
        """
        best_model = None
        best_inliers = []
        best_score = 0

        for i in range(self.max_iterations):
            # Randomly select a subset of data points
            sample_indices = random.sample(range(len(self.data)), self.min_sample_size)
            sample = self.data[sample_indices]
            
            # Fit the model to the sample data
            model_params = self.model.fit(sample)
            inliers = []

            # Determine inliers based on the distance threshold
            for point in self.data:
                if self.model.distance(point, model_params) < self.distance_threshold:
                    inliers.append(point)
            
            score = len(inliers)
            
            # Update best model if the current model has more inliers
            if score > best_score:
                best_model = model_params
                best_inliers = inliers
                best_score = score
                print(f"Iteration {i}: Found new best model with {best_score} inliers")

        print("Final model found with", best_score, "inliers")
        return best_model, np.array(best_inliers)

class LinearModel:
    def fit(self, data):
        """
        Fits a linear model to the given 2D data using least squares.
        :param data: Numpy array of shape (N,2)
        :return: Model parameters (slope m and intercept c)
        """
        x = data[:, 0]
        y = data[:, 1]
        A = np.vstack([x, np.ones(len(x))]).T
        model, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        return model

    def distance(self, point, model):
        """
        Computes the vertical distance from a point to the fitted line.
        :param point: (x, y) coordinates
        :param model: (slope, intercept)
        :return: Absolute distance
        """
        m, c = model
        x, y = point
        return abs(y - (m * x + c))

# Example usage
data = np.random.rand(100, 2)  # Generate some random 2D data points
ransac = RANSAC(data, model=LinearModel(), min_sample_size=5, distance_threshold=0.1, max_iterations=50)
best_model, inliers = ransac.run()
print("Best model parameters:", best_model)