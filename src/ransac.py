import numpy as np
import random

class RANSAC:
    def __init__(self, data, model=None, min_sample_size=10, distance_threshold=0.05, max_iterations=100):
        self.data = data
        self.model = model
        self.min_sample_size = min_sample_size
        self.distance_threshold = distance_threshold
        self.max_iterations = max_iterations

    def run(self):
        best_model = None
        best_inliers = []
        best_score = 0

        for i in range(self.max_iterations):
            sample = np.random.sample(self.data, self.min_sample_size)
            model = self.model.fit(sample)
            inliers = []
            for point in self.data:
                if self.model.distance(point, model) < self.distance_threshold:
                    inliers.append(point)
            score = len(inliers)
            if score > best_score:
                best_model = model
                best_inliers = inliers
                best_score = score

        return best_model, best_inliers