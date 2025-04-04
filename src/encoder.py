import numpy as np
import random

class Encoder:
    def __init__(self, patch_size=8):
        self.patch_size = patch_size

    def encode(self, image):
        pass

class RANSACEncoder(Encoder):
    def __init__(self, patch_size=8, min_sample_size=5, distance_threshold=10, max_iterations=50):
        super().__init__(patch_size)
        self.min_sample_size = min_sample_size
        self.distance_threshold = distance_threshold
        self.max_iterations = max_iterations
    
    def encode(self, image):
        h, w = image.shape
        latent_representation = np.zeros_like(image)
        noise_estimate = np.zeros_like(image, dtype=np.float32)
        
        for i in range(0, h, self.patch_size):
            for j in range(0, w, self.patch_size):
                patch = image[i:i+self.patch_size, j:j+self.patch_size].flatten()
                X = np.arange(len(patch)).reshape(-1, 1)
                
                best_model, inliers, residuals = self.run_ransac(X, patch)
                denoised_patch = np.polyval(best_model, X).reshape(self.patch_size, self.patch_size)
                latent_representation[i:i+self.patch_size, j:j+self.patch_size] = denoised_patch
                noise_estimate[i:i+self.patch_size, j:j+self.patch_size] = residuals.reshape(self.patch_size, self.patch_size)
        
        return latent_representation.astype(np.uint8), noise_estimate
    
    def run_ransac(self, X, y):
        best_model = None
        best_inliers = []
        best_score = 0
        best_residuals = np.zeros_like(y)
        
        for _ in range(self.max_iterations):
            sample_indices = random.sample(range(len(y)), self.min_sample_size)
            sample_X, sample_y = X[sample_indices], y[sample_indices]
            model = np.polyfit(sample_X.flatten(), sample_y, 1)
            residuals = np.abs(y - np.polyval(model, X).flatten())
            inliers = y[residuals < self.distance_threshold]
            
            if len(inliers) > best_score:
                best_model = model
                best_inliers = inliers
                best_score = len(inliers)
                best_residuals = residuals
        
        return best_model, best_inliers, best_residuals