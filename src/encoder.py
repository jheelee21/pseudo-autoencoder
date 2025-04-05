import numpy as np
import random
import cv2

class Encoder:
    def __init__(self):
        pass
    
    def encode(self, image):
        pass

class MeanEncoder(Encoder):
    def __init__(self, patch_size=8, threshold=10):
        self.patch_size = patch_size
        self.threshold = threshold
    
    def encode(self, image):
        h, w = image.shape
        latent_representation = np.zeros_like(image)
        outliers_mask = np.zeros_like(image, dtype=np.uint8)
        
        for i in range(0, h, self.patch_size):
            for j in range(0, w, self.patch_size):
                patch = image[i:i+self.patch_size, j:j+self.patch_size]
                patch_mean = np.mean(patch)
                
                latent_representation[i:i+self.patch_size, j:j+self.patch_size] = patch_mean
                
                residuals = np.abs(patch - patch_mean)
                outlier_patch = (residuals > self.threshold).astype(np.uint8)
                outliers_mask[i:i+self.patch_size, j:j+self.patch_size] = outlier_patch
        
        return latent_representation.astype(np.uint8), outliers_mask

class RANSACEncoder(Encoder):
    def __init__(self, patch_size=8, threshold=10, degree=2, max_iterations=50, min_sample_size=5):
        self.patch_size = patch_size
        self.threshold = threshold
        self.degree = degree  # Degree of the polynomial fit
        self.max_iterations = max_iterations
        self.min_sample_size = min_sample_size
    
    def encode(self, image):
        h, w = image.shape
        latent_representation = np.zeros_like(image)
        noise_estimate = np.zeros_like(image, dtype=np.float32)
        outliers_mask = np.zeros_like(image, dtype=np.uint8)
        
        for i in range(0, h, self.patch_size):
            for j in range(0, w, self.patch_size):
                patch = image[i:i+self.patch_size, j:j+self.patch_size].flatten()
                X = np.arange(len(patch)).reshape(-1, 1)
                
                best_model, inliers, residuals = self.run_ransac(X, patch)
                denoised_patch = np.polyval(best_model, X).reshape(self.patch_size, self.patch_size)
                latent_representation[i:i+self.patch_size, j:j+self.patch_size] = denoised_patch
                noise_estimate[i:i+self.patch_size, j:j+self.patch_size] = residuals.reshape(self.patch_size, self.patch_size)

                residuals_patch = residuals.reshape(self.patch_size, self.patch_size)
                outlier_patch = (residuals_patch > self.threshold).astype(np.uint8)
                outliers_mask[i:i+self.patch_size, j:j+self.patch_size] = outlier_patch
         
        return latent_representation.astype(np.uint8), outliers_mask
    
    def run_ransac(self, X, y):
        best_model = None
        best_inliers = []
        best_score = 0
        best_residuals = np.zeros_like(y)
        
        for _ in range(self.max_iterations):
            sample_indices = random.sample(range(len(y)), self.min_sample_size)
            sample_X, sample_y = X[sample_indices], y[sample_indices]
            model = np.polyfit(sample_X.flatten(), sample_y, self.degree)
            residuals = np.abs(y - np.polyval(model, X).flatten())
            inliers = y[residuals < self.threshold]
            
            if len(inliers) > best_score:
                best_model = model
                best_inliers = inliers
                best_score = len(inliers)
                best_residuals = residuals
        
        return best_model, best_inliers, best_residuals


class SVDEncoder(Encoder):
    def __init__(self, patch_size=8, k=10, threshold=10):
        self.patch_size = patch_size
        self.k = k  # Number of singular values to keep
        self.threshold = threshold
    
    def encode(self, image):
        U, S, Vt = np.linalg.svd(image, full_matrices=False)
        S[self.k:] = 0
        latent_representation = (U @ np.diag(S) @ Vt).clip(0, 255).astype(np.uint8)
        noise_estimate = np.abs(image - latent_representation)
        outliers_mask = (noise_estimate > self.threshold).astype(np.uint8)

        print(f"Number of outliers: {np.sum(outliers_mask)}")
        return latent_representation, outliers_mask