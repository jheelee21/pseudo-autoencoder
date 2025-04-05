import numpy as np

class Sampler:
    def __init__(self):
        pass

    def sample(self, image, outliers_mask):
        pass

class MeanSampler(Sampler):
    def __init__(self, patch_size=3):
        self.patch_size = patch_size

    def sample(self, image, outliers_mask):
        h, w = image.shape
        sampled_image = image.copy()

        for x, y in zip(*np.where(outliers_mask == 1)):
            patch = sampled_image[max(x-self.patch_size, 0):min(h, x+self.patch_size+1),
                                    max(y-self.patch_size, 0):min(w, y+self.patch_size+1)]
            sampled_image[x, y] = np.mean(patch)
        
        return sampled_image.astype(np.uint8)

class MCMCSampler(Sampler):
    def __init__(self, iterations=100, sigma=5, acceptance_threshold=20):
        self.iterations = iterations
        self.sigma = sigma
        self.acceptance_threshold = acceptance_threshold

    def sample(self, image, outliers_mask):
        h, w = image.shape
        sampled_image = image.copy()

        for x, y in zip(*np.where(outliers_mask == 1)):
            current_value = sampled_image[x, y]
            for _ in range(self.iterations):
                proposed_value = np.random.normal(current_value, self.sigma)

                if (0 <= proposed_value <= 255 and 
                    abs(proposed_value - current_value) < self.acceptance_threshold):
                    current_value = proposed_value
            sampled_image[x, y] = current_value

        return sampled_image.astype(np.uint8)


class ImportanceSampler(Sampler):
    def __init__(self, iterations=10, k=4):
        self.iterations = iterations
        self.k = k

    def sample(self, image, outliers_mask):
        h, w = image.shape
        sampled_image = image.copy()

        for _ in range(self.iterations):
            for x, y in zip(*np.where(outliers_mask == 1)): 
                patch = sampled_image[max(x-self.k, 0):min(h, x+self.k+1),
                                      max(y-self.k, 0):min(w, y+self.k+1)]

                patch_weights = patch / np.sum(patch, dtype=np.float32)
                patch_flat = patch.flatten()
                patch_weights_flat = patch_weights.flatten()

                sampled_value = np.random.choice(patch_flat, p=patch_weights_flat)
                sampled_image[x, y] = sampled_value

        return sampled_image.astype(np.uint8)