import numpy as np

class Sampler:
    def __init__(self) -> None:
        pass

    def sample(self, image):
        pass

class MCMCSampler(Sampler):
    def __init__(self, iterations=500, sigma=5):
        super().__init__()
        self.iterations = iterations
        self.sigma = sigma
    
    def sample(self, denoised_image):
        h, w = denoised_image.shape
        sampled_image = denoised_image.copy()
        
        for _ in range(self.iterations):
            x, y = np.random.randint(0, h), np.random.randint(0, w)
            proposed_value = np.random.normal(sampled_image[x, y], self.sigma)
            
            if 0 <= proposed_value <= 255:
                sampled_image[x, y] = proposed_value
        
        return sampled_image.astype(np.uint8)

class ImportanceSampler(Sampler):
    def __init__(self, num_samples=1000):
        super().__init__()
        self.num_samples = num_samples
    
    def sample(self, denoised_image):
        h, w = denoised_image.shape
        sampled_image = np.zeros_like(denoised_image, dtype=np.float32)
        
        weights = np.exp(-denoised_image / 255.0)
        weights /= np.sum(weights)
        
        for _ in range(self.num_samples):
            indices = np.random.choice(h * w, p=weights.flatten())
            x, y = divmod(indices, w)
            sampled_image[x, y] += 1
        
        sampled_image = (sampled_image / np.max(sampled_image)) * 255
        return sampled_image.astype(np.uint8)