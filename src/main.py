import cv2
import numpy as np
import matplotlib.pyplot as plt
from sampler import *
from encoder import *

def add_noise(image, noise_level=25):
    """Adds Gaussian noise to an image."""
    noise = np.random.normal(0, noise_level, image.shape)
    noisy_image = np.clip(image + noise, 0, 255)
    return noisy_image.astype(np.uint8)

def pseudo_autoencoder(image, encoder, sampler):
    denoised_image = encoder.encode(noisy_image)
    final_image = sampler.sample(denoised_image)
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image, cmap='gray'); axes[0].set_title("Original Image")
    axes[1].imshow(noisy_image, cmap='gray'); axes[1].set_title("Noisy Image")
    axes[2].imshow(final_image, cmap='gray'); axes[2].set_title("Denoised Image")
    plt.show()

if __name__ == "__main__":
    image = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)  # Load grayscale image
    noisy_image = add_noise(image)

    encoder = RANSACEncoder()
    sampler = MCMCSampler()
    pseudo_autoencoder(image, encoder, sampler)
