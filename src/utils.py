import cv2
import numpy as np
import matplotlib.pyplot as plt
from sampler import *
from encoder import *

def load_image(image_path, colour = 'b/w'):
    if colour == 'b/w':
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(image_path)
    np_image = np.array(image)
    return np_image

def show_image(image, colour = 'b/w'):
    if colour == 'b/w':
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.axis('off')
    plt.show()

def plot_images(original, denoised, sampled):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original, cmap='gray'); axes[0].set_title("Original Image")
    axes[1].imshow(denoised, cmap='gray'); axes[1].set_title("Encoded Image")
    axes[2].imshow(sampled, cmap='gray'); axes[2].set_title("Sampled Image")
    plt.show()

def add_noise(image, noise_level=25):
    noise = np.random.normal(0, noise_level, image.shape)
    noisy_image = np.clip(image + noise, 0, 255)
    return noisy_image.astype(np.uint8)

def pseudo_autoencoder(noisy_image, encoder, sampler):
    latent_representation, noise_estimate = encoder.encode(noisy_image)
    sampled_image = sampler.sample(latent_representation, noise_estimate)
    return latent_representation, sampled_image