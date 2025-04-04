from utils import *

if __name__ == "__main__":
    original_image = load_image("data/grey12_2/france.tif")
    # original_image = add_noise(image)

    encoder = RANSACEncoder()
    sampler = SimpleSampler()

    pseudo_autoencoder(original_image, encoder, sampler)

    latent_representation, sampled_image = pseudo_autoencoder(original_image, encoder, sampler)

    plot_images(original_image, latent_representation, sampled_image)
