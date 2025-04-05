from utils import *

if __name__ == "__main__":
    original_image = load_image("data/grey12_1/bridge.tif")
    # original_image = add_noise(original_image)

    # encoder = RANSACEncoder(patch_size=4, degree=3)
    # encoder = MeanEncoder()
    # sampler = ImportanceSampler(iterations=2, k=2)
    # sampler = MCMCSampler()

    encoder = SVDEncoder(k=20, patch_size=8, threshold=200)
    # sampler = ImportanceSampler(iterations=2, k=2)

    sampler = MeanSampler(patch_size=4)


    latent_representation, sampled_image = pseudo_autoencoder(original_image, encoder, sampler)

    plot_images(original_image, latent_representation, sampled_image)