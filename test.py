import tensorflow as tf
from matplotlib import pyplot as plt
from model import make_generator_model  # Assuming this method initializes the model architecture correctly

def load_and_generate_images(generator_path, num_images=5, noise_dim=100):
    # Load the generator model
    generator = tf.keras.models.load_model(generator_path)
    
    # Generate random noise to feed into the generator
    random_noise = tf.random.normal([num_images, noise_dim])

    # Use the generator to create images from random noise
    generated_images = generator(random_noise, training=False)

    # Plotting generated images
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(5, 5, i + 1)
        plt.imshow(generated_images[i, :, :, :] * 0.5 + 0.5)  # Adjusting normalization
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    generator_model_path = 'models/generator_final.h5'
    load_and_generate_images(generator_model_path)
