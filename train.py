import tensorflow as tf
import matplotlib.pyplot as plt
from dataset import get_datasets
from model import make_generator_model, make_discriminator_model

def discriminator_loss(real_output, fake_output):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_output), real_output)
    fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_output), fake_output)

def clear_gpu_memory():
    """Clears GPU memory to avoid out-of-memory issues (optional use)"""
    tf.keras.backend.clear_session()
    print("GPU memory cleared")

def train_step(images, generator, discriminator, generator_optimizer, discriminator_optimizer, batch_size):
    noise = tf.random.normal([batch_size, 100])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss.numpy(), disc_loss.numpy()

def validate_discriminator(validation_dataset, generator, discriminator, batch_size):
    validation_loss = []
    for image_batch in validation_dataset:
        noise = tf.random.normal([batch_size, 100])
        generated_images = generator(noise, training=False)
        real_output = discriminator(image_batch, training=False)
        fake_output = discriminator(generated_images, training=False)
        v_loss = discriminator_loss(real_output, fake_output)
        validation_loss.append(v_loss.numpy())
    return tf.reduce_mean(validation_loss)

def train_gan(train_dataset, validation_dataset, test_dataset, epochs, batch_size):
    generator = make_generator_model()
    discriminator = make_discriminator_model()

    # Modified: Adjusting learning rates and beta1 for Adam optimizer
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

    train_disc_losses = []
    validation_disc_losses = []

    for epoch in range(epochs):
        print(f"Starting Epoch {epoch+1}/{epochs}")
        batch_losses = []
        for batch_index, image_batch in enumerate(train_dataset):
            gen_loss, disc_loss = train_step(image_batch, generator, discriminator, generator_optimizer, discriminator_optimizer, batch_size)
            batch_losses.append(disc_loss)
            # Modified: Added detailed print statement for each batch to monitor losses more closely
            print(f"Epoch {epoch+1}, Batch {batch_index+1}, Generator Loss: {gen_loss}, Discriminator Loss: {disc_loss}")

        avg_epoch_loss = tf.reduce_mean(batch_losses)
        train_disc_losses.append(avg_epoch_loss)

        v_loss = validate_discriminator(validation_dataset, generator, discriminator, batch_size)
        validation_disc_losses.append(v_loss)
        print(f'Epoch {epoch + 1}, Validation Loss: {v_loss}')

        # Optional: Clear GPU memory after each epoch
        clear_gpu_memory()

    # Save models at the end of training
    generator.save('models/generator_final.h5')
    discriminator.save('models/discriminator_final.h5')

    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_disc_losses, label='Training Discriminator Loss')
    plt.plot(validation_disc_losses, label='Validation Discriminator Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_ds, validation_ds, test_ds = get_datasets()
    train_gan(train_ds, validation_ds, test_ds, 5, 50)  # Set epochs and batch size
