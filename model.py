import tensorflow as tf

def make_generator_model():
    model = tf.keras.Sequential([
        # Start with a fully connected layer
        tf.keras.layers.Dense(8*8*256, use_bias=False, input_shape=(100,)),  # Adjusted for an 8x8 start size
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        # Reshape into an 8x8 256-channel feature map
        tf.keras.layers.Reshape((8, 8, 256)),

        # Transposed convolution layer, from 8x8x256 into 16x16x128
        tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        # Transposed convolution layer, from 16x16x128 to 32x32x64
        tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        # Transposed convolution layer, from 32x32x64 to 64x64x32
        tf.keras.layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        # Transposed convolution layer, from 64x64x32 to 128x128x3 (final image dimension)
        tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model

def make_discriminator_model():
    model = tf.keras.Sequential([
        # Convolutional layer, from image dimension 128x128x3 to 64x64x64
        tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[128, 128, 3]),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.3),

        # Another convolution layer, from 64x64x64 to 32x32x128
        tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.3),

        # Flatten the output layer to 1D, followed by a dense layer for classification
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1)
    ])
    return model
