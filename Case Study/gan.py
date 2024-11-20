import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    BatchNormalization,
    LeakyReLU,
    Reshape,
    Conv2DTranspose,
    Conv2D,
    Flatten,
    Dropout,
)
import matplotlib.pyplot as plt

# Load and preprocess the MNIST dataset
(train_images, _), _ = tf.keras.datasets.mnist.load_data()
train_images = (train_images.reshape(-1, 28, 28, 1) - 127.5) / 127.5
train_dataset = (
    tf.data.Dataset.from_tensor_slices(train_images).shuffle(60000).batch(512)
)  # Increased batch size


# Define the Generator
def generator_model():
    model = tf.keras.Sequential(
        [
            Dense(
                7 * 7 * 512, use_bias=False, input_shape=(100,)
            ),  # Increased capacity
            BatchNormalization(),
            LeakyReLU(),
            Reshape((7, 7, 512)),
            Conv2DTranspose(
                256, (5, 5), strides=(1, 1), padding="same", use_bias=False
            ),
            BatchNormalization(),
            LeakyReLU(),
            Conv2DTranspose(
                128, (5, 5), strides=(2, 2), padding="same", use_bias=False
            ),
            BatchNormalization(),
            LeakyReLU(),
            Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", use_bias=False),
            BatchNormalization(),
            LeakyReLU(),
            Conv2DTranspose(
                1,
                (5, 5),
                strides=(1, 1),
                padding="same",
                use_bias=False,
                activation="tanh",
            ),
        ]
    )
    return model


# Define the Discriminator
def discriminator_model():
    model = tf.keras.Sequential(
        [
            Conv2D(64, (5, 5), strides=(2, 2), padding="same", input_shape=(28, 28, 1)),
            LeakyReLU(),
            Dropout(0.3),
            Conv2D(128, (5, 5), strides=(2, 2), padding="same"),
            LeakyReLU(),
            Dropout(0.3),
            Flatten(),
            Dense(1),
        ]
    )
    return model


# Losses and optimizers
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_loss = lambda fake_output: cross_entropy(
    tf.ones_like(fake_output), fake_output
)
discriminator_loss = lambda real_output, fake_output: cross_entropy(
    tf.ones_like(real_output), real_output
) + cross_entropy(tf.zeros_like(fake_output), fake_output)

generator = generator_model()
discriminator = discriminator_model()
gen_opt = tf.keras.optimizers.Adam(1e-4)
disc_opt = tf.keras.optimizers.Adam(1e-4)


# Training step
@tf.function
def train_step(images):
    noise = tf.random.normal([512, 100])  # Match batch size
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(fake_images, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gen_grad = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_grad = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    gen_opt.apply_gradients(zip(gen_grad, generator.trainable_variables))
    disc_opt.apply_gradients(zip(disc_grad, discriminator.trainable_variables))


# Generate and display sample images
def generate_images(generator, test_input):
    predictions = generator(test_input, training=False)
    plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap="gray")
        plt.axis("off")
    plt.show()


# Train function
def train(dataset, epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            train_step(image_batch)
        print(f"Epoch {epoch+1}/{epochs} completed")
        generate_images(generator, seed)  # Display images after each epoch


# Training the GAN
noise_dim = 100
seed = tf.random.normal([16, noise_dim])  # For consistent generated samples
train(train_dataset, 20)  # Increased epochs to 50
generate_images(generator, seed)
