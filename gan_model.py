import tensorflow as tf
from enum import Enum


class ModelType(Enum):
    GAN = "gan"
    WGAN = "wgan"


class GAN(object):
    def __init__(self, generator_lr, discriminator_lr, model_type: ModelType = ModelType.GAN,
                 discriminator_l1: float = 10, discriminator_l2: float = 0.1):
        if model_type is ModelType.GAN:
            self.discriminator_loss = self._discriminator_gan_loss
            self.generator_loss = self._generator_gan_loss
            self._cross_entropy_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        elif model_type is ModelType.WGAN:
            self.discriminator_loss = self._discriminator_wasserstein_loss
            self.generator_loss = self._generator_wasserstein_loss
        else:
            raise NotImplementedError("model type is not implemented")

        self.generator_model = tf.keras.Sequential([
            tf.keras.layers.Dense(7 * 7 * 256, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Reshape((7, 7, 256)),

            tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(1, 1), padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(1, 1), padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', use_bias=False,
                                            activation='tanh'),
        ])

        self.discriminator_model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Conv2D(1, (3, 3), strides=(2, 2), padding='same'),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1)
        ])

        self.generator_optimizer = tf.keras.optimizers.Adam(generator_lr)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(discriminator_lr)

        self.discriminator_l1 = discriminator_l1
        self.discriminator_l2 = discriminator_l2

    def run_generator(self, input, training=None):
        return self.generator_model(input, training=training)

    def run_discriminator(self, input, training=None):
        return self.discriminator_model(input, training=training)

    def generator_train_step(self, noise):
        with tf.GradientTape() as gen_tape:
            gen_loss = self.generator_loss(noise, training=True)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator_model.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator_model.trainable_variables))
        return gen_loss

    def _generator_wasserstein_loss(self, noise, training=None):
        generated_images = self.generator_model(noise, training=training)
        fake_output = self.discriminator_model(generated_images, training=training)
        return -tf.math.reduce_mean(fake_output)

    def _generator_gan_loss(self, noise, training=None):
        generated_images = self.generator_model(noise, training=training)
        fake_output = self.discriminator_model(generated_images, training=training)
        return self._cross_entropy_loss(tf.ones_like(fake_output), fake_output)

    def _discriminator_wasserstein_loss(self, noise, images, epsilon, training=None):
        generated_images = self.generator_model(noise, training=training)

        real_output = self.discriminator_model(images, training=training)
        fake_output = self.discriminator_model(generated_images, training=training)

        mixed_images = generated_images + epsilon * (images - generated_images)
        with tf.GradientTape() as tape:
            tape.watch(mixed_images)
            mixed_scores = self.discriminator_model(mixed_images, training=training)

        grad = tape.gradient(mixed_scores, mixed_images)
        grad = tf.reshape(grad, [-1, 28 * 28, 1])
        grad_norm = tf.norm(grad, axis=[1, 2])
        l1_penalty = tf.reduce_mean((grad_norm - 1) ** 2)
        l2_penalty = tf.reduce_mean([tf.reduce_sum(w) ** 2 for w in self.discriminator_model.trainable_variables])
        disc_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
        disc_loss += self.discriminator_l1 * l1_penalty + self.discriminator_l2 * l2_penalty
        return disc_loss

    def _discriminator_gan_loss(self, noise, images, _, training=None):
        generated_images = self.generator_model(noise, training=training)

        real_output = self.discriminator_model(images, training=training)
        fake_output = self.discriminator_model(generated_images, training=training)

        real_loss = self._cross_entropy_loss(tf.ones_like(real_output), real_output)
        fake_loss = self._cross_entropy_loss(tf.zeros_like(fake_output), fake_output)
        l2_penalty = tf.reduce_mean([tf.reduce_sum(w) ** 2 for w in self.discriminator_model.trainable_variables])
        disc_loss = real_loss + fake_loss + self.discriminator_l2 * l2_penalty
        return disc_loss

    def discriminator_train_step(self, noise, images, epsilon=None):
        with tf.GradientTape() as disc_tape:
            disc_loss = self.discriminator_loss(noise, images, epsilon, training=True)

        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator_model.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator_model.trainable_variables))
        return disc_loss

