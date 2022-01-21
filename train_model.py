import os
import tensorflow as tf
from gan_model import GAN, ModelType
from data_utils import make_mnist_dataset, generate_and_save_images


NUM_EPOCHS = 50
NUM_ITERATION = 5000
BATCH_SIZE = 64
NOISE_DIMENTION = 100
L1_PENALTY = 10
L2_PENALTY = 10     # WGAN 10 ; GAN 0.01
GENERATOR_LEARNING_RATE = 5e-5
DISCRIMINATOR_LEARNING_RATE = 5e-5
NUM_DISCRIMINATOR_STEPS = 5
MODEL_TYPE = ModelType.WGAN  # ModelType.WGAN ; ModelType.GAN
LOG_FOLDER = "logs"

master_test_seed = tf.random.normal([16, NOISE_DIMENTION])
real_dataset = make_mnist_dataset(BATCH_SIZE)

gan = GAN(GENERATOR_LEARNING_RATE, DISCRIMINATOR_LEARNING_RATE, MODEL_TYPE, L1_PENALTY, L2_PENALTY)
### warmup
gan.run_discriminator(gan.run_generator(master_test_seed))
###


@tf.function
def wasserstein_train_step(dataset_iterator):
    for step in range(NUM_DISCRIMINATOR_STEPS):
        images = next(dataset_iterator)
        noise = tf.random.normal([BATCH_SIZE, NOISE_DIMENTION])
        epsilon = tf.random.uniform([BATCH_SIZE, 1, 1, 1], 0.0, 1.0)
        disc_loss = gan.discriminator_train_step(noise, images, epsilon)

    noise = tf.random.normal([BATCH_SIZE, NOISE_DIMENTION])
    gen_loss = gan.generator_train_step(noise)

    return gen_loss, disc_loss


@tf.function
def gan_train_step(dataset_iterator):
    images = next(dataset_iterator)
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIMENTION])
    disc_loss = gan.discriminator_train_step(noise, images)

    noise = tf.random.normal([BATCH_SIZE, NOISE_DIMENTION])
    gen_loss = gan.generator_train_step(noise)

    return gen_loss, disc_loss


if MODEL_TYPE is ModelType.GAN:
    train_step = gan_train_step
elif MODEL_TYPE is ModelType.WGAN:
    train_step = wasserstein_train_step
else:
    raise NotImplementedError("model type is not implemented")

log_folder = os.path.join(LOG_FOLDER, MODEL_TYPE.value)
gen_loss = tf.keras.metrics.Mean()
disc_loss = tf.keras.metrics.Mean()
for epoch in range(NUM_EPOCHS):
    gen_loss.reset_state()
    disc_loss.reset_state()

    iterator = iter(real_dataset)
    for it in range(NUM_ITERATION):
        gl, dl = train_step(iterator)
        gen_loss(gl)
        disc_loss(dl)

    generate_and_save_images(gan.generator_model, epoch, master_test_seed, log_folder)

    print(f"EPOCH {epoch} | Loss generator: {gen_loss.result()} | Loss discriminator: {disc_loss.result()}")
