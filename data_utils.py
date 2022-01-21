import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as pyplot


def make_mnist_dataset(batch_size):
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype(np.float32)
    train_images = train_images / 127.5 - 1.  # Normalize to [-1, 1]

    real_dataset = tf.data.Dataset.from_tensor_slices(train_images) \
        .shuffle(train_images.shape[0]) \
        .batch(batch_size, drop_remainder=True) \
        .repeat() \
        .prefetch(2)
    return real_dataset


def generate_and_save_images(model, epoch, test_input, log_folder):
    os.makedirs(log_folder, exist_ok=True)

    predictions = model(test_input, training=False)

    fig = pyplot.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        pyplot.subplot(4, 4, i+1)
        pyplot.imshow((predictions[i, :, :, 0] + 1) * 127.5, cmap='gray')
        pyplot.axis('off')

    impath = os.path.join(log_folder,"image_{:04d}.png".format(epoch))
    pyplot.savefig(impath)
    pyplot.close(fig)
