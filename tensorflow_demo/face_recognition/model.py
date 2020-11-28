import numpy as np
import os

import tensorflow as tf
from tensorflow.keras import Model, layers


def preprocess(x, y):
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3)

    x = tf.cast(x, tf.float32) / 255.0

    y = tf.convert_to_tensor(y)

    return x, y


def my_face():
    path = os.listdir("./my_faces")
    image_path = [os.path.join("./my_faces/", img) for img in path]

    return image_path


def other_face():
    path = os.listdir("./other_faces")
    image_path = [os.path.join("./other_faces/", img) for img in path]

    return image_path


class CNN_WORK(Model):
    def __init__(self):
        super(CNN_WORK, self).__init__()
        self.conv1 = layers.Conv2D(32, kernel_size=5, activation=tf.nn.relu)
        self.maxpool1 = layers.MaxPool2D(2, strides=2)

        self.conv2 = layers.Conv2D(64, kernel_size=3, activation=tf.nn.relu)
        self.maxpool2 = layers.MaxPool2D(2, strides=2)

        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(1024)
        self.dropout = layers.Dropout(rate=0.5)
        self.out = layers.Dense(2)

    def call(self, x, is_training=False):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x, training=is_training)
        x = self.out(x)

        if not is_training:
            x = tf.nn.softmax(x)
        return x


model = CNN_WORK()


def cross_entropy_loss(x, y):
    y = tf.cast(y, tf.int64)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
    return tf.reduce_mean(loss)


def accuracy(y_pred, y_true):
    correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32), axis=-1)


optimizer = tf.optimizers.SGD(0.002)


def run_optimizer(x, y):
    with tf.GradientTape() as g:
        pred = model(x, is_training=True)
        loss = cross_entropy_loss(pred, y)
    training_variabel = model.trainable_variables
    gradient = g.gradient(loss, training_variabel)
    optimizer.apply_gradients(zip(gradient, training_variabel))


def main():
    image_path = my_face().__add__(other_face())
    label_my = [1 for i in my_face()]
    label_other = [0 for i in other_face()]
    label = label_my.__add__(label_other)
    data = tf.data.Dataset.from_tensor_slices((image_path, label))
    data_loader = data.repeat().shuffle(5000).map(preprocess).batch(128).prefetch(1)
    for i in range(2):
        for step, (batch_x, batch_y) in enumerate(data_loader.take(128), 1):

            run_optimizer(batch_x, batch_y)
            if step % 20 == 0:
                pred = model(batch_x, is_training=False)
                loss = cross_entropy_loss(pred, batch_y)
                acc = accuracy(pred, batch_y)
                print(batch_y)
                a = tf.argmax(pred, 1)
                print(a)
                print("loss:%f; acc:%f" % (loss, acc))


main()
try:
    model.save_weights("face_weight")
except:
    pass
try:
    model.save("my_face.h5")
except:
    pass