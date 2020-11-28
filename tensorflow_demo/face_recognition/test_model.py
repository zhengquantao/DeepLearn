import tensorflow as tf
from tensorflow.keras import Model, layers


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
        # x = tf.reshape(x,[-1,64,64,3])

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
model.load_weights("face_weight")
x = "./other_faces/.jpg"
x = tf.io.read_file(x)
x = tf.image.decode_jpeg(x, channels=3)

x = tf.cast(x, tf.float32) / 255.0
x = tf.reshape(x, [-1, 64, 64, 3])
y = tf.convert_to_tensor([1])

a = model(x)

print(tf.argmax(a, axis=1).numpy())