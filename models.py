

import tensorflow as tf
import tensorflow_datasets as tfds

class CustomClassifier(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        self.cnn = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.cnn(x)
        return x
