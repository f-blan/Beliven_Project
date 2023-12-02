

import tensorflow as tf
import tensorflow_datasets as tfds

class CustomClassifier(tf.keras.Model):
    def __init__(self, num_classes: int = 2):
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
            tf.keras.layers.Dense(num_classes)
        ])
    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.cnn(x)
        return x

class ResnetClassifier(tf.keras.Model):
    def __init__(self, num_classes:int = 2):
        super().__init__()

        self.resnet = tf.keras.applications.resnet50.ResNet50(
          include_top=False,
          weights='imagenet',
          input_tensor=None,
          input_shape=None,
          pooling=None,
          classes=2,
        )
    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.resnet(x)
        return x