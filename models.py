import tensorflow as tf
import numpy as np

class CustomClassifier(tf.keras.Model):
    def __init__(self, num_classes: int = 2, apply_augmentation: bool = False):
        super().__init__()

        self.apply_augmentation = apply_augmentation
        if apply_augmentation:
            self.aug = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.2),
                #tf.keras.layers.RandomBrightness(0.2),
                tf.keras.layers.RandomContrast(0.2),
            ])

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
    
    def call(self, x: tf.Tensor, training: bool = None) -> tf.Tensor:
        if self.apply_augmentation:
            x = self.aug(x, training=training)
        x = self.cnn(x)
        return x

class ResnetClassifier(tf.keras.Model):
    def __init__(self, num_classes: int = 2, apply_augmentation: bool = False):
        super().__init__()

        self.apply_augmentation = apply_augmentation
        if apply_augmentation:
            self.aug = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.2),
                tf.keras.layers.RandomContrast(0.2),
            ])
        
        net = tf.keras.applications.resnet50.ResNet50(
                include_top=False,
                weights='imagenet',
            ) 
        self.resnet =  tf.keras.Sequential([
            net,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1000, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')

        ])
    def call(self, x: tf.Tensor, training: bool = None) -> tf.Tensor:
        if self.apply_augmentation:
            x = self.aug(x, training = training)
        x = self.resnet(x)
        return x