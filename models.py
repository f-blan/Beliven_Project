import tensorflow as tf
import numpy as np
from typing import Tuple

class CustomClassifier(tf.keras.Model):
    def __init__(self, num_classes: int = 2, apply_augmentation: bool = False, crop_size:int=256):
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
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape= (crop_size, crop_size,3)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation="softmax")
        ])
    
    def call(self, x: tf.Tensor, training: bool = None) -> tf.Tensor:
        if self.apply_augmentation:
            x = self.aug(x, training=training)
        x = self.cnn(x)
        return x

def make_resnet(input_shape:Tuple[int,int,int]=(256,256,3), num_classes:int =2):
  base_model = tf.keras.applications.resnet50.ResNet50(
                include_top=False,
                weights='imagenet',
                input_shape=input_shape
            )
  
  for layer in base_model.layers:
    layer.trainable = False

  x = tf.keras.layers.Flatten()(base_model.output)
  x = tf.keras.layers.Dense(1000, activation='relu')(x)
  predictions = tf.keras.layers.Dense(num_classes, activation = 'softmax')(x)

  model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
  return model

class ResnetClassifier(tf.keras.Model):
    def __init__(self, num_classes: int = 2, apply_augmentation: bool = False, crop_size:int=256):
        super().__init__()

        self.apply_augmentation = apply_augmentation
        if apply_augmentation:
            self.aug = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.2),
                tf.keras.layers.RandomContrast(0.2),
            ])

        self.resnet = make_resnet(input_shape=(crop_size,crop_size,3))
    def call(self, x: tf.Tensor, training: bool = None) -> tf.Tensor:
        if self.apply_augmentation:
            x = self.aug(x, training = training)
        x = self.resnet(x)
        return x

def make_vgg(input_shape=(256,256,3), num_classes =2):
    base_model = tf.keras.applications.VGG16(input_shape=input_shape , weights='imagenet',include_top=False)
    for layer in base_model.layers:
        layer.trainable = False

    x = tf.keras.layers.Flatten()(base_model.output)
    predictions = tf.keras.layers.Dense(num_classes, activation = 'softmax')(x)

    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
    return model

class VGGClassifier(tf.keras.Model):
    def __init__(self, num_classes: int = 2, apply_augmentation: bool = False, crop_size:int=256):
        super().__init__()

        self.apply_augmentation = apply_augmentation
        if apply_augmentation:
            self.aug = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.2),
                tf.keras.layers.RandomContrast(0.2),
            ])
        self.vgg = make_vgg(input_shape=(crop_size, crop_size, 3), num_classes=num_classes)
    
    def call(self, x: tf.Tensor, training: bool = None) -> tf.Tensor:
        if self.apply_augmentation:
            x = self.aug(x, training = training)
        x = self.vgg(x)
        return x