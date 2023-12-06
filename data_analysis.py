import tensorflow as tf
import tensorflow_datasets as tfds
import PIL
import PIL.Image
import matplotlib.pyplot as plt
import pathlib
from argparse import Namespace
import os
import numpy as np

def data_analysis(args: Namespace):
  train_dir = pathlib.Path(args.train_dir)

  #check classes distribution
  dog_count = len(list(train_dir.glob('dogs/*.jpg')))
  cat_count = len(list(train_dir.glob('cats/*.jpg')))

  print("number of dog images: ", dog_count)
  print("number of cat images: ", cat_count)


  #check size distribution
  widths = []
  heights = []

  for img_path in list(train_dir.glob('*/*.jpg')):
    img = PIL.Image.open(img_path)  
    wid, hgt = img.size 
    widths.append(wid)
    heights.append(hgt)

  plt.scatter(widths, heights) 
  plt.xlabel("width")
  plt.ylabel("height")
  plt.title("Image sizes")
  figpath = os.path.join(".", "figs", "scatter.png")
  plt.savefig(figpath, format= "png")
  plt.close()


  #visualize some of the pictures
  train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(args.crop_size, args.crop_size),
    batch_size=1)

  class_names = train_ds.class_names

  #show cat images
  cats_ds = train_ds.filter(lambda imgs, labels: tf.math.equal(labels[0], 0))
  i=0
  
  for images, labels in cats_ds.take(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[0].numpy().astype("uint8"))
    plt.axis("off")
    i+=1
  figpath = os.path.join(".", "figs", "cats.png")
  plt.savefig(figpath, format= "png")
  plt.close()

  #show dog images
  dogs_ds = train_ds.filter(lambda imgs, labels: tf.math.equal(labels[0], 1))
  i=0
  for images, labels in dogs_ds.take(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[0].numpy().astype("uint8"))
    #plt.title(class_names[labels[0]])
    plt.axis("off")
    i+=1
  figpath = os.path.join(".", "figs", "dogs.png")
  plt.savefig(figpath, format= "png")
  plt.close()

  #show augmented images
  aug = tf.keras.Sequential([
          tf.keras.layers.RandomFlip("horizontal"),
          tf.keras.layers.GaussianNoise(0.1),
          tf.keras.layers.RandomBrightness(0.2),
          tf.keras.layers.RandomContrast(0.2)
        ])

  aug_ds = dogs_ds.map(lambda x, y: (aug(x, training=True), y))
  i=0
  for images, labels in aug_ds.take(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[0].numpy().astype("uint8"))
    #plt.title(class_names[labels[0]])
    plt.axis("off")
    i+=1
  figpath = os.path.join(".", "figs", "aug_dogs.png")
  plt.savefig(figpath, format= "png")
  plt.close()

  test_dir = pathlib.Path(args.test_dir)
  test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        seed=123,
        image_size=(args.crop_size, args.crop_size),
        batch_size=args.batch_size,
        label_mode='categorical',)