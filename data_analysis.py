import tensorflow as tf
import tensorflow_datasets as tfds
import PIL
import PIL.Image
import os
from argument_parser import parse_arguments
import sys
import matplotlib.pyplot as plt
import pathlib

args = parse_arguments(sys.argv[1:])
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

plt.show()


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
plt.show()

#show dog images
dogs_ds = train_ds.filter(lambda imgs, labels: tf.math.equal(labels[0], 1))
i=0
for images, labels in dogs_ds.take(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(images[0].numpy().astype("uint8"))
  #plt.title(class_names[labels[0]])
  plt.axis("off")
  i+=1
plt.show()