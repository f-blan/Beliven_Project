from argparse import Namespace
import tensorflow as tf
import tensorflow_datasets as tfds
import PIL
import PIL.Image
import pathlib
from models import CustomClassifier


def train(args: Namespace):
    
    train_dir = pathlib.Path(args.train_dir)

    # PREPARE THE DATASETS
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(args.crop_size, args.crop_size),
        batch_size=args.batch_size)
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(args.crop_size, args.crop_size),
        batch_size=args.batch_size)
    
    rescaling_norm = tf.keras.layers.Rescaling(1./255)

    #normalize in range [0,1]
    train_ds = train_ds.map(lambda x, y: (rescaling_norm(x), y))
    val_ds = val_ds.map(lambda x, y: (rescaling_norm(x), y))

    # INSTANTIATE MODEL

    model = CustomClassifier()
    
    #choose optimizer and loss
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy'])

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.n_epochs
    )