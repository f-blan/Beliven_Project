from argparse import Namespace
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import PIL.Image
import pathlib
from models import CustomClassifier, ResnetClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_recall_fscore_support, accuracy_score
import numpy as np

def train(args: Namespace):
    
    train_dir = pathlib.Path(args.train_dir)
    test_dir = pathlib.Path(args.test_dir)

    # PREPARE THE DATASETS
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(args.crop_size, args.crop_size),
        batch_size=args.batch_size,
        label_mode='categorical',)
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(args.crop_size, args.crop_size),
        batch_size=args.batch_size,
        label_mode='categorical',)
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        seed=123,
        image_size=(args.crop_size, args.crop_size),
        batch_size=args.batch_size,
        label_mode='categorical',)
    
    rescaling_norm = tf.keras.layers.Rescaling(1./255)

    #normalize in range [0,1]
    train_ds = train_ds.map(lambda x, y: (rescaling_norm(x), y))
    val_ds = val_ds.map(lambda x, y: (rescaling_norm(x), y))
    test_ds = test_ds.map(lambda x, y: (rescaling_norm(x), y))

    # INSTANTIATE MODEL

    model = CustomClassifier(apply_augmentation=args.use_augmentation, crop_size=args.crop_size) if args.model == "custom" else ResnetClassifier(apply_augmentation=args.use_augmentation, crop_size= args.crop_size)
    
    #choose optimizer and loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate = args.lr),
        loss="categorical_crossentropy",
        metrics=['accuracy'])

    plateau_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=3, min_lr=0.00001)

    history= model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.n_epochs,
        callbacks=[plateau_callback]
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    lr = history.history['lr']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    figpath = os.path.join(".", "figs", "accuracy_history.png")
    plt.savefig(figpath, format= "png")
    plt.close()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    figpath = os.path.join(".", "figs", "loss_history.png")
    plt.savefig(figpath, format= "png")
    plt.close()

    plt.plot(epochs, lr, 'bo', label='Learning rate')
    plt.title('Training and validation loss')
    plt.legend()

    figpath = os.path.join(".", "figs", "lr_history.png")
    plt.savefig(figpath, format= "png")
    plt.close()

    print("performing inference on the test set")
    x = np.concatenate([x for x, y in test_ds], axis=0)
    y = np.concatenate([y for x, y in test_ds], axis=0) 

    y=np.argmax(y,1)
      
    preds = model.predict(x)
    preds = tf.math.argmax(preds, 1)

    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds) 
    precision, recall,_,__ = precision_recall_fscore_support(y, preds)

    print("accuracy: ", acc)
    print("f1: ", f1)
    print("recall: ", recall)
    print("precision: ", precision)

    cm=confusion_matrix(y, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["cats", "dogs"],)
    disp.plot()
    figpath = os.path.join(".", "figs", "confusion.png")
    plt.savefig(figpath, format= "png")
    plt.close()
