from argparse import Namespace
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import PIL.Image
import pathlib
from models import CustomClassifier, ResnetClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
        batch_size=args.batch_size)
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(args.crop_size, args.crop_size),
        batch_size=args.batch_size)
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        seed=123,
        image_size=(args.crop_size, args.crop_size),
        batch_size=args.batch_size)
    
    rescaling_norm = tf.keras.layers.Rescaling(1./255)

    #normalize in range [0,1]
    train_ds = train_ds.map(lambda x, y: (rescaling_norm(x), y))
    val_ds = val_ds.map(lambda x, y: (rescaling_norm(x), y))

    # INSTANTIATE MODEL

    model = CustomClassifier(apply_augmentation=args.use_augmentation) if args.model == "custom" else ResnetClassifier(apply_augmentation=args.use_augmentation)
    
    #choose optimizer and loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate = args.lr),
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
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
    x, y = test_ds 
    preds = model.predict(x, training=False)

    acc = tf.keras.metrics.Accuracy()
    acc = acc.update_state(preds,y)

    f1 = tf.keras.metrics.F1Score(num_classes=2)
    f1 = f1.update_state(preds, y)

    recall = tf.keras.metrics.Recall()
    recall = recall.update_state(preds, y)

    precision = tf.keras.metrics.Precision()
    precision = precision.update_state(preds, y)

    print("accuracy: ", acc.result().numpy())
    print("f1: ", f1.result().numpy())
    print("recall: ", recall.result().numpy())
    print("precision: ", precision.result().numpy())

    cm=confusion_matrix(y, preds, ["cats", "dogs"],)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    figpath = os.path.join(".", "figs", "confusion.png")
    plt.savefig(figpath, format= "png")
    plt.close()
    
