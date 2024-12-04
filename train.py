import tensorflow as tf
from tensorflow import keras
from data_loader import preprocess_data

EPOCHS = 10

def data_augmentation(x):
    augmentation_layers = [
        layers.RandomRotation(0.2),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom(0.2),
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2)
    ]
    for layer in augmentation_layers:
        x = layer(x)
    return x

def train_model(model, images, labels):
    train_ds, val_ds, test_ds = preprocess_data(images, labels)

    train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))

    train_ds = train_ds.map(lambda x, y: (keras.applications.efficientnet.preprocess_input(x), y))
    val_ds = val_ds.map(lambda x, y: (keras.applications.efficientnet.preprocess_input(x), y))
    test_ds = test_ds.map(lambda x, y: (keras.applications.efficientnet.preprocess_input(x), y))

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE).cache()
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE).cache()
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE).cache()

    checkpoint_filepath = 'defect_classification.weights.h5'
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_sparse_categorical_accuracy',
        mode='max',
        save_best_only=True)

    history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=[model_checkpoint_callback])

    return history
