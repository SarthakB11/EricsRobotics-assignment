import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMAGE_SIZE = (224, 224)

def build_model():
    base_model = keras.applications.EfficientNetB2(
        include_top=False,
        weights="imagenet",
        input_shape=IMAGE_SIZE+(3,)
    )

    base_model.trainable = False

    defect_types = ['rolled-in_scale', 'patches', 'crazing', 'pitted_surface', 'inclusion', 'scratches']
    x = layers.Flatten()(base_model.output)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    predictions = layers.Dense(len(defect_types), activation='softmax')(x)

    model = keras.Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=keras.optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=[keras.metrics.SparseCategoricalAccuracy()])

    return model
