import os
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

def load_images_and_labels(train_dir, annotation_dir):
    images = []
    labels = []
    defect_types = ['rolled-in_scale', 'patches', 'crazing', 'pitted_surface', 'inclusion', 'scratches']

    for defect_type in defect_types:
        defect_path = os.path.join(train_dir, defect_type)
        for image_name in os.listdir(defect_path):
            image_path = os.path.join(defect_path, image_name)
            annotation_path = os.path.join(annotation_dir, image_name.replace('.jpg', '.xml'))

            if not os.path.exists(annotation_path):
                print(f"Annotation file not found: {annotation_path}")
                continue

            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, IMAGE_SIZE)
            images.append(image)

            tree = ET.parse(annotation_path)
            root = tree.getroot()
            label = defect_types.index(defect_type)
            labels.append(label)

    return np.array(images), np.array(labels)

def preprocess_data(images, labels):
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)

    return train_ds, val_ds, test_ds
