import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_history(history):
    plt.plot(history.history['sparse_categorical_accuracy'])
    plt.plot(history.history['val_sparse_categorical_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def plot_confusion_matrix(y_test, y_pred):
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(['rolled-in_scale', 'patches', 'crazing', 'pitted_surface', 'inclusion', 'scratches']))
    plt.xticks(tick_marks, ['rolled-in_scale', 'patches', 'crazing', 'pitted_surface', 'inclusion', 'scratches'], rotation=45)
    plt.yticks(tick_marks, ['rolled-in_scale', 'patches', 'crazing', 'pitted_surface', 'inclusion', 'scratches'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
