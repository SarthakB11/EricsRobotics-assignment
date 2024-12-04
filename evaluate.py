import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import preprocess_data

def evaluate_model(model, images, labels):
    train_ds, val_ds, test_ds = preprocess_data(images, labels)

    model.load_weights('defect_classification.weights.h5')

    val_loss, val_accuracy = model.evaluate(val_ds)
    print('Validation loss:', val_loss)
    print('Validation accuracy:', val_accuracy)

    test_loss, test_accuracy = model.evaluate(test_ds)
    print('Test loss:', test_loss)
    print('Test accuracy:', test_accuracy)

    y_pred = np.argmax(model.predict(test_ds), axis=1)
    y_test = np.concatenate([y for x, y in test_ds], axis=0)
    print(classification_report(y_test, y_pred, target_names=['rolled-in_scale', 'patches', 'crazing', 'pitted_surface', 'inclusion', 'scratches']))

    return y_test, y_pred
