import os
from data_loader import load_images_and_labels
from model_builder import build_model
from train import train_model
from evaluate import evaluate_model
from visualize import plot_history, plot_confusion_matrix

def main():
    username = "YOUR_KAGGLE_USERNAME"
    key = "YOUR_KAGGLE_KEY"
    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_KEY'] = key

    os.system("kaggle datasets download -d kaustubhdikshit/neu-surface-defect-database")
    os.system("unzip neu-surface-defect-database.zip")

    train_dir = 'NEU-DET/train/images'
    annotation_dir = 'NEU-DET/train/annotations'

    images, labels = load_images_and_labels(train_dir, annotation_dir)
    model = build_model()
    history = train_model(model, images, labels)
    evaluate_model(model, images, labels)
    plot_history(history)
    plot_confusion_matrix(model, images, labels)

if __name__ == "__main__":
    main()
