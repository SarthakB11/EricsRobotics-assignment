# EricsRobotics-assignment

## Approach and Methodology

### Dataset
The dataset used for this project is the NEU Surface Defect Database, which contains images of various types of surface defects. The dataset is divided into training and validation sets.

### Data Preprocessing
1.  Loading Images and Labels : Images and their corresponding annotations were loaded and preprocessed. The images were resized to 224x224 pixels and normalized.
2.  Data Augmentation : Techniques such as random rotation, translation, zoom, flip, contrast, and brightness adjustments were applied to augment the training data.
3.  Dataset Splitting : The dataset was split into training, validation, and test sets with a ratio of 70:15:15.

### Model Architecture
The model used is based on the EfficientNetB2 architecture, pre-trained on ImageNet. The base model was frozen, and additional layers were added:
1.  Flatten Layer : To flatten the output from the base model.
2.  Dense Layers : Two dense layers with 1024 and 512 units respectively, using ReLU activation.
3.  Output Layer : A dense layer with softmax activation to classify the defect types.

### Training
The model was compiled using the Adam optimizer with a learning rate of 1e-5 and sparse categorical cross-entropy loss. The model was trained for 10 epochs with early stopping based on validation accuracy.

## Model Performance and Metrics

### Training and Validation Performance
![Training and Validation Accuracy](https://github.com/user-attachments/assets/e13ee7f6-8a1b-474d-b3df-12a7ed1c98e3)
![Training and Validation Loss](https://github.com/user-attachments/assets/dea29468-ab0a-48e7-907d-49381a2ad091)

### Evaluation Metrics
-  Validation Loss : 0.1052456945180893
-  Validation Accuracy : 0.9768518805503845
-  Test Loss : 0.10745564103126526
-  Test Accuracy : 0.9629629850387573

### Classification Report
![Classification Report](https://github.com/user-attachments/assets/d3a4877b-7958-46c2-8417-4929151aff02)


### Confusion Matrix
![Confusion Matrix](https://github.com/user-attachments/assets/6b817e10-ee5c-4f76-9160-9be4c2e19277)


## Bonus Objective

### Identification and Classification of Specific Defect Types
The model successfully identified and classified the specific types of defects in the defective images, including:
- Rolled-in Scale
- Patches
- Crazing
- Pitted Surface
- Inclusion
- Scratches

## Insights and Challenges

### Insights
-  Data Augmentation : Significantly improved the model's performance by increasing the diversity of the training data.
-  Transfer Learning : Using a pre-trained model (EfficientNetB2) helped in achieving better performance with less training time.

### Challenges
-  Imbalanced Dataset : Some defect types had fewer samples, which could lead to biased predictions.
-  Annotation Quality : Inconsistencies in annotations affected the model's learning process.

## Conclusion
The model achieved high accuracy in classifying different types of surface defects. The use of data augmentation and transfer learning played a crucial role in improving the model's performance. Future work could focus on addressing the imbalanced dataset and improving annotation quality.

## Acknowledgments
Special thanks to the providers of the NEU Surface Defect Database for making the dataset publicly available.
