# EricsRobotics-assignment

## Approach and Methodology

### Dataset
The dataset used for this project is the NEU Surface Defect Database, which contains images of various types of surface defects. The dataset is divided into training and validation sets.

### Data Preprocessing
1.  **Loading Images and Labels** : Images and their corresponding annotations were loaded and preprocessed. The images were resized to 224x224 pixels and normalized.
2.  **Data Augmentation** : Techniques such as random rotation, translation, zoom, flip, contrast, and brightness adjustments were applied to augment the training data.
3.  **Dataset Splitting** : The dataset was split into training, validation, and test sets with a ratio of 70:15:15.
4.  **Caching operations** : The prefetch method loads the data required for the next step in memory, when the current step is being executed. It reduces the step time to the maximum (as opposed to the sum) of the training and the time it takes to extract the data. The cache method will cache the dataset and save some operations (like file opening and data reading) from being executed during each epoch.

### Model Architecture
The model used is based on the EfficientNetB2 architecture, pre-trained on ImageNet. The base model was frozen, and additional layers were added:
1.  **Flatten Layer** : To flatten the output from the base model.
2.  **Dense Layers** : Two dense layers with 1024 and 512 units respectively, using ReLU activation.
3.  **Output Layer** : A dense layer with softmax activation to classify the defect types.

### Training
The model was compiled using the Adam optimizer with a learning rate of 1e-5 and sparse categorical cross-entropy loss. The model was trained for 10 epochs with early stopping based on validation accuracy.

## Model Performance and Metrics

### Training and Validation Performance
![Training and Validation Accuracy](https://github.com/user-attachments/assets/e13ee7f6-8a1b-474d-b3df-12a7ed1c98e3)
![Training and Validation Loss](https://github.com/user-attachments/assets/dea29468-ab0a-48e7-907d-49381a2ad091)

### Evaluation Metrics
-  **Validation Loss** : 0.1052456945180893
-  **Validation Accuracy** : 0.9768518805503845
-  **Test Loss** : 0.10745564103126526
-  **Test Accuracy** : 0.9629629850387573

### Classification Report
![Classification Report](https://github.com/user-attachments/assets/d3a4877b-7958-46c2-8417-4929151aff02)

The classification report provides valuable insights into the model's performance across different defect types:

- **Rolled-in Scale**: The model achieved a precision of 0.56 and a recall of 0.56, indicating a balanced performance in identifying this defect type. The F1-score of 0.56 suggests moderate accuracy.
- **Patches**: With a precision of 0.71 and a perfect recall of 1.00, the model excels at identifying patches. The high F1-score of 0.83 reflects excellent performance for this defect type.
- **Crazing**: The model demonstrates strong performance with a precision of 0.93 and a recall of 0.98, resulting in an F1-score of 0.96. This indicates high accuracy in detecting crazing defects.
- **Pitted Surface**: The precision is 0.86 and the recall is 0.94, leading to an F1-score of 0.90. This shows that the model is effective in identifying pitted surfaces.
- **Inclusion**: The model has a precision of 0.85 and a recall of 0.54, with an F1-score of 0.69. This suggests that while the model is precise in identifying inclusions, it may miss some instances (lower recall).
- **Scratches**: The model achieves perfect precision (1.00) but has a lower recall of 0.74, resulting in an F1-score of 0.85. This indicates that the model is very accurate when it identifies scratches but may not detect all instances.


### Confusion Matrix
![Confusion Matrix](https://github.com/user-attachments/assets/6b817e10-ee5c-4f76-9160-9be4c2e19277)

The confusion matrix provides a detailed breakdown of the model's predictions versus the actual labels:

- **Diagonal Elements**: The high values along the diagonal indicate that the model correctly identifies most defect types with high accuracy, particularly for patches, crazing, and pitted surfaces.
- **Off-Diagonal Elements**: The off-diagonal elements show where the model made incorrect predictions. For example, some rolled-in scale defects were misclassified as patches or inclusions, and some inclusions were misclassified as scratches.
- **Misclassifications**: The model struggles most with rolled-in scale and inclusion defects, as indicated by the higher off-diagonal values for these classes. This aligns with the lower recall values for these defect types in the classification report.

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
- **Strengths**: The model performs exceptionally well in identifying patches, crazing, and pitted surfaces, with high precision and recall. The confusion matrix confirms this with high diagonal values for these classes.
- **Balanced Performance**: The balanced accuracy (macro and weighted averages) indicates that the model performs consistently across different defect types, with no significant bias towards any particular type. The confusion matrix supports this by showing a generally good distribution of correct predictions along the diagonal.
-  **Data Augmentation** : Significantly improved the model's performance by increasing the diversity of the training data.
-  **Transfer Learning** : Using a pre-trained model (EfficientNetB2) helped in achieving better performance with less training time.

### Challenges
-  **Imbalanced Dataset** : Some defect types had fewer samples, which could lead to biased predictions.The model shows lower recall for defect types with fewer samples, such as rolled-in scale and inclusions. This is evident in both the classification report and the confusion matrix.
-  **Annotation Quality** : Inconsistencies in annotations affected the model's learning process.Misclassifications observed in the confusion matrix, such as rolled-in scale being confused with patches or inclusions, could be partly due to annotation errors.
-  **Interpretatbility** : Deep learning models, particularly those based on complex architectures like EfficientNet are known for their lack of interpretability. It can be difficult to understand why the model makes certain predictions, which is crucial for trust and debugging. Grad-CAM or SHAP can provide insights.

## Conclusion
The model achieved high accuracy in classifying different types of surface defects. The use of data augmentation and transfer learning played a crucial role in improving the model's performance. 

## Future work 
- It could focus on addressing the imbalanced dataset and improving annotation quality.
- The model could benefit from improvements in detecting rolled-in scale and inclusions, where the recall is relatively lower. The confusion matrix highlights specific misclassifications, such as rolled-in scale being confused with patches or inclusions. Enhancing the recall for scratches could also improve overall performance.

## Acknowledgments
[1]: NEU Surface Defect Database Link: https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database/data

[2]: Tan, M., & Le, Q.V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ArXiv, abs/1905.11946.
