# VGG16 CNN Model with PyTorch

This repository contains a Convolutional Neural Network (CNN) implemented using the VGG16 architecture with PyTorch. The model is trained on a custom dataset and includes several important features to enhance performance and generalization.

## Key Features:
- **VGG16 Architecture**: Utilizes a pre-trained VGG16 model as the base, with modifications to the fully connected layers for classification tasks.
- **Learning Rate Decay**: Implements learning rate decay to improve training efficiency and avoid overfitting.
- **Early Stopping**: Monitors validation loss to halt training early and prevent unnecessary overfitting.
- **Training & Testing**: Tracks both training and testing loss, as well as accuracy metrics.
- **F1 Score**: Computes the F1 score to evaluate the model’s precision and recall performance.
- **Custom Dataset**: The model is trained and evaluated on a custom dataset, with data loading and preprocessing steps handled within the notebook.

## Requirements:
- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- Torchvision
- PIL
- Scikit-Learn

## Model Performance:
- **Training Loss/Accuracy**: Evaluates model performance during training.
- **Test Loss/Accuracy**: Assesses the model on unseen data.
- **F1 Score**: Provides a balance between precision and recall.

## License:
This project is licensed under the MIT License.
