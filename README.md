# Pest Detection Using CNN

This repository contains code for training and validating a Convolutional Neural Network (CNN) model for pest detection. The model is trained to identify 14 different insect species using a dataset consisting of approximately 5000 images.

## Overview

Pest detection is crucial in agriculture to monitor and control pest infestations, thereby ensuring crop health and productivity. Traditional methods of pest detection can be time-consuming and labor-intensive. This project aims to leverage deep learning techniques, specifically CNNs, to automate the process of identifying pests in agricultural settings.

## Dataset

The dataset used for training and validation consists of images of 14 different insect species commonly found in agricultural fields. The dataset is divided into training, validation, and test sets, with approximately 5000 images in total. Each image is labeled with the corresponding insect species.

## Model Architecture

The CNN model architecture used for pest detection is designed to effectively learn features from the input images and classify them into one of the 14 insect species. The architecture comprises several convolutional layers followed by max-pooling layers and fully connected layers. 

## Training Details

- Optimizer: Adam optimizer is used for training the model. It is an adaptive learning rate optimization algorithm that is well-suited for training deep neural networks.

- Learning Rate: The initial learning rate is set to 0.001. Learning rate scheduling techniques such as reducing the learning rate on plateau or using a cosine annealing schedule can be applied to improve training convergence.

- Activation Function: ReLU (Rectified Linear Unit) activation function is used in the convolutional layers to introduce non-linearity into the model. Softmax activation is applied to the output layer to obtain probability distributions over the 14 insect classes.

- Loss Function: Categorical cross-entropy loss is used as the loss function, which is suitable for multi-class classification problems.

## Training

The model training process involves feeding the training data through the CNN model and adjusting the model parameters to minimize the classification error. The training process is repeated for multiple epochs until the model converges and achieves satisfactory performance on the validation set.

## Evaluation

The trained model's performance is evaluated using the test set, which contains unseen images not used during training or validation. Evaluation metrics such as accuracy, precision, recall, and F1-score are computed to assess the model's effectiveness in accurately identifying pest species.

## Usage

To train the model:

1. Clone this repository:

2. Navigate to the project directory:


3. Install the required dependencies:

4. Run the training script:


5. Monitor the training progress and evaluate the model's performance on the validation set.

## Results

The trained model achieves 94%-97% ACCURACY on the test set, demonstrating its effectiveness in pest detection.

## Future Work

- Explore data augmentation techniques to further improve model generalization.
- Investigate transfer learning approaches using pre-trained models for pest detection.
- Extend the model to detect additional pest species for broader applicability.


## Acknowledgments

- https://www.kaggle.com/datasets/rtlmhjbn/ip02-dataset

---

Feel free to customize the README file further based on your specific project requirements and additional information you want to include.
