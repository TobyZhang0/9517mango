# Aerial Scene Classification with ResNet-18

This project implements aerial scene classification on the SkyView dataset (15 classes × 800 images) using a pretrained ResNet-18 backbone. We compare three data-augmentation strategies—Minimal, Default, and Extensive—and provide training, validation, testing, and confusion-matrix visualization.

## Table of Contents

- [Project Structure](#project-structure)  
- [Requirements](#requirements)   
- [Data Preparation](#data-preparation)  
- [Usage](#usage)   
- [Code Organization](#code-organization)   
- [Third-Party Libraries](#third-party-libraries)  

## Project Structure

├── README.md # AerialDataset and get_transforms  
└── resnet_18.ipynb # Model training and evaluation

## Requirements

torch>=1.12  
torchvision>=0.13  
numpy  
Pillow  
scikit-learn  
matplotlib  
seaborn  

## Data Preparation
Download the SkyView dataset from Kaggle:  
https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset  

Unzip into the Aerial_Landscapes/ directory so that:  

Aerial_Landscapes/  
  Agriculture/  
    img1.jpg  
    img2.jpg  
    ...  
  Airport/  
  ...  

## Usage
Run the Notebook

Open classification_notebook.ipynb and run all cells. The notebook will:

Load and split the dataset (60/20/20)

Define AerialDataset, transforms, model, training and evaluation functions

Train ResNet-18 with each augmentation strategy (Minimal, Default, Extensive)

Save the best model weights (best_{strategy}.pth)

Plot and save training curves (curves_{strategy}.png)

Compute and display confusion matrices (cm_{strategy}.png)

Use the notebook’s parameter cell at the top to adjust hyperparameters (batch_size, epochs, lr, wd, patience_es, strategy).

## Code Organization
All functionality lives in one Jupyter Notebook:

resnet_18.ipynb

Sections:

Imports & Config – dependencies and hyperparameters

Dataset Definition – AerialDataset class

Transforms – get_transforms for three strategies

DataLoaders – building train/val/test loaders

Model & Training – build_model, train_one_epoch, eval_one_epoch

Training Loop – runs experiments for each augmentation strategy

Evaluation – loads saved weights, computes accuracy & report

Visualization – saves curves and confusion matrices

Inline comments are provided throughout the notebook to explain logic and function parameters.

## Third-Party Libraries
This project uses the following external libraries—please refer to their official documentation for more details:

PyTorch & Torchvision: model definitions, pretrained weights, data loading

NumPy: numerical operations

Pillow: image reading and processing

scikit-learn: metrics (confusion matrix, classification report)

Matplotlib & Seaborn: plotting training curves and heatmaps
