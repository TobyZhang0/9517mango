# 9517mango - Aerial Image Scene Classification Project

This project is an aerial image scene classification system that employs various deep learning models for classifying aerial images. It includes implementations of both traditional methods and multiple deep learning models to compare their performance on aerial image classification tasks.

## Requirements

- Python 3.8+
- PyTorch 1.8.0+
- torchvision 0.9.0+
- numpy 1.19.0+
- matplotlib 3.3.0+
- scikit-learn 0.24.0+
- albumentations (for data augmentation)
- kornia (for data augmentation)
- pytorch_grad_cam (for GradCAM visualization)

## Dataset

The project uses the Aerial_Landscapes dataset, which includes the following 15 scene categories:
- Residential
- Agriculture
- Grassland
- Mountain
- Railway
- Parking
- Highway
- Airport
- Forest
- Desert
- River
- Beach
- Port
- Lake
- City

## Data Augmentation Strategies

The project implements multiple data augmentation strategies through the `get_transforms` function in `dataset.py`:

1. Minimal Augmentation
   - Resize to 224x224
   - Random horizontal flip
   - Basic normalization using ImageNet statistics

2. Default Augmentation
   - Resize to 256x256
   - Random crop to 224x224
   - Random horizontal flip
   - Random resized crop (scale: 0.6-1.0)
   - Random Gaussian blur (p=0.3)
   - Random solarize (threshold=0.5, p=0.2)
   - Random rotation (20 degrees)
   - Color jitter (brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1)
   - Random affine (translation=0.1)
   - Random erasing (p=0.2)

3. Extensive Augmentation
   - All default augmentations
   - Random vertical flip
   - Random resized crop (scale: 0.5-1.0)
   - More aggressive color transformations
   - Higher rotation angles (30 degrees)
   - Random perspective
   - Random sharpness adjustment
   - Random autocontrast
   - Random equalization

4. New Augmentation Strategy
   - Resize to 224x224
   - Random rotation (30 degrees)
   - Random horizontal and vertical flips
   - Color jitter
   - Random perspective
   - Random sharpness adjustment
   - Random autocontrast
   - Random equalization

## Model Descriptions

### 1. Traditional Method
- Implemented by: Yiheng Zhang
- Features:
  - Uses HOG (Histogram of Oriented Gradients) and LBP (Local Binary Patterns) for feature extraction
  - Supports multiple classifiers: SVM and KNN
  - Includes complete data preprocessing and feature extraction pipeline
  - Provides detailed evaluation metrics and visualization

### 2. Vision Transformer (ViT)
- Implemented by: Hengzhang
- Features:
  - Based on timm library's vit_base_patch16_224 pretrained model
  - Custom classification head: 512-dimensional intermediate layer with BatchNorm and Dropout
  - Supports multiple data augmentation strategies: minimal, default, extensive, new
  - Uses AdamW optimizer and ReduceLROnPlateau learning rate scheduler
  - Includes class weight balancing and label smoothing
  - GradCAM visualization for model interpretability
    - Uses pytorch_grad_cam library
    - Targets the last transformer block's normalization layer
    - Generates heatmaps for both correct and incorrect predictions
    - Visualizes attention patterns in transformer layers

### 3. ResNet
- Implemented by: Wang Yuyang
- Features:
  - Based on ResNet18 pretrained model
  - Supports multiple data augmentation strategies
  - Complete training and evaluation pipeline
  - Includes model saving and loading functionality

### 4. DenseNet
- Implemented by: Haitao Ye
- Features:
  - Based on DenseNet121 pretrained model
  - Added SE (Squeeze-and-Excitation) attention modules
  - Custom network architecture with SE blocks at key positions
  - Supports multiple data augmentation strategies
  - Includes complete training and evaluation pipeline

### 5. EfficientNet
- Implemented by: Boya Liu
- Features:
  - Based on EfficientNet-B0 pretrained model
  - Supports multiple data augmentation strategies
  - Complete training history recording and visualization
  - Detailed evaluation metrics: accuracy, precision, recall, F1-score
  - Confusion matrix visualization

## Project Structure

```
.
├── Aerial_Landscapes/      # Dataset directory
├── traditional/           # Traditional method implementation
│   ├── traditional_cv.ipynb  # Main implementation file
│   └── README.md          # Method description
├── vit/                  # Vision Transformer implementation
│   ├── train_vit_base_final.ipynb  # Main implementation file
│   ├── gradcam_visualization.png    # Sample GradCAM visualization
│   ├── misclassified_images.png     # Analysis of misclassified samples
│   ├── confusion_matrix.png         # Confusion matrix visualization
│   ├── training_history.png         # Training progress visualization
│   └── saved_models_vit/  # Saved models
├── resnet/               # ResNet implementation
│   ├── resnet.py         # Model definition
│   └── resnet_update.ipynb  # Training and evaluation
├── DenseNet/             # DenseNet implementation
│   ├── DenseNet121_SE.ipynb  # SE module enhanced version
│   └── DenseNet121.ipynb     # Base version
├── EfficientNet/         # EfficientNet implementation
│   ├── EfficientNet.ipynb    # Main implementation file
│   ├── training_history_minimal.png    # Training history with minimal augmentation
│   ├── training_history_default.png    # Training history with default augmentation
│   └── training_history_extensive.png  # Training history with extensive augmentation
├── evaluate/             # Evaluation related code
│   └── evaluate_main.ipynb  # Main evaluation notebook
├── dataset.py            # Dataset processing and augmentation
└── loadtest.ipynb        # Testing and loading experiments
```

## How to Run

1. Data Preparation
   ```bash
   # Ensure the dataset is correctly placed in the Aerial_Landscapes directory
   ```

2. Model Training
   ```bash
   # Navigate to the corresponding model directory
   cd [model_name]
   # Run the training script
   python train.py  # or run the corresponding notebook file
   ```

3. Model Evaluation
   ```bash
   # Navigate to the evaluation directory
   cd evaluate
   # Run the evaluation script
   python evaluate.py
   ```

4. Visualization Generation
   ```bash
   # For GradCAM visualization (ViT model)
   cd vit
   python generate_gradcam.py
   ```

## Contributors

- Yiheng Zhang - Traditional method implementation
- Hengzhang - Vision Transformer implementation
- Wang Yuyang - ResNet implementation
- Haitao Ye - DenseNet implementation
- Boya Liu - EfficientNet implementation 