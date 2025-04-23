# Aerial Scene Classification with EfficientNet

This project explores the task of multi-class classification on aerial landscape images using the **EfficientNet** architecture, a state-of-the-art model known for its balance between performance and efficiency.

The classification is performed on the publicly available [Aerial_Landscapes](https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset) dataset.

---

## Dataset

- Dataset: **Aerial_Landscapes**
- Format: RGB images categorized into 15 scene classes (e.g., airport, beach, forest, etc.)
- Directory placement:
  - Please download and extract the dataset to a folder named `Aerial_Landscapes` **in the parent directory** of the notebook.
  - Structure example:
    ```
    ├── Aerial_Landscapes/
    │   ├── airport/
    │   ├── beach/
    │   └── ...
    └── EfficientNet/
        └── EfficientNet.ipynb
    ```

---

## Model Description

The notebook implements the **EfficientNet** model for aerial scene classification. EfficientNet uses compound scaling of depth, width, and resolution to achieve better performance with fewer parameters.

The notebook includes:

- Data preprocessing and augmentation
- EfficientNet model construction and training
- Evaluation metrics and visualizations

---

## Results & Visualization

- Metrics: Accuracy, Precision, Recall, F1-score
- Visuals: Confusion matrix, training/validation accuracy and loss curves
- All results and plots can be viewed directly in the notebook output cells.

---

## How to Run

1. Clone or download this repository (9517mango).
2. Make sure the dataset folder `Aerial_Landscapes` is located in the parent directory of the notebook.
3. Open `EfficientNet.ipynb` in Jupyter Notebook or VS Code with Jupyter extension.
4. Run the cells sequentially to train and evaluate the model.

> Tip: The notebook is self-contained and does not require additional configuration beyond dataset placement.

---

## External Resources Used

- Dataset: [Aerial_Landscapes on Kaggle](https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset)

---

## Acknowledgements

This project is part of an academic exploration of aerial image classification using modern deep learning models. Thanks to the authors of EfficientNet for their contributions to the field.