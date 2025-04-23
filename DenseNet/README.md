# Aerial Scene Classification with DenseNet121 and SE Block

This project explores the task of multi-class classification on aerial landscape images using two variants of the DenseNet architecture:

- **DenseNet121**: Standard dense convolutional network.
- **DenseNet121 with SE Block**: Integrates a channel attention mechanism (Squeeze-and-Excitation block) to enhance feature representations.

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
    └── DenseNet/
        ├── DenseNet121.ipynb
        ├── DenseNet121_SE.ipynb
    ```

---

## Model Variants

The following two notebook files are included:

1. **DenseNet121.ipynb**: Implements the baseline DenseNet121 model for image classification.
2. **DenseNet121_SE.ipynb**: Extends the base model by adding a Squeeze-and-Excitation (SE) block for channel attention.

Both notebooks include data loading, training, evaluation, and result visualization.

---

## Results & Visualization

- Metrics: Accuracy, Precision, Recall, F1-score
- Visuals: Confusion matrix, training/validation loss and accuracy plots
- You can find the complete evaluation results and training curves in the output cells of the respective notebooks.

---

## How to Run

1. Clone or download this repository(9517mango).
2. Make sure the dataset folder `Aerial_Landscapes` is located in the parent directory of the notebooks.
3. Open `DenseNet121.ipynb` or `DenseNet121_SE.ipynb` in Jupyter Notebook or VS Code with Jupyter extension.
4. Run the cells sequentially to train and evaluate the models.

>  Tip: These notebooks are self-contained and do not require additional configuration beyond dataset placement.

---

## External Resources Used

- Dataset: [Aerial_Landscapes on Mendeley Data](https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset)
- SE Block concept: [Squeeze-and-Excitation Networks (Hu et al., 2018)](https://arxiv.org/abs/1709.01507)

---

## Acknowledgements

This project is part of an academic exploration of aerial image classification using modern deep learning models. Thanks to the authors of DenseNet and SE Block for their foundational work.
