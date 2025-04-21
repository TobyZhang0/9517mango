Traditional computer vision methods refer to the common ideas used to process images before the emergence of deep learning. Manually designed features (feature extraction) + classic machine learning classifier (ML classifier)
In this section, I completed four model(tuned)
HOG+SVM
HOG+KNN
LBP+SVM
LBP+KNN

## File Structure
`traditional_cv.ipynb`: Main notebook that includes:
  - Image preprocessing and grayscale conversion
  - Feature extraction using HOG and LBP
  - Model training using SVM and KNN with hyperparameter tuning
  - Evaluation with classification metrics and confusion matrices
  - Performance comparison and best model selection
- `README.md`: Project overview and instructions

## Dataset
We use the SkyView aerial landscape dataset, consisting of 15 categories with 800 images each.
[SkyView Dataset on Kaggle](https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset)
The dataset needs to be in the same directory for `traditional_cv.ipynb` to run.(No dataset uploaded to github because of size of dataset)
Each image is grayscale-processed before feature extraction.

## Methods Compared:
| Feature | Classifier | Description        |
|---------|------------|--------------------|
| HOG     | SVM        | Linear/RBF kernel with tuning |
| HOG     | KNN        | K=3,5,7 search     |
| LBP     | SVM        | Histogram-based texture features |
| LBP     | KNN        | Local texture + distance-based voting |

Each model uses `GridSearchCV` for hyperparameter tuning with 3-fold cross-validation.

## Evaluation:
Each model is evaluated using:
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-score (weighted)
- Classification report
- Confusion matrix

A final bar chart is displayed to compare the performance across all models.
The best-performing model based on **F1-score** is saved automatically as `best_model.joblib`.

## Requirements
- Python 3.8+
- `scikit-learn`
- `matplotlib`
- `scikit-image`
- `torchvision`
- `joblib`