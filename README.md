# Diabetes Health Indicators - Machine Learning Project

**COMP90049, Introduction to Machine Learning, Fall 2025**  
**Group: Phoenix (G06)**  
**School of Computing and Information Systems, The University of Melbourne**

---

## Project Overview

This project analyzes the **Diabetes Health Indicators Dataset** using various machine learning classification algorithms to predict diabetes status. The target variable has three classes: **No diabetes**, **Prediabetes**, and **Diabetes**.

We implement and compare four different machine learning models with comprehensive preprocessing, feature selection, and hyperparameter tuning to achieve optimal classification performance.

---

## Research Paper

You can read the full research paper here:

[View PDF](paper/MachineLearning_DiabitesPredictions.pdf)

---

## Code

The complete implementation and analysis can be found in the Jupyter notebook:

[View Notebook](MachineLearning_DiabitesPredictionsCode.ipynb)

> **Note:** GitHub renders Jupyter notebooks natively, so you can view the code, visualizations, and results directly in the browser without downloading the file.

---

## Dataset

The dataset used is the **Diabetes Health Indicators Dataset** from Kaggle:

-   **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
-   **File:** `data/dataset.csv`
-   **Description:** Contains health indicators and demographic information used to predict diabetes status

---

## Models Implemented

The project implements and compares the following machine learning models:

-   **K-Nearest Neighbors (KNN)**
-   **Logistic Regression**
-   **Decision Tree**
-   **Multilayer Perceptron (MLP) Neural Network**

### Methodology

All models utilize:

-   **Feature Selection:** Mutual information-based feature selection
-   **Class Imbalance Handling:** SMOTE (Synthetic Minority Oversampling Technique) and undersampling
-   **Hyperparameter Tuning:** GridSearchCV with cross-validation
-   **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, and ROC-AUC

---

## Getting Started

### Prerequisites

-   Python 3.7+
-   Jupyter Notebook or JupyterLab
-   pip (Python package manager)

### Installation

1. Clone or download this repository:

    ```bash
    git clone <repository-url>
    cd MLproject
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1. Ensure the dataset file `data/dataset.csv` exists in the project directory
2. Open `MachineLearning_DiabitesPredictionsCode.ipynb` in Jupyter Notebook
3. Run all cells sequentially to reproduce the analysis

---

## Requirements

All required Python packages are listed in `requirements.txt`. Key dependencies include:

-   `pandas` - Data manipulation and analysis
-   `numpy` - Numerical computing
-   `scikit-learn` - Machine learning algorithms and utilities
-   `imbalanced-learn` - Handling class imbalance
-   `matplotlib` & `seaborn` - Data visualization
-   `jupyter` - Notebook environment

---

## Group Members

-   **Xinyang Sun** – xinyang.sun.3@student.unimelb.edu.au
-   **Nina Lock** – ninalock@student.unimelb.edu.au
-   **Vadym Musiienko** – vmusiienko@student.unimelb.edu.au

---

## Acknowledgments

-   Dataset provided by [Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
-   This project was completed for **COMP90049 Introduction to Machine Learning** at The University of Melbourne, Fall 2025
