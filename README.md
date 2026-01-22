
# Two-Class MI EEG Signal Classification

## Overview
This project focuses on the classification of two-class Motor Imagery (MI) EEG signals for Brain–Computer Interface (BCI) applications. The objective is to evaluate and compare classical machine learning algorithms for effective EEG-based human–computer interaction.

## Problem Statement
Motor Imagery EEG signal classification is a challenging task due to low signal-to-noise ratio, high dimensionality, and subject variability. This project aims to analyze the effectiveness of different machine learning models in distinguishing two MI classes.

## Dataset
The dataset consists of EEG signals recorded during motor imagery tasks. Standard preprocessing and feature extraction techniques are applied before model training. Due to size and licensing constraints, the dataset is not directly included in the repository.

## Methodology
The project follows the steps below:
1. EEG signal preprocessing and noise reduction
2. Feature extraction from MI EEG signals
3. Training and evaluation of machine learning models
4. Hyperparameter tuning for performance improvement
5. Comparative analysis of model performance

## Models Implemented
- Linear Discriminant Analysis (LDA)
- Support Vector Machine (SVM)
- Artificial Neural Network (ANN)

## Hyperparameter Tuning
GridSearchCV was used to optimize model parameters including learning rate, activation functions, number of epochs, and batch size to improve classification performance.

## Results
- Classification accuracy range: 62% to 69%
- Average accuracy across models: approximately 65%
- Comparative performance evaluation conducted for LDA, SVM, and ANN

## Tools and Technologies
- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Google Colab

  ## Project Structure
Two-Class-MI-EEG-Classification/
├── notebooks/
│   └── MI_EEG_Two_Class_Classification.ipynb
├── data/
│   └── README.md
├── results/
│   └── metrics.txt
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   └── models.py
├── requirements.txt
└── README.md




## How to Run
1. Clone the repository:
   git clone https://github.com/yourusername/Two-Class-MI-EEG-Classification.git
2. Install required dependencies:
   pip install -r requirements.txt
3. Open the notebook using Jupyter Notebook or Google Colab and run all cells sequentially.

## Notes
Initial experimentation and development were performed using Google Colab. The final implementation and results are provided in this repository.

## Author
Anuj Kumar Kashyap
