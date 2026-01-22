# Two-Class MI EEG Signal Classification

## Overview

This project focuses on the classification of two-class Motor Imagery (MI) EEG signals for Brain-Computer Interface (BCI) applications. The objective is to evaluate and compare classical machine learning algorithms for effective EEG-based human-computer interaction.

## Problem Statement

Motor Imagery EEG signal classification is a challenging task due to several factors:

- **Low signal-to-noise ratio**: EEG signals are inherently noisy and susceptible to artifacts
- **High dimensionality**: Multi-channel EEG recordings generate large feature spaces
- **Subject variability**: Individual differences in brain activity patterns affect generalization

This project aims to analyze the effectiveness of different machine learning models in distinguishing between two MI classes while addressing these challenges.

## Dataset

The dataset consists of EEG signals recorded during motor imagery tasks. Standard preprocessing and feature extraction techniques are applied before model training.

**Note**: Due to size and licensing constraints, the dataset is not directly included in the repository. Please refer to `data/README.md` for instructions on obtaining and preparing the dataset.

## Methodology

The project follows a systematic approach:

1. **EEG Signal Preprocessing**: Noise reduction and artifact removal
2. **Feature Extraction**: Extraction of discriminative features from MI EEG signals
3. **Model Training**: Implementation and training of multiple machine learning classifiers
4. **Hyperparameter Tuning**: Optimization of model parameters for improved performance
5. **Comparative Analysis**: Evaluation and comparison of model performance metrics

## Models Implemented

Three classical machine learning approaches were implemented and evaluated:

- **Linear Discriminant Analysis (LDA)**: Statistical classification based on linear decision boundaries
- **Support Vector Machine (SVM)**: Kernel-based classification with optimal margin separation
- **Artificial Neural Network (ANN)**: Deep learning approach with multiple hidden layers

## Hyperparameter Tuning

GridSearchCV was employed to systematically optimize model parameters, including:

- Learning rate
- Activation functions
- Number of epochs
- Batch size
- Regularization parameters

This optimization process aimed to maximize classification accuracy while preventing overfitting.

## Results

### Performance Summary

- **Classification accuracy range**: 62% to 69%
- **Average accuracy across models**: Approximately 65%
- **Best performing model**: [To be specified based on experimental results]

### Analysis

Comparative performance evaluation was conducted across all three models (LDA, SVM, and ANN), with detailed metrics including accuracy, precision, recall, and F1-score documented in the results directory.

## Tools and Technologies

- **Python**: Primary programming language
- **NumPy**: Numerical computation and array operations
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms and evaluation metrics
- **Matplotlib**: Data visualization and result plotting
- **Google Colab**: Cloud-based development environment for initial experimentation

## Project Structure

```
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
```

### Directory Descriptions

- **notebooks/**: Jupyter notebooks containing the complete workflow and analysis
- **data/**: Dataset information and preparation guidelines
- **results/**: Performance metrics and evaluation outputs
- **src/**: Source code modules for preprocessing, feature extraction, and model implementation

## Installation and Usage

### Prerequisites

Ensure you have Python 3.7 or higher installed on your system.

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/anujxsun/Two-Class-MI-EEG-Classification.git
   cd Two-Class-MI-EEG-Classification
   ```

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the dataset:**
   - Follow the instructions in `data/README.md` to obtain and prepare the EEG dataset
   - Place the prepared data in the appropriate directory

4. **Run the analysis:**
   - Open the notebook using Jupyter Notebook:
     ```bash
     jupyter notebook notebooks/MI_EEG_Two_Class_Classification.ipynb
     ```
   - Alternatively, upload to Google Colab for cloud-based execution
   - Execute all cells sequentially to reproduce the results

## Reproducibility

All experiments were conducted with fixed random seeds to ensure reproducibility. The complete workflow, from preprocessing to model evaluation, is documented in the Jupyter notebook.

## Limitations and Future Work

### Current Limitations

- Binary classification only (two-class MI)
- Limited to classical machine learning approaches
- Dataset-specific performance may vary with different BCI paradigms

### Future Directions

- Extension to multi-class MI classification
- Integration of deep learning architectures (CNN, LSTM, Transformers)
- Real-time BCI system implementation
- Cross-subject and cross-session validation
- Exploration of ensemble methods for improved robustness

## Contributing

Contributions, issues, and feature requests are welcome. Please feel free to check the issues page or submit a pull request.

## License

This project is available under the MIT License. See the LICENSE file for more details.

## Author

**Anuj Kumar Kashyap**

For questions, suggestions, or collaborations, please reach out through the repository's issue tracker.

## Acknowledgments

Special thanks to the BCI research community and open-source contributors whose work has informed this project. Initial experimentation and development were performed using Google Colab.

---

*Last updated: January 2026*
