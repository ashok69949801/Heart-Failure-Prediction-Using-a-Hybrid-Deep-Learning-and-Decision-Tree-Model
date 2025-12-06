# Heart Failure Prediction Using Hybrid ANN–Decision Tree Model

This project builds a hybrid deep learning and machine learning model to predict heart failure from clinical patient data using the Cleveland Heart Disease dataset. An Artificial Neural Network (ANN) is used for feature extraction, and a Decision Tree classifier uses these optimized features to provide accurate and interpretable predictions.

## Dataset

- Source: Cleveland Heart Disease dataset (Kaggle / UCI benchmark)  
- Records: 1,025 patient entries  
- Attributes: 14 clinically relevant features (13 input features + 1 target indicating presence or absence of heart failure)

## Methodology

- Data preprocessing:
  - Handling missing values for key features
  - Normalization of numerical attributes
  - One-hot encoding for categorical variables
- Model architecture:
  - ANN (Multi-Layer Perceptron) with ReLU activation for non-linear feature learning
  - Penultimate ANN layer used to extract 8 optimized features from the original 14 attributes
  - Decision Tree classifier trained on these extracted features for final prediction
- Evaluation:
  - Train–test split (80% training, 20% testing)
  - Metrics: accuracy, precision, recall, confusion matrix

## Results

The hybrid ANN–Decision Tree model achieves 98.54% test accuracy on the Cleveland dataset, outperforming several traditional models such as Naive Bayes, Logistic Regression, k-NN, SVM, AdaBoost, and standalone Decision Trees.

## Tech Stack

- Python  
- NumPy, Pandas  
- scikit-learn  
- TensorFlow/Keras (or PyTorch, based on implementation)  
- Matplotlib, Seaborn

## Project Structure

- `data/` – Dataset files or instructions to download them  
- `notebooks/` – Exploratory data analysis and experiments  
- `src/` – Model, training, and evaluation scripts  
- `models/` – Saved model weights/checkpoints  
- `reports/` – Plots, metrics, and documentation
  
## Future Work

- Integrate additional real-world clinical data (EHRs, wearable devices, continuous ECG).  
- Explore ensemble methods and advanced optimization to further boost robustness.  
- Apply explainable AI techniques to enhance clinical interpretability and trust.

