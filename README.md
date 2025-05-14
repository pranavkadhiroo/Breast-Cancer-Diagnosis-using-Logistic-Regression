Here's a polished and presentable `README.md` for your GitHub project:


# Breast Cancer Diagnosis using Logistic Regression

![Machine Learning](https://img.shields.io/badge/-Machine%20Learning-blueviolet) ![Healthcare](https://img.shields.io/badge/-Healthcare-teal) ![Logistic Regression](https://img.shields.io/badge/-Logistic%20Regression-orange)

A machine learning project that implements logistic regression from scratch to classify breast tumors as malignant or benign using the Breast Cancer Wisconsin Diagnostic Dataset.

## 📌 Overview

This project demonstrates how logistic regression can be used for binary classification in medical diagnostics. Key features:

- **End-to-end implementation** of logistic regression using NumPy
- **Data preprocessing pipeline** for medical data
- **Model evaluation** with accuracy metrics
- **Educational focus** on understanding the underlying mathematics

The model achieves ~88% accuracy in classifying tumors, showcasing the potential of machine learning in early cancer detection.

## 📂 Dataset

The [Breast Cancer Wisconsin Diagnostic Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) contains 30 features computed from digitized images of fine needle aspirates (FNA) of breast masses, including:

- Radius (mean of distances from center to points on the perimeter)
- Texture (standard deviation of gray-scale values)
- Perimeter
- Area
- Smoothness (local variation in radius lengths)

**Target Variable:**
- `M` - Malignant (cancerous)
- `B` - Benign (non-cancerous)

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/pranavkadhiroo/breast-cancer-logistic-regression.git
   cd breast-cancer-logistic-regression
   ```

2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```

## 🚀 Usage

1. Place the dataset (`data.csv`) in the project directory
2. Run the main script:
   ```bash
   python breast_cancer_logistic_regression.py
   ```

The script will:
- Load and preprocess the data
- Train the logistic regression model
- Evaluate performance on test data
- Display accuracy metrics

## 🛠️ Project Structure

1. **Data Preprocessing**:
   - Handling missing values
   - Label encoding (M → 1, B → 0)
   - Feature normalization (Z-score standardization)

2. **Model Implementation**:
   - Sigmoid activation function
   - Forward and backward propagation
   - Gradient descent optimization
   - L2 regularization implementation

3. **Evaluation**:
   - Training/Test split (80/20)
   - Accuracy calculation
   - Loss curve visualization

## 📈 Results

| Metric        | Value   |
|---------------|---------|
| Train Accuracy| 90.68%  |
| Test Accuracy | 88.37%  |

The model demonstrates strong generalization performance, making it suitable for binary classification tasks in medical diagnostics.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements.
