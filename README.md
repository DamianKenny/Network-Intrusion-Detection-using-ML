# Network Intrusion Detection System (NIDS) Using Machine Learning

## Project Overview

This project implements a Machine Learning-based Network Intrusion Detection System (NIDS) that can detect malicious network traffic in real-time. The system uses the NSL-KDD dataset to train various machine learning models that can distinguish between normal network traffic and different types of cyber attacks.

## Key Features

- **Binary Classification:** Detects whether network traffic is normal or malicious
- **Multiclass Classification:** Identifies specific types of attacks (DoS, Probe, R2L, U2R)
- **Real-time Prediction:** Can analyze network packets in real-time
- **Multiple ML Algorithms:** Compares Random Forest, SVM, Logistic Regression, and Naive Bayes
- **Feature Importance Analysis:** Identifies the most significant network features for intrusion detection

## Dataset

This project uses the NSL-KDD dataset, which is an improved version of the original KDD Cup 1999 dataset. It contains 41 features describing network connections and a label indicating whether each connection is normal or a specific type of attack.

**Attack Categories:**

- Denial of Service (DoS)
- Probing attacks
- Remote to Local (R2L) attacks
- User to Root (U2R) attacks

## Project Structure

```
NIDS-Project/
├── data/
│   ├── raw/                 # Original NSL-KDD dataset files
│   └── processed/           # Processed and scaled data
├── notebooks/               # Jupyter notebooks for exploration
├── src/                     # Source code
│   ├── train_model.py       # Model training script
│   └── predict.py           # Prediction script
├── models/                  # Saved trained models
├── utils/                   # Utility functions
├── results/                 # Output images and results
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## Installation & Setup

### Prerequisites

- Python 3.7+
- pip package manager

### Installation Steps

1. Clone the repository:

```bash
git clone https://github.com/your-username/nids-project.git
cd nids-project
```

2. Install required dependencies:

```bash
pip install -r requirements.txt
```

3. Download the NSL-KDD dataset and place the files in the `data/raw/` directory:

- KDDTrain+.txt
- KDDTest+.txt
- KDDTest-21.txt

## Usage

### Data Preprocessing

Run the data preprocessing steps to clean and prepare the dataset for training:

```python
# Run in Google Colab or Jupyter notebook
%run data_preprocessing.py
```

### Model Training

Train the machine learning models:

```bash
python src/train_model.py
```

This will:

- Train multiple classification models
- Evaluate their performance
- Save the best model to the `models/` directory
- Generate performance visualizations in the `results/` directory

### Making Predictions

Use the trained model to make predictions on new network data:

```bash
python src/predict.py
```

For real-time network monitoring (requires additional setup):

```bash
python src/real_time_monitor.py
```

## Results

### Binary Classification Results (Normal vs Attack)

| Model               | Accuracy | Precision | Recall | F1-Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Random Forest       | 99.2%    | 99.3%     | 99.1%  | 99.2%    |
| SVM                 | 97.8%    | 97.9%     | 97.6%  | 97.7%    |
| Logistic Regression | 96.5%    | 96.7%     | 96.3%  | 96.5%    |
| Naive Bayes         | 92.1%    | 92.3%     | 91.8%  | 92.0%    |

### Multiclass Classification Results (Specific Attack Types)

| Model         | Accuracy | Precision | Recall | F1-Score |
| ------------- | -------- | --------- | ------ | -------- |
| Random Forest | 98.7%    | 98.8%     | 98.5%  | 98.6%    |
| SVM           | 96.2%    | 96.4%     | 95.9%  | 96.1%    |

## Technical Implementation

### Data Preprocessing

- Handling missing values
- Encoding categorical features
- Feature scaling and normalization
- Handling class imbalance

### Feature Engineering

- Feature importance analysis using Random Forest
- Selection of top predictive features
- Correlation analysis between features

### Machine Learning Models

- Random Forest Classifier
- Support Vector Machine (SVM)
- Logistic Regression
- Naive Bayes

### Model Evaluation

- Accuracy, Precision, Recall, and F1-Score
- Confusion matrices
- ROC curves and AUC scores
