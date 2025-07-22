# -Hybrid-Machine-Learning-Model-for-Network-Anomalies-Detection
This project implements a **hybrid machine learning framework** that combines supervised and unsupervised learning methods to detect **network anomalies**. It aims to improve accuracy, robustness, and adaptability for real-time and large-scale network environments.
## 🧠 Project Highlights

- **Hybrid Approach**: Combines decision trees, random forest, Naive Bayes, SVM, KNN, and logistic regression.
- **Ensemble & Stacking**: Implements model stacking by combining predictions (e.g., NB → Decision Tree).
- **False Positive Reduction**: Second-level classifier on misclassified samples.
- **Feature Selection**: Uses RandomForestClassifier and Recursive Feature Elimination (RFE).
- **Custom Evaluation Metrics**: Confusion matrix, precision, recall, and specific TP/TN/FP/FN analysis.

---

## 🚀 Features

- Real-time anomaly detection in network traffic
- Handles **class imbalance** using `imblearn`
- Visual performance comparison between models
- Evaluates models on false positives and false negatives
- Modular and reproducible pipeline

---

## 🧰 Technologies & Libraries

- **Programming Language**: Python 3.x
- **Libraries**:
  - `pandas`, `numpy` — Data preprocessing
  - `matplotlib`, `seaborn` — Visualization
  - `scikit-learn` — ML models and evaluation
  - `imblearn` — For handling class imbalance
- **Development Environment**: VS Code (Windows 11)

---

## 🖥️ System Requirements

### Hardware
- CPU: Multi-core processor
- RAM: Minimum 64 GB (preferred 128 GB)
- Storage: SSD with at least 1 TB
- Network: 10 Gbps+ recommended

### Software
- Windows 11 OS
- Python ≥ 3.8
- Required libraries (install with pip):
pip install -r requirements.txt

text

**`requirements.txt` sample**:
pandas
numpy
matplotlib
seaborn
scikit-learn
imblearn

text

---

## 📁 Dataset

- **Input**: Network traffic datasets (e.g., NSL-KDD)
- **Files**: `Train_data.csv`, `Test_data.csv`
- Ensure both files are placed in the `./input/` directory.

---

## ⚙️ Run Instructions

Clone the repository:
git clone https://github.com/yourusername/hybrid-network-anomaly-detection.git
cd hybrid-network-anomaly-detection

text

Prepare data and install dependencies:
pip install -r requirements.txt

text

Run the main code:
python main.py

text

---

## 📊 Results & Evaluation

- **Cross-Validation**: 10-fold CV used
- **Models Evaluated**:
  - Naive Bayes
  - Decision Tree
  - KNN
  - Logistic Regression
- **Custom Stacking**:
  - Misclassified instances from NB passed to Decision Tree
- **Metrics**:
  - Accuracy
  - Precision & Recall
  - Confusion Matrix
  - Specific gains of hybrid model vs base models

**Performance Comparison Example**:

| Metric         | Naive Bayes | Hybrid (NB → DT) |
|----------------|-------------|------------------|
| Accuracy       | 85.2%       | 91.3%            |
| Precision      | 78.4%       | 86.7%            |
| Sensitivity    | 80.1%       | 89.2%            |
| Specificity    | 87.3%       | 94.1%            |

---

## 📸 Visualizations

- Feature Importance
- Confusion Matrix
- Performance Metric Comparison Charts

---

## 👨‍💻 Authors

> Developed as part of a mini-project for **Visvesvaraya Technological University (VTU)**.

-**Punya M Shetty** 
- **Chandana S**  
- **Manushree M**   
- **Guide**: Prof. Shwetha N

---

## 📚 References

- [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html)
- IEEE Papers on hybrid anomaly detection methods—see project report for full citations.

---

## 📄 License

This project is for academic use only. For any commercial or reuse inquiries, please contact the project authors.
