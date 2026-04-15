# 💳 Credit Card Fraud Detection

A machine learning project that detects **fraudulent credit card transactions** using **Logistic Regression**, addressing the severe class imbalance problem through under-sampling.

---

## 📌 Project Overview

Credit card fraud is rare but costly. The dataset is highly imbalanced — only 0.17% of transactions are fraudulent. This project applies under-sampling to create a balanced training set, then trains a Logistic Regression classifier to distinguish legitimate from fraudulent transactions.

| Item | Detail |
|------|--------|
| **Algorithm** | Logistic Regression |
| **Task** | Binary Classification |
| **Dataset** | [Credit Card Fraud Detection – Kaggle (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| **Target** | `Class` — Normal (0) / Fraudulent (1) |

---

## 📂 Project Structure

```
credit_card_fraud_detection/
│
├── credit_card_fraud_detection.ipynb   # Jupyter Notebook (full walkthrough)
├── credit_card_fraud_detection.py      # Clean Python script
├── requirements.txt                    # Dependencies
├── creditcard.csv                      # Dataset (download from Kaggle)
├── eda_amount_distribution.png         # Transaction amount distributions
├── confusion_matrix.png                # Confusion matrix
└── README.md
```

---

## 📊 Dataset Features

| Feature | Description |
|---------|-------------|
| `Time` | Seconds elapsed between this and the first transaction |
| `V1–V28` | PCA-transformed features (anonymised for confidentiality) |
| `Amount` | Transaction amount in USD |
| `Class` | ✅ **Target** — 0 = Normal, 1 = Fraudulent |

> The dataset contains **284,807 transactions**: 284,315 legitimate and only **492 fraudulent**.

---

## ⚠️ Handling Class Imbalance

The dataset is severely imbalanced. Training directly on it would cause the model to predict "Normal" for everything and still achieve ~99.8% accuracy — which is meaningless.

**Solution: Under-sampling**
- Randomly sample 492 legitimate transactions
- Combine with all 492 fraudulent transactions
- Result: a balanced 50/50 dataset for fair training

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root.

### 4. Run
```bash
python credit_card_fraud_detection.py
```

---

## 🔄 Pipeline

```
Raw CSV Data (284,807 transactions)
    │
    ▼
EDA — Class distribution + Amount analysis
    │
    ▼
Under-Sampling (492 legit + 492 fraud = 984 total)
    │
    ▼
Train / Test Split (90% / 10%, stratified)
    │
    ▼
Logistic Regression Training
    │
    ▼
Accuracy + ROC-AUC + Classification Report + Confusion Matrix
    │
    ▼
Single-transaction Fraud Detection
```

---

## 📈 Results

| Split | Accuracy |
|-------|----------|
| Training | ~94% |
| Test | ~92% |

> ROC-AUC score provides a better measure than raw accuracy for imbalanced datasets.

---

## 🛠️ Tech Stack

- **Python 3.x**
- **pandas / numpy** — data processing
- **scikit-learn** — Logistic Regression, metrics, train/test split
- **seaborn / matplotlib** — visualization

---

## 🚀 Future Improvements

- [ ] Try SMOTE (Synthetic Minority Oversampling) instead of under-sampling
- [ ] Test Isolation Forest or Autoencoder for anomaly detection
- [ ] Add threshold tuning (optimise for recall to catch more fraud)
- [ ] Evaluate with Precision-Recall AUC (better for imbalanced data)
- [ ] Real-time fraud scoring API with FastAPI

---

## 📄 License

MIT License

---

## 🙋 Author

**[Your Name]**  
[GitHub](https://github.com/your-username) | [LinkedIn](https://linkedin.com/in/your-profile)
