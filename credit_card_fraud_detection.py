# =============================================================================
# Credit Card Fraud Detection using Logistic Regression
# Author: [Your Name]
# Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# 1. Data Loading
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """Load the credit card transactions dataset."""
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nClass distribution:\n{df['Class'].value_counts()}")
    print(f"\n0 = Normal Transaction | 1 = Fraudulent Transaction")
    return df


# =============================================================================
# 2. Handle Class Imbalance via Under-Sampling
# =============================================================================

def balance_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Under-sample the majority class (legitimate) to match the minority
    class (fraud) for a balanced dataset.

    Fraud transactions: 492
    Legitimate transactions sampled: 492
    """
    legit = df[df.Class == 0]
    fraud = df[df.Class == 1]

    print(f"\nOriginal — Legit: {legit.shape[0]} | Fraud: {fraud.shape[0]}")

    legit_sample = legit.sample(n=len(fraud), random_state=1)
    balanced_df  = pd.concat([legit_sample, fraud], axis=0).reset_index(drop=True)

    print(f"Balanced — Legit: {len(legit_sample)} | Fraud: {len(fraud)}")
    print(f"\nMean comparison by class:\n{balanced_df.groupby('Class').mean()[['Amount', 'Time']]}")
    return balanced_df


# =============================================================================
# 3. EDA
# =============================================================================

def plot_eda(df: pd.DataFrame) -> None:
    """Plot transaction amount distributions for each class."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Credit Card Fraud – Transaction Amount Distribution", fontsize=14)

    df[df.Class == 0]['Amount'].plot(
        kind='hist', bins=50, ax=axes[0],
        color='steelblue', title='Legitimate Transactions'
    )
    axes[0].set_xlabel("Amount ($)")

    df[df.Class == 1]['Amount'].plot(
        kind='hist', bins=50, ax=axes[1],
        color='salmon', title='Fraudulent Transactions'
    )
    axes[1].set_xlabel("Amount ($)")

    plt.tight_layout()
    plt.savefig("eda_amount_distribution.png", dpi=150)
    plt.show()
    print("EDA saved as 'eda_amount_distribution.png'")


# =============================================================================
# 4. Feature / Target Split
# =============================================================================

def split_features_target(df: pd.DataFrame):
    X = df.drop(columns='Class', axis=1)
    Y = df['Class']
    return X, Y


# =============================================================================
# 5. Train / Test Split
# =============================================================================

def split_data(X, Y, test_size=0.1, random_state=1):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, stratify=Y, random_state=random_state
    )
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    return X_train, X_test, Y_train, Y_test


# =============================================================================
# 6. Model Training
# =============================================================================

def train_model(X_train, Y_train):
    """Train a Logistic Regression model."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)
    print("Model training complete.")
    return model


# =============================================================================
# 7. Model Evaluation
# =============================================================================

def evaluate_model(model, X_train, Y_train, X_test, Y_test) -> None:
    """Accuracy, ROC-AUC, classification report, and confusion matrix."""
    train_preds = model.predict(X_train)
    test_preds  = model.predict(X_test)

    print(f"\nTraining Accuracy : {accuracy_score(Y_train, train_preds):.4f}")
    print(f"Test     Accuracy : {accuracy_score(Y_test,  test_preds):.4f}")
    print(f"Test     ROC-AUC  : {roc_auc_score(Y_test, model.predict_proba(X_test)[:,1]):.4f}")

    print("\nClassification Report (Test Set):")
    print(classification_report(Y_test, test_preds,
                                target_names=['Legitimate', 'Fraudulent']))

    # Confusion matrix
    cm = confusion_matrix(Y_test, test_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate', 'Fraudulent'],
                yticklabels=['Legitimate', 'Fraudulent'])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()
    print("Confusion matrix saved as 'confusion_matrix.png'")


# =============================================================================
# 8. Predictive System
# =============================================================================

def predict_transaction(model, input_data: tuple) -> str:
    """
    Classify a single transaction as Legitimate or Fraudulent.

    Parameters
    ----------
    input_data : tuple of 30 values
        (Time, V1–V28, Amount) — same order as the dataset columns
    """
    arr = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(arr)
    if prediction[0] == 0:
        return "✅ Normal Transaction"
    else:
        return "🚨 Fraudulent Transaction — Flag for review!"


# =============================================================================
# Main Pipeline
# =============================================================================

if __name__ == "__main__":
    DATA_PATH = "creditcard.csv"   # update path if needed

    df = load_data(DATA_PATH)
    plot_eda(df)

    balanced_df = balance_dataset(df)

    X, Y = split_features_target(balanced_df)
    X_train, X_test, Y_train, Y_test = split_data(X, Y)

    model = train_model(X_train, Y_train)
    evaluate_model(model, X_train, Y_train, X_test, Y_test)

    # Sample prediction
    sample = (
        10, 0.38497821518095, 0.616109459176472, -0.874299702595052,
        -0.0940186259679115, 2.92458437838817, 3.31702716826156,
        0.470454671805879, 0.53824722837695, -0.558894612428441,
        0.30975539423728, -0.259115563735702, -0.326143233995877,
        -0.0900467227020648, 0.362832368569793, 0.928903660629178,
        -0.129486811402759, -0.809978925963589, 0.359985390219981,
        0.70766382644648, 0.12599157561542, 0.049923685888971,
        0.238421512225103, 0.00912986861262866, 0.996710209581086,
        -0.767314827174801, -0.492208295340017, 0.042472441919027,
        -0.0543373883732122, 9.99
    )
    result = predict_transaction(model, sample)
    print(f"\nSample Prediction: {result}")
