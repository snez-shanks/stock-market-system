import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

from xgboost import XGBClassifier  # 🔥 Advanced model

from src.load_data import load_data


def train_model():
    # =========================
    # LOAD DATA
    # =========================
    df = load_data()

    if df is None or df.empty:
        print("❌ No data found")
        return

    # =========================
    # FEATURE ENGINEERING
    # =========================
    df['Daily_Return'] = df['Adj Close'].pct_change()
    df['Target_Next_Day_Close'] = df['Adj Close'].shift(-1)

    # Binary classification target
    df['Price_Up'] = (df['Target_Next_Day_Close'] > df['Adj Close']).astype(int)

    # Remove nulls
    df = df.dropna()

    # =========================
    # FEATURES & TARGET
    # =========================
    features = ['Open', 'High', 'Low', 'Volume', 'Daily_Return']
    X = df[features]

    y_reg = df['Target_Next_Day_Close']   # regression
    y_clf = df['Price_Up']                # classification

    # =========================
    # TRAIN TEST SPLIT
    # =========================
    X_train, X_test, y_reg_train, y_reg_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    _, _, y_clf_train, y_clf_test = train_test_split(
        X, y_clf, test_size=0.2, random_state=42
    )

    # =========================
    # 🔹 LINEAR REGRESSION
    # =========================
    lr = LinearRegression()
    lr.fit(X_train, y_reg_train)

    preds = lr.predict(X_test)

    # Residuals
    residuals = y_reg_test - preds

    # =========================
    # RESIDUAL PLOT
    # =========================
    plt.figure(figsize=(6, 4))
    plt.scatter(preds, residuals)
    plt.axhline(y=0, linestyle='--')
    plt.title("Residual Plot")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.show()

    # =========================
    # OUTLIER DETECTION
    # =========================
    z_scores = np.abs((residuals - residuals.mean()) / residuals.std())
    outliers = residuals[z_scores > 3]

    print("\n===== OUTLIERS DETECTED =====")
    print(outliers)

    # Metrics
    mse = mean_squared_error(y_reg_test, preds)
    r2 = r2_score(y_reg_test, preds)

    print("\n====== LINEAR REGRESSION ======")
    print("MSE:", mse)
    print("R2 Score:", r2)

    # =========================
    # 🔹 LOGISTIC REGRESSION
    # =========================
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_clf_train)

    log_preds = log_model.predict(X_test)
    log_acc = accuracy_score(y_clf_test, log_preds)

    print("\n====== LOGISTIC REGRESSION ======")
    print("Accuracy:", log_acc)

    # =========================
    # 🔹 RANDOM FOREST
    # =========================
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    rf.fit(X_train, y_clf_train)

    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_clf_test, rf_preds)

    print("\n====== RANDOM FOREST ======")
    print("Accuracy:", rf_acc)

    # =========================
    # 🔥 XGBOOST (BEST MODEL)
    # =========================
    xgb = XGBClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    xgb.fit(X_train, y_clf_train)

    xgb_preds = xgb.predict(X_test)
    xgb_acc = accuracy_score(y_clf_test, xgb_preds)

    print("\n====== XGBOOST (BEST MODEL) ======")
    print("Accuracy:", xgb_acc)

    print("\nClassification Report:\n")
    print(classification_report(y_clf_test, xgb_preds))

    # =========================
    # RETURN MODELS
    # =========================
    return {
        "linear_model": lr,
        "logistic_model": log_model,
        "random_forest": rf,
        "xgboost": xgb
    }


# =========================
# RUN DIRECTLY
# =========================
if __name__ == "__main__":
    train_model()