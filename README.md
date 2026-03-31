# 📈 Stock Market Analysis & Prediction System

## 📌 Project Overview

This project focuses on analyzing historical stock market data and building predictive models using statistical and machine learning techniques.

The system performs:

* Exploratory Data Analysis (EDA)
* Hypothesis Testing
* Dimensionality Reduction using Eigenvalues & Eigenvectors (PCA)
* Predictive Modeling using Regression & Classification

---

## 🎯 Problem Statement

Given a historical stock dataset, the objective is to:

* Analyze stock behavior and volatility
* Reduce feature dimensionality using linear algebra
* Build models to predict future stock price movements

---

## 📂 Project Structure

```
stock-market-system/
│
├── data/
│   └── stock_data.csv
│
├── src/
│   ├── load_data.py
│   ├── preprocess.py
│   ├── model.py
│   ├── pca_analysis.py
│
├── notebooks/
│   └── analysis.ipynb
│
├── models/
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Description

The dataset contains historical stock data with the following columns:

* Date
* Open
* High
* Low
* Close
* Adj Close
* Volume


## 🧠 Project Phases

### ✅ Phase 1: EDA & Hypothesis Testing

* Created:

  * `Daily_Return`
  * `Target_Next_Day_Close`
* Performed:

  * Mean, Variance, Standard Deviation
* Visualization:

  * Time series plot of stock prices
* Hypothesis Testing:

  * Compared trading volume on up-days vs down-days

---

### ✅ Phase 2: Feature Optimization (Linear Algebra)

* Constructed Covariance Matrix
* Calculated:

  * Eigenvalues
  * Eigenvectors
* Identified Principal Component
* Reduced feature dimensionality

---

### ✅ Phase 3: Statistical Modeling

* ✔ Linear Regression (Price Prediction)
* ✔ Logistic Regression (Up/Down Prediction)
* ✔ Random Forest (Improved Classification)

### 📈 Model Performance

* Linear Regression:

  * High R² Score (~0.99)
* Logistic Regression:

  * Accuracy ~56%
* Random Forest:

  * Accuracy ~57–58%

---

### 🔍 Diagnostics

* Residual Plot Analysis
* Outlier Detection using Z-score

---


## 📈 Results

| Model               | Performance       |
| ------------------- | ----------------- |
| Linear Regression   | R² ≈ 0.99         |
| Logistic Regression | Accuracy ≈ 55–57% |
| Random Forest       | Accuracy ≈ 57–58% |

---

## ⚠️ Important Insight

Stock market prediction is inherently uncertain due to:

* Market volatility
* External factors (news, economy, sentiment)

Hence, even ~55–60% accuracy is considered reasonable.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* SciPy

---

## ⚙️ Installation

### 1. Clone Repository

```
git clone <your-repo-link>
cd stock-market-system
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Run Project

### Run Model (Terminal)

```
python -m src.model
```

### Run Streamlit App

```
streamlit run app/app.py
```

---

## 🌐 Web App Features

* View stock dataset
* Train ML models
* Display predictions
* Visualize trends

---

## 📌 Future Improvements

* Use LSTM (Deep Learning)
* Add more technical indicators
* Hyperparameter tuning
* Real-time stock prediction

---


## 👩‍💻 Author

Sneha Shankarwal
B.Tech (IT) – Machine Learning & Data Analytics

---

## ⭐ Conclusion

This project successfully demonstrates:

* Statistical analysis of stock data
* Application of linear algebra (PCA)
* Machine learning for prediction

It provides a strong foundation for financial data science and predictive analytics.









