import pandas as pd
import numpy as np

from src.load_data import load_data


def run_pca():
    df = load_data()

    # Select relevant columns
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = df[features].dropna()

    print("\n===== ORIGINAL DATA SAMPLE =====")
    print(data.head())

    # ==============================
    # 🔹 COVARIANCE MATRIX
    # ==============================
    cov_matrix = np.cov(data.T)

    print("\n===== COVARIANCE MATRIX =====")
    print(cov_matrix)

    # ==============================
    # 🔹 EIGEN VALUES & VECTORS
    # ==============================
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    print("\n===== EIGENVALUES =====")
    print(eigenvalues)

    print("\n===== EIGENVECTORS =====")
    print(eigenvectors)

    # ==============================
    # 🔹 PRINCIPAL COMPONENT
    # ==============================
    max_index = np.argmax(eigenvalues)
    principal_component = eigenvectors[:, max_index]

    print("\n===== PRINCIPAL COMPONENT =====")
    print(principal_component)

    # ==============================
    # 🔹 DIMENSION REDUCTION
    # ==============================
    reduced_data = data.dot(principal_component)

    print("\n===== REDUCED DATA SAMPLE =====")
    print(reduced_data.head())


if __name__ == "__main__":
    run_pca()