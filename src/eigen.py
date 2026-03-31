import numpy as np

def eigen_analysis(df):

    print("\n===== PHASE 2: EIGEN ANALYSIS =====")

    features = df[['Open','High','Low','Volume']]

    # ========================
    # 1. COVARIANCE MATRIX
    # ========================
    cov_matrix = features.cov()
    print("\nCovariance Matrix:\n", cov_matrix)

    # ========================
    # 2. EIGENVALUES & VECTORS
    # ========================
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    print("\nEigenvalues:\n", eigenvalues)
    print("\nEigenvectors:\n", eigenvectors)

    # ========================
    # 3. PRINCIPAL COMPONENT
    # ========================
    max_index = np.argmax(eigenvalues)
    principal_component = eigenvectors[:, max_index]

    print("\nPrincipal Component (Max Variance):")
    print(principal_component)

    print("\nExplanation:")
    print("This vector shows the direction of maximum variance.")
    print("It helps reduce dimensions while keeping maximum information.")