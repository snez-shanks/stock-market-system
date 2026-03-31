import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

def perform_eda(df):

    print("\n===== PHASE 1: EDA =====")

    # ========================
    # 1. DESCRIPTIVE STATS
    # ========================
    print("\n--- Volume Stats ---")
    print("Mean:", df['Volume'].mean())
    print("Variance:", df['Volume'].var())
    print("Std Dev:", df['Volume'].std())

    print("\n--- Daily Return Stats ---")
    print("Mean:", df['Daily_Return'].mean())
    print("Variance:", df['Daily_Return'].var())
    print("Std Dev:", df['Daily_Return'].std())

    # ========================
    # 2. GRAPH
    # ========================
    plt.figure(figsize=(10,5))
    plt.plot(df['Date'], df['Adj Close'])
    plt.title("Stock Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Adj Close")
    plt.show()

    # ========================
    # 3. HYPOTHESIS TESTING
    # ========================
    print("\n===== HYPOTHESIS TEST =====")

    # Create Up/Down column
    df['Up'] = (df['Daily_Return'] > 0).astype(int)

    vol_up = df[df['Up'] == 1]['Volume']
    vol_down = df[df['Up'] == 0]['Volume']

    # t-test
    t_stat, p_value = ttest_ind(vol_up, vol_down)

    print("T-statistic:", t_stat)
    print("P-value:", p_value)

    print("\nInterpretation:")
    print("H0: Volume same on up & down days")
    print("H1: Volume is different")

    if p_value < 0.05:
        print("✅ Reject H0 → Volume is significantly different")
    else:
        print("❌ Fail to reject H0 → No significant difference")