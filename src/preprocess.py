import pandas as pd

def preprocess(df):
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

    df = df.sort_values(by='Date')

    # Phase 1 features
    df['Daily_Return'] = df['Adj Close'].pct_change()
    df['Target_Next_Day_Close'] = df['Adj Close'].shift(-1)

    df = df.dropna()

    return df