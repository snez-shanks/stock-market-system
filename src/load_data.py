import pandas as pd

def load_data():
    try:
        df = pd.read_csv("data/stock_data.csv")

        # Convert Date column
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

        # Sort data
        df = df.sort_values(by='Date')

        # Create features (important for your project)
        df['Daily_Return'] = df['Adj Close'].pct_change()
        df['Target_Next_Day_Close'] = df['Adj Close'].shift(-1)

        # Drop missing values
        df = df.dropna()

        return df

    except Exception as e:
        print("Error loading data:", e)
        return None