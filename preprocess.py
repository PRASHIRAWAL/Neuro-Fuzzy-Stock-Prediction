import yfinance as yf
import pandas as pd
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_data(tickers, start_date, end_date):
    """
    Fetch stock data from Yahoo Finance.
    """
    logging.info(f"Fetching data for {tickers} from {start_date} to {end_date}")
    df_list = []
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        if not data.empty:
            data['Ticker'] = ticker
            df_list.append(data)
    
    if not df_list:
        logging.warning("No data fetched from Yahoo Finance. Falling back to synthetic data.")
        return generate_synthetic_data(tickers, start_date, end_date)
        
    combined_df = pd.concat(df_list)
    return combined_df

def generate_synthetic_data(tickers, start_date, end_date):
    """
    Fallback method to generate synthetic data if yfinance fails.
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    df_list = []
    
    np.random.seed(42)  # For reproducibility
    for ticker in tickers:
        n_days = len(dates)
        
        # Simulate realistic stock patterns
        base_price = np.random.uniform(1000, 3000)
        daily_returns = np.random.normal(0.0005, 0.02, n_days)
        prices = base_price * np.exp(np.cumsum(daily_returns))
        
        # Simulate OHLCV
        highs = prices * np.random.uniform(1.0, 1.03, n_days)
        lows = prices * np.random.uniform(0.97, 1.0, n_days)
        opens = prices * np.random.uniform(0.99, 1.01, n_days)
        closes = prices
        volumes = np.random.negative_binomial(1, 0.0001, n_days) + 1000000
        
        df = pd.DataFrame({
            'Date': dates,
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': closes,
            'Volume': volumes,
            'Ticker': ticker
        })
        # Set Date as index to mimic yfinance output
        df.set_index('Date', inplace=True)
        df_list.append(df)
        
    return pd.concat(df_list)

def engineer_features(df):
    """
    Calculate Volatility, Trend, and normalized Volume.
    """
    logging.info("Engineering features: Volatility, Trend, Volume")
    
    # Flatten MultiIndex columns if necessary (yfinance sometimes returns MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1) # Drop ticker names in case of yf multi-index
    
    # Handle missing values
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    # Calculate Features
    df['Volatility'] = df['High'] - df['Low']
    df['Trend'] = df['Close'] - df['Open']
    
    # Volume normalization (min-max scaling per ticker or overall; here we do overall as placeholder, 
    # but practically we should just use raw volume scaled later via StandardScaler. 
    # For the formula `Volume * 0.1` to make sense alongside price diffs, normalization to similar magnitude is good)
    # We will just normalize Volume to a 0-1 range locally to create the risk score initially.
    df['Volume_Norm'] = (df['Volume'] - df['Volume'].min()) / (df['Volume'].max() - df['Volume'].min())
    
    return df

def calculate_risk_score_and_labels(df):
    """
    Risk Score = (Volatility * 0.6) - (Trend * 0.3) + (Volume_Norm * 0.1)
    Converts Risk Score into discrete labels (0: Low, 1: Medium, 2: High)
    """
    logging.info("Calculating Risk Score and generating labels")
    
    # Calculate Risk Score
    df['Risk_Score'] = (df['Volatility'] * 0.6) - (df['Trend'] * 0.3) + (df['Volume_Norm'] * 0.1)
    
    # Define thresholds for labels based on quantiles to ensure balanced classes
    low_thresh = df['Risk_Score'].quantile(0.33)
    high_thresh = df['Risk_Score'].quantile(0.67)
    
    def categorize_risk(score):
        if score <= low_thresh:
            return 0  # Low
        elif score <= high_thresh:
            return 1  # Medium
        else:
            return 2  # High
            
    df['Risk_Class'] = df['Risk_Score'].apply(categorize_risk)
    
    return df

def preprocess_pipeline():
    """
    Main preprocessing pipeline.
    """
    TICKERS = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']
    START_DATE = '2020-01-01'
    END_DATE = '2024-01-01'
    OUTPUT_PATH = 'data/preprocessed_data.csv'
    
    os.makedirs('data', exist_ok=True)
    
    # 1. Fetch Data
    df = fetch_data(TICKERS, START_DATE, END_DATE)
    
    # 2. Feature Engineering
    df = engineer_features(df)
    
    # 3. Calculate Labels
    df = calculate_risk_score_and_labels(df)
    
    # 4. Save to CSV (keep only required columns for training)
    features = ['Volatility', 'Trend', 'Volume_Norm', 'Risk_Class', 'Risk_Score', 'Ticker']
    final_df = df[features]
    final_df.to_csv(OUTPUT_PATH, index=False)
    
    logging.info(f"Preprocessing complete. Saved {len(final_df)} records to {OUTPUT_PATH}")
    logging.info(f"Class distribution:\n{final_df['Risk_Class'].value_counts(normalize=True)}")

if __name__ == "__main__":
    try:
        preprocess_pipeline()
    except Exception as e:
        logging.error(f"Error in preprocessing: {e}")
