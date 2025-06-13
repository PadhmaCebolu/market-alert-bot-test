import pandas as pd
import datetime
import pytz
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# ---------- Fetch SPX Close from Investing.com ----------
def get_spx_close():
    try:
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--log-level=3")
        options.add_experimental_option("excludeSwitches", ["enable-logging"])

        driver = webdriver.Chrome(options=options)
        driver.get("https://www.investing.com/indices/us-spx-500")
        time.sleep(4)
        price = driver.find_element(By.CSS_SELECTOR, '[data-test="instrument-price-last"]').text.replace(",", "")
        driver.quit()
        return float(price)
    except Exception as e:
        print("Error fetching SPX close:", e)
        return None

# ---------- Evaluate and Update Prediction Logs ----------
def evaluate_predictions(log_file="market_predictions.csv"):
    df = pd.read_csv(log_file)

    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Fill missing columns if absent
    for col in ['actual_spx_close', 'actual_trend', 'Match/Miss']:
        if col not in df.columns:
            df[col] = "N/A"

    # Normalize missing fields
    df.fillna("N/A", inplace=True)
    df['actual_trend'] = df['actual_trend'].astype(str)
    df['Match/Miss'] = df['Match/Miss'].astype(str)

    # Get today's date in CST
    now_cst = datetime.datetime.now(pytz.timezone("US/Central"))
    today_str = now_cst.strftime("%Y-%m-%d")

    # Identify rows to update
    mask = (
        df['timestamp'].dt.strftime("%Y-%m-%d") == today_str
    ) & (
        df['actual_trend'].isin(["N/A", "nan", ""])  # match all "missing" styles
    )

    if mask.sum() == 0:
        print("✅ No missing actuals for today.")
        return

    # Fetch actual SPX close
    actual_close = get_spx_close()
    if actual_close is None:
        print("❌ Failed to fetch SPX close.")
        return

    # Apply updates
    for idx in df[mask].index:
        predicted = df.loc[idx, 'predicted_trend']
        spx_open = df.loc[idx, 'spx']

        if isinstance(spx_open, (int, float)) or (isinstance(spx_open, str) and spx_open.replace('.', '', 1).isdigit()):
            spx_open = float(spx_open)
            actual_trend = "Bullish" if actual_close > spx_open else "Bearish"
            match = "1" if actual_trend in predicted else "0"
        else:
            actual_trend, match = "N/A", "N/A"

        df.at[idx, 'actual_spx_close'] = str(actual_close)
        df.at[idx, 'actual_trend'] = str(actual_trend)
        df.at[idx, 'Match/Miss'] = match

    df.to_csv(log_file, index=False)
    print(f"✅ Updated {mask.sum()} entries with actual close: {actual_close}")

# Run
if __name__ == "__main__":
    evaluate_predictions()
