import requests
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import datetime
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv
import csv
import openai
import pytz

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
finnhub_api_key = os.getenv("FINNHUB_API_KEY")
marketaux_api_key = os.getenv("MARKETAUX_API_KEY")
openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")



# =============================
# 📋 Utility Functions
# =============================

import json

def classify_headlines_openai_bulk(headlines, batch_size=10):
    score_map = {"📈": 3, "📉": -3, "🔹": 0}
    all_classified = []

    for i in range(0, len(headlines), batch_size):
        chunk = headlines[i:i+batch_size]
        try:
            prompt = [
                {"role": "system", "content": (
                    "You are a financial sentiment classifier.\n\n"
                    "Classify each headline as:\n"
                    "📈 (bullish), 📉 (bearish), or 🔹 (neutral), with confidence (0.0–1.0) and a brief reason.\n"
                    "Respond ONLY with a JSON list like:\n"
                    "[{\"sentiment\": \"📈\", \"confidence\": 0.8, \"reason\": \"Rate cuts expected\"}, ...]"
                )},
                {"role": "user", "content": "\n".join(chunk)}
            ]

            response = openai.chat.completions.create(
                model=openai_model,
                messages=prompt,
                temperature=0.3,
                max_tokens=700
            )

            result_text = response.choices[0].message.content.strip()
            if result_text.startswith("```"):
                result_text = result_text.strip("```json").strip("```").strip()

            try:
                parsed = json.loads(result_text)
            except json.JSONDecodeError:
                last_valid = result_text.rfind("}")
                result_text = result_text[:last_valid+1] + "]"
                parsed = json.loads(result_text)

            for item in parsed:
                emoji = item.get("sentiment", "🔹")
                score = score_map.get(emoji, 0)
                reason = item.get("reason", "")
                conf = float(item.get("confidence", 0))
                all_classified.append((emoji, score, reason, conf))

        except Exception as e:
            print("❌ GPT classification failed:", e)
            all_classified.extend([("🔹", 0, "Error", 0)] * len(chunk))

    return all_classified

    
def is_market_relevant(text):
    keywords = [
        "fed", "federal reserve", "interest rate", "tariff", "rate", "inflation", "disinflation",
        "deflation", "yields", "bond", "treasury", "quantitative tightening", "quantitative easing",
        "hawkish", "dovish", "fomc", "central bank", "earnings", "revenue", "guidance",
        "stocks", "markets", "indices", "s&p", "nasdaq", "dow", "volatility", "vix",
        "recession", "soft landing", "jobless", "unemployment", "nonfarm", "cpi", "ppi", "gdp",
        "retail sales", "housing", "consumer sentiment", "ism", "pce", "commodities", "oil prices",
        "energy prices", "geopolitical", "china", "opec", "rate hike", "rate cut", "ecb", "boe", "boj",
        "debt ceiling", "fiscal policy", "economic outlook", "layoffs", "banking crisis", "credit rating",
        "supply chain", "default", "liquidity", "volumes", "short selling", "options expiry", "earnings call"
    ]
    text_lower = text.lower()
    return any(k in text_lower for k in keywords)

# =============================
# 📋 News Scrapers
# =============================

def scrape_headlines(url, selector, base_url=""):
    headlines = []
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        for el in soup.select(selector)[:10]:
            text = el.get_text(strip=True)
            link = el.get("href", "")
            if text and is_market_relevant(text):
                full_link = f"{base_url}{link}" if link.startswith("/") else link
                headlines.append(f"{text} - {full_link}")
    except Exception as e:
        print(f"⚠️ Error scraping {url}:", e)
    return headlines

def get_all_market_news():
    headlines_raw = []

    # 📰 1. Macenews, CNBC, Reuters (Web scraping)
    #headlines_raw += scrape_headlines("https://macenews.com/", ".elementor-heading-title")
    headlines_raw += scrape_headlines("https://www.cnbc.com/world/?region=world", "a.Card-title")
    #headlines_raw += scrape_headlines("https://www.reuters.com/", "a[data-testid='Heading']", base_url="https://www.reuters.com")

    # 📰 2. Finnhub News
    def fetch_finnhub_news():
        url = f"https://finnhub.io/api/v1/news?category=general&token={finnhub_api_key}"
        try:
            response = requests.get(url).json()
            for item in response[:10]:
                title = item.get("headline", "")
                url = item.get("url", "")
                if title:
                    headlines_raw.append(f"{title} - {url}")
        except Exception as e:
            print("❌ Finnhub news fetch failed:", e)

        

    # 📰 3. Marketaux News
    def fetch_marketaux_news():
        published_after = (datetime.utcnow() - timedelta(hours=3)).isoformat() + "Z"
        url = f"https://api.marketaux.com/v1/news/all?symbols=SPY&published_after={published_after}&filter_entities=true&language=en&api_token={marketaux_api_key}"
        try:
            response = requests.get(url).json()
           for article in response.get("data", [])[:10]:
               title = article.get("title", "")
               url = article.get("url", "")
                if title:
                    headlines_raw.append(f"{title} - {url}")
       except Exception as e:
           print("❌ Marketaux news fetch failed:", e)

    # Fetch from APIs
    fetch_finnhub_news()
    fetch_marketaux_news()

    # Filter and classify
    headlines_raw = [h for h in headlines_raw if is_market_relevant(h)]
    classified = classify_headlines_openai_bulk(headlines_raw)

    enhanced_news = []
    for original, sentiment in zip(headlines_raw, classified):
        emoji, score, reason, confidence = sentiment
        enhanced_news.append((emoji, score, f"{emoji} {original} — {reason}", confidence))


    return enhanced_news


# =============================
# 📋 Market Data
# =============================

def get_price_from_investing(url):
    try:
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--log-level=3")  # Suppress USB/logging warnings
        options.add_experimental_option("excludeSwitches", ["enable-logging"])  # Optional extra suppressor

        driver = webdriver.Chrome(options=options)
        driver.get(url)
        time.sleep(4)
        price = driver.find_element(By.CSS_SELECTOR, '[data-test="instrument-price-last"]').text.replace(",", "")
        driver.quit()
        return float(price)
    except Exception as e:
        print(f"⚠️ Error retrieving price from {url}:", e)
        return "N/A"

def get_spx():
    return get_price_from_investing("https://www.investing.com/indices/us-spx-500")

def get_vix():
    return get_price_from_investing("https://www.investing.com/indices/volatility-s-p-500")

def get_es():
    return get_price_from_investing("https://www.investing.com/indices/us-spx-500-futures")

# =============================
# 📋 Analysis & Bias
# =============================

def estimate_direction(spx, es, sentiment_score, vix, news):
    # Only use sentiment_score for now
    if sentiment_score > 0:
        direction = "📈 Bullish"
    elif sentiment_score < 0:
        direction = "📉 Bearish"
    else:
        direction = "🔹 Neutral"

    avg_conf = sum(c for _, _, _, c in news) / len(news) if news else 0.5
    direction_with_conf = f"{direction} (Confidence: {round(avg_conf, 2)})"

    # Debug log
    print("🧪 DEBUG NEWS-ONLY DIRECTION")
    print(f"  - Total Sentiment Score: {sentiment_score}")
    print(f"  - Avg News Confidence: {round(avg_conf, 2)}")
    print(f"  - Direction: {direction_with_conf}")

    return direction_with_conf, [f"News-driven sentiment score = {sentiment_score}"]

# =============================
# 📅 Logging
# =============================

def log_premarket_prediction(date, spx, es, vix, sentiment_score, direction, news):
    log_file = "market_predictions.csv"
    file_exists = os.path.isfile(log_file)

    news_str = " | ".join([h for _, _, h, _ in news])



    with open(log_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                "date", "spx", "es", "vix", "sentiment_score",
                "predicted_trend", "actual_trend", "actual_spx_close", "Match/Miss", "news"
            ])
        writer.writerow([
            date, spx, es, vix, sentiment_score,
            direction, "nan", "nan", "nan", news_str
        ])



# =============================
# 📧 Email Notifier
# =============================

def send_email(subject, spx, vix, es, news, direction, reasons, move_msg, to_email):
    try:
        import pytz
        # Get current time in US/Eastern
        eastern = pytz.timezone('US/Eastern')
        current_time_est = datetime.datetime.now(eastern).strftime('%I:%M %p ET')

        message = MIMEMultipart("alternative")
        message["From"] = os.getenv("EMAIL_USER")
        message["To"] = to_email
        message["Subject"] = subject
        headline_lines = "\n".join([f"- {h}" for _, _, h, _ in news])
        reason_lines = "\n".join([f"- {r}" for r in reasons])
        # Plaintext fallback
        body_text = f"""
                
                📊 Pre-Market Alert for {datetime.date.today()}
                🔹 SPX: {spx}  🔺 VIX: {vix}  📉 ES: {es}

                📰 Headlines:
                {headline_lines}

                📊 Market Bias: {direction}
                {reason_lines}

                📉 Expected Move: {move_msg}
                Generated by CDUS Trading Bot • {current_time_est}
                """

        # HTML version
        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px; color: #333;">
            <h2 style="color: #0d6efd;">📊 Pre-Market Alert for {datetime.date.today()}</h2>
            <p>
                <strong>🔹 SPX:</strong> {spx} &nbsp;&nbsp;
                <strong>🔺 VIX:</strong> {vix} &nbsp;&nbsp;
                <strong>📉 ES:</strong> {es}
            </p>

            <h3>📰 Headlines:</h3>
            <ul>
                {''.join(f"<li>{h}</li>" for _, _, h, _ in news)}
            </ul>

            <h3>📊 Market Bias: {direction}</h3>
                        
            <br>
            <p style="font-size: 0.9em; color: #888;">Generated by CDUS Trading Bot • {current_time_est}</p>
        </body>
        </html>
        """

        # Attach both parts
        message.attach(MIMEText(body_text, "plain"))
        message.attach(MIMEText(html, "html"))

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(os.getenv("EMAIL_USER"), os.getenv("EMAIL_PASS"))
            server.send_message(message)
            print("✅ Email sent.")
    except Exception as e:
        print("❌ Email failed:", e)

# =============================
# 📊 Main
# =============================

def main():
    today = datetime.date.today()
    print(f"🧠 Using OpenAI model: {openai_model}")
    # 1. Get market data
    spx = get_spx()
    vix = get_vix()
    es = get_es()

    # 2. Scrape and classify headlines
    news = get_all_market_news()
    sentiment_score = sum(score for _, score, _, _ in news)


    # 3. Market bias and direction
    direction, reasons = estimate_direction(spx, es, sentiment_score, vix, news)

    # 4. Expected move from Market Chameleon
    #move_pts, move_msg = get_expected_move_chameleon()

    # 5. Print console version of alert (optional)
    print(f"\n📊 Pre-Market Alert for {today} Test Env")
    print(f"🔹 SPX: {spx}  🔺 VIX: {vix}  📉 ES: {es}")
    print("\n📰 Headlines:")
    for _, _, h, _ in news:
        print(f"- {h}")
    print(f"\n📊 Market Bias: {direction}")
    for r in reasons:
        print(f"- {r}")
    #print(f"\n📉 Expected Move: {move_msg}")

    # 6. Log results to CSV
    log_premarket_prediction(today, spx, es, vix, sentiment_score, direction, news)


    # 7. Send styled email
    send_email(
        subject="📊 Pre-Market Alert Test Env",
        spx=spx,
        vix=vix,
        es=es,
        news=news,
        direction=direction,
        reasons=reasons,
        move_msg="N/A", 
        to_email=os.getenv("EMAIL_TO")
    )

if __name__ == "__main__":
    main()
