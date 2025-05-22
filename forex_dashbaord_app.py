#forex_dashboard_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import requests
import smtplib
from email.message import EmailMessage
from streamlit_autorefresh import st_autorefresh
import time


# === Configurable constants (use Streamlit Secrets in production)
NEWS_API_KEY = st.secrets.get("NEWS_API_KEY", "")
EMAIL_ADDRESS = st.secrets.get("EMAIL_USER", "")
EMAIL_PASSWORD = st.secrets.get("EMAIL_PASS", "")
TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID", "")

SIGNAL_LOG_FILE = 'signal_log.csv'

# --- Updated fetch_market_data ---
@st.cache_data(ttl=60)  # cache for 60 seconds to limit API calls
def fetch_market_data():
    """Fetch live EUR/USD price from ExchangeRate-API with 429 handling and caching."""
    url = 'https://open.er-api.com/v6/latest/USD'
    retries = 3
    backoff = 5  # seconds
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 429:
                # Rate limited, wait and retry
                time.sleep(backoff * (attempt + 1))
                continue
            response.raise_for_status()
            data = response.json()
            if data.get('result') == 'success' and 'rates' in data:
                eur_rate = data['rates'].get('EUR')
                if eur_rate is not None:
                    price = 1 / eur_rate  # Convert USD/EUR to EUR/USD
                    sentiment = (
                        'Bullish' if price > 1.12
                        else 'Bearish' if price < 1.10
                        else 'Neutral'
                    )
                    return round(price, 5), sentiment
            return None, None
        except Exception as e:
            if attempt == retries - 1:
                st.error(f"Error fetching market data: {e}")
            else:
                time.sleep(backoff * (attempt + 1))
    return None, None


# --- Updated fetch_top_news ---
@st.cache_data(ttl=300)  # cache for 5 minutes to reduce calls
def fetch_top_news():
    """Fetch top business news headlines using NewsAPI with 429 handling and caching."""
    if not NEWS_API_KEY:
        return ["News API key not configured."]
    url = f'https://newsapi.org/v2/top-headlines?category=business&language=en&apiKey={NEWS_API_KEY}'
    retries = 3
    backoff = 5  # seconds
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 429:
                time.sleep(backoff * (attempt + 1))
                continue
            response.raise_for_status()
            data = response.json()
            articles = data.get('articles', [])
            top_headlines = [article['title'] for article in articles[:5]]
            return top_headlines if top_headlines else ["No major news currently."]
        except Exception as e:
            if attempt == retries - 1:
                return [f"Error fetching news: {e}"]
            else:
                time.sleep(backoff * (attempt + 1))
    return ["No major news currently."]


def generate_signal(price):
    """Generate trade signal based on price thresholds."""
    if price is None:
        return None
    if price > 1.1450:
        return "SELL"
    elif price < 1.1050:
        return "BUY"
    else:
        return "NO TRADE"


def plot_chart(prices):
    """Plot live price chart in dark mode style."""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(prices, color='lime', linewidth=2)
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5)
    ax.set_title('Live Price Chart (EUR/USD)', fontsize=16, color='white')
    ax.set_xlabel('Ticks', color='white')
    ax.set_ylabel('Price', color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    st.pyplot(fig)


def log_signal(time, signal, price, stop_loss):
    """Append new signal data to the CSV log file."""
    file_exists = os.path.isfile(SIGNAL_LOG_FILE)
    with open(SIGNAL_LOG_FILE, 'a') as f:
        if not file_exists:
            f.write('Time,Signal,Price,Stop Loss\n')
        f.write(f"{time},{signal},{price},{stop_loss}\n")


def load_signal_log():
    """Load the signal log as a DataFrame or create empty if not exists."""
    if os.path.isfile(SIGNAL_LOG_FILE):
        return pd.read_csv(SIGNAL_LOG_FILE)
    else:
        return pd.DataFrame(columns=['Time', 'Signal', 'Price', 'Stop Loss'])


def send_email(subject, body):
    """Send email notification."""
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        st.warning("Email credentials not configured.")
        return
    try:
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = EMAIL_ADDRESS
        msg.set_content(body)

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
    except Exception as e:
        st.error(f"Email notification error: {e}")


def send_telegram_message(message):
    """Send Telegram notification."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        st.warning("Telegram credentials not configured.")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message}
        requests.post(url, data=payload)
    except Exception as e:
        st.error(f"Telegram notification error: {e}")


def main():
    st.set_page_config(page_title="Forex Signal Dashboard", layout="wide")
    st.title("📊 Forex Trade Signal Dashboard")

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Control Panel")
        stop_loss_offset = st.number_input(
            "Stop-Loss Offset (in pips)", min_value=5, max_value=50, value=20
        ) / 10000
        refresh_interval = st.number_input(
            "Auto-Refresh Interval (secs)", min_value=10, max_value=300, value=60
        )
        st.caption("Set refresh rate and stop-loss distance.")

        trade_mode = st.radio(
            "Trade Execution Mode",
            ["Trade Immediately", "Trade at Advised Price"],
            index=0,
            help="Choose whether to execute trades instantly at market price or wait for the advised price level."
        )

    # Auto-refresh the app
    st_autorefresh(interval=refresh_interval * 1000, key="data_refresh")

    # Fetch market data and sentiment
    price, sentiment = fetch_market_data()

    if price is None:
        st.warning("⚠️ Live market price unavailable. No signals generated.")
        return

    # Display key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📈 EUR/USD Price", f"{price:.5f}")
    with col2:
        st.metric("📊 Market Sentiment", sentiment)
    with col3:
        st.write("**📰 Top Economic News:**")
        top_news = fetch_top_news()
        for headline in top_news:
            st.write(f"- {headline}")

    st.divider()

    # Generate trade signal
    signal = generate_signal(price)
    if signal is None:
        st.info("No signal generated due to unavailable price.")
        return

    st.subheader(f"📢 Current Trade Signal: **{signal}**")

    # Load previous log to avoid duplicate signals
    if os.path.exists(SIGNAL_LOG_FILE):
        prev_log = pd.read_csv(SIGNAL_LOG_FILE)
        last_signal = prev_log['Signal'].iloc[-1] if ('Signal' in prev_log.columns and not prev_log.empty) else None
    else:
        last_signal = None

    # Log and notify if new signal generated and is actionable
    if signal != last_signal and signal in ['BUY', 'SELL']:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        stop_loss = round(price - stop_loss_offset if signal == "BUY" else price + stop_loss_offset, 5)
        log_signal(now, signal, price, stop_loss)

        st.toast(f"🚨 New {signal} Signal Triggered at {now}", icon="🚀")
        st.success(f"{signal} SIGNAL LOGGED at {now} | Price: {price:.5f} | Stop Loss: {stop_loss:.5f}")

        message = f"🚨 {signal} Signal Triggered!\nTime: {now}\nPrice: {price:.5f}\nStop Loss: {stop_loss:.5f}"
        send_email(f"Forex {signal} Signal", message)
        send_telegram_message(message)

    # Show signal log history
    st.subheader("📜 Signal Log History")
    log_df = load_signal_log()
    st.dataframe(log_df, use_container_width=True)

    csv = log_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Signal Log (CSV)", data=csv, file_name='signal_log.csv', mime='text/csv')

    st.divider()

    # Live price chart
    st.subheader("📊 Live Price Chart (Trading Platform Style)")

    if 'price_history' not in st.session_state:
        st.session_state.price_history = []

    st.session_state.price_history.append(price)
    if len(st.session_state.price_history) > 50:
        st.session_state.price_history = st.session_state.price_history[-50:]

    plot_chart(st.session_state.price_history)


if __name__ == "__main__":
    main()
