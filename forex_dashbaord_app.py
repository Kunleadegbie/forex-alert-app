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

# === Configurable constants (securely via Streamlit Secrets in production)
NEWS_API_KEY = st.secrets["NEWS_API_KEY"]  # Set in Streamlit Cloud Secrets
EMAIL_ADDRESS = st.secrets["email_address"]
EMAIL_PASSWORD = st.secrets["email_password"]
TELEGRAM_BOT_TOKEN = st.secrets["telegram_bot_token"]
TELEGRAM_CHAT_ID = st.secrets["telegram_chat_id"]

SIGNAL_LOG_FILE = 'signal_log.csv'


# Function to fetch market data
def fetch_market_data():
    price = np.random.uniform(1.1000, 1.1500)
    sentiment = np.random.choice(['Bullish', 'Bearish', 'Neutral'])
    return round(price, 5), sentiment


# Function to fetch top economic news
def fetch_top_news():
    try:
        response = requests.get(f'https://newsapi.org/v2/top-headlines?category=business&language=en&apiKey={NEWS_API_KEY}')
        data = response.json()
        articles = data.get('articles', [])
        top_headlines = [article['title'] for article in articles[:5]]
        return top_headlines if top_headlines else ["No major news currently."]
    except Exception as e:
        return [f"Error fetching news: {e}"]


# Signal generation logic
def generate_signal(price):
    if price > 1.1450:
        return "SELL"
    elif price < 1.1050:
        return "BUY"
    else:
        return "NO TRADE"


# Price chart function
def plot_chart(prices):
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


# Log signals to file
def log_signal(time, signal, price, stop_loss):
    file_exists = os.path.isfile(SIGNAL_LOG_FILE)
    with open(SIGNAL_LOG_FILE, 'a') as f:
        if not file_exists:
            f.write('Time,Signal,Price,Stop Loss\n')
        f.write(f"{time},{signal},{price},{stop_loss}\n")


# Load existing log
def load_signal_log():
    if os.path.isfile(SIGNAL_LOG_FILE):
        return pd.read_csv(SIGNAL_LOG_FILE)
    else:
        return pd.DataFrame(columns=['Time', 'Signal', 'Price', 'Stop Loss'])


# Email Notification
def send_email(subject, body):
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


# Telegram Notification
def send_telegram_message(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message
        }
        requests.post(url, data=payload)
    except Exception as e:
        st.error(f"Telegram notification error: {e}")


# Main app
def main():
    st.set_page_config(page_title="Forex Signal Dashboard", layout="wide")
    st.title("📊 Forex Trade Signal Dashboard")

    # === Control Panel
    with st.sidebar:
        st.header("⚙️ Control Panel")
        stop_loss_offset = st.number_input("Stop-Loss Offset (in pips)", min_value=5, max_value=50, value=20) / 10000
        refresh_interval = st.number_input("Auto-Refresh Interval (secs)", min_value=10, max_value=300, value=60)
        st.caption("Set refresh rate and stop-loss distance.")

    st_autorefresh(interval=refresh_interval * 1000, key="data_refresh")

    # Market data
    price, sentiment = fetch_market_data()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📈 Latest Market Price (EUR/USD)", f"{price:.5f}")

    with col2:
        st.metric("📊 Market Sentiment", sentiment)

    with col3:
        st.write("**📰 Top Economic News:**")
        top_news = fetch_top_news()
        for headline in top_news:
            st.write(f"- {headline}")

    st.divider()

    signal = generate_signal(price)
    st.subheader(f"📢 Current Trade Signal: **{signal}**")

    if os.path.exists(SIGNAL_LOG_FILE):
        prev_log = pd.read_csv(SIGNAL_LOG_FILE)
        if 'Signal' in prev_log.columns and not prev_log.empty:
            last_signal = prev_log['Signal'].iloc[-1]
        else:
            last_signal = None
    else:
        last_signal = None

    if signal != last_signal and signal in ['BUY', 'SELL']:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        stop_loss = round(price - stop_loss_offset if signal == "BUY" else price + stop_loss_offset, 5)
        log_signal(now, signal, price, stop_loss)
        st.toast(f"🚨 New {signal} Signal Triggered at {now}", icon="🚀")
        st.success(f"{signal} SIGNAL LOGGED at {now} | Price: {price:.5f} | Stop Loss: {stop_loss:.5f}")

        # Send notifications
        message = f"🚨 {signal} Signal Triggered!\nTime: {now}\nPrice: {price:.5f}\nStop Loss: {stop_loss:.5f}"
        send_email(f"Forex {signal} Signal", message)
        send_telegram_message(message)

    st.subheader("📜 Signal Log History")
    log_df = load_signal_log()
    st.dataframe(log_df, use_container_width=True)

    csv = log_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Signal Log (CSV)", data=csv, file_name='signal_log.csv', mime='text/csv')

    st.divider()

    st.subheader("📊 Live Price Chart (Trading Platform Style)")

    if 'price_history' not in st.session_state:
        st.session_state.price_history = []

    st.session_state.price_history.append(price)
    if len(st.session_state.price_history) > 50:
        st.session_state.price_history = st.session_state.price_history[-50:]

    plot_chart(st.session_state.price_history)


if __name__ == "__main__":
    main()
