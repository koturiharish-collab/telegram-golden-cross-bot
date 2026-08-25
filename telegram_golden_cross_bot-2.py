"""
NSE Daily 9 EMA / 21 EMA Crossover Scanner
Scans NSE equity (EQ series) stocks and sends fresh daily
9 EMA / 21 EMA crossover alerts to Telegram.
"""

import io
import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests
import yfinance as yf


# =========================
# TELEGRAM SETTINGS
# =========================

BOT_TOKEN = os.environ["TG_BOT_TOKEN"]
CHAT_ID = os.environ["TG_DESTINATION"]

TELEGRAM_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"


# =========================
# NSE STOCK LIST
# =========================

NSE_LIST_URL = (
    "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
)


def get_nse_symbols():
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
        )
    }

    response = requests.get(
        NSE_LIST_URL,
        headers=headers,
        timeout=30
    )
    response.raise_for_status()

    df = pd.read_csv(io.BytesIO(response.content))

    df.columns = [str(c).strip().upper() for c in df.columns]

    # Main NSE equity series only
    if "SERIES" in df.columns:
        df = df[df["SERIES"].astype(str).str.upper().eq("EQ")]

    symbols = (
        df["SYMBOL"]
        .astype(str)
        .str.strip()
        .dropna()
        .unique()
        .tolist()
    )

    return symbols


# =========================
# TELEGRAM
# =========================

def send_telegram(message):
    response = requests.post(
        TELEGRAM_URL,
        data={
            "chat_id": CHAT_ID,
            "text": message,
            "disable_web_page_preview": True,
        },
        timeout=20,
    )

    response.raise_for_status()


# =========================
# EMA CROSSOVER CHECK
# =========================

def check_stock(symbol):
    ticker = symbol + ".NS"

    try:
        data = yf.download(
            ticker,
            period="4mo",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if data is None or data.empty:
            return None

        # Handle yfinance MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if "Close" not in data.columns:
            return None

        close = pd.to_numeric(
            data["Close"],
            errors="coerce"
        ).dropna()

        if len(close) < 30:
            return None

        ema9 = close.ewm(
            span=9,
            adjust=False
        ).mean()

        ema21 = close.ewm(
            span=21,
            adjust=False
        ).mean()

        # Latest COMPLETED daily candle
        prev9 = float(ema9.iloc[-2])
        prev21 = float(ema21.iloc[-2])

        curr9 = float(ema9.iloc[-1])
        curr21 = float(ema21.iloc[-1])

        price = float(close.iloc[-1])

        candle_date = close.index[-1]

        # Convert timestamp to readable date
        try:
            candle_date = candle_date.strftime("%d-%m-%Y")
        except Exception:
            candle_date = str(candle_date)

        # FRESH BULLISH CROSS
        if prev9 <= prev21 and curr9 > curr21:
            return {
                "type": "BULLISH",
                "symbol": symbol,
                "price": price,
                "ema9": curr9,
                "ema21": curr21,
                "date": candle_date,
            }

        # FRESH BEARISH CROSS
        if prev9 >= prev21 and curr9 < curr21:
            return {
                "type": "BEARISH",
                "symbol": symbol,
                "price": price,
                "ema9": curr9,
                "ema21": curr21,
                "date": candle_date,
            }

        return None

    except Exception as e:
        print(f"{symbol}: ERROR - {e}")
        return None


# =========================
# MAIN SCANNER
# =========================

def main():

    print("======================================")
    print("NSE 9 EMA / 21 EMA DAILY SCANNER")
    print("======================================")

    symbols = get_nse_symbols()

    print(f"NSE stocks found: {len(symbols)}")

    bullish = []
    bearish = []

    # Scan in batches to reduce load
    for number, symbol in enumerate(symbols, start=1):

        print(
            f"[{number}/{len(symbols)}] "
            f"Checking {symbol}"
        )

        result = check_stock(symbol)

        if result:

            if result["type"] == "BULLISH":
                bullish.append(result)

            elif result["type"] == "BEARISH":
                bearish.append(result)

        # Small pause to avoid excessive requests
        time.sleep(0.15)

    # =========================
    # SEND BULLISH ALERTS
    # =========================

    for signal in bullish:

        message = (
            "🟢 9 EMA / 21 EMA BULLISH CROSS\n\n"
            f"Stock: {signal['symbol']}\n"
            f"Date: {signal['date']}\n"
            f"Price: ₹{signal['price']:.2f}\n"
            f"9 EMA: ₹{signal['ema9']:.2f}\n"
            f"21 EMA: ₹{signal['ema21']:.2f}\n\n"
            "Timeframe: 1 DAY\n"
            "Signal: FRESH CROSSOVER"
        )

        send_telegram(message)

    # =========================
    # SEND BEARISH ALERTS
    # =========================

    for signal in bearish:

        message = (
            "🔴 9 EMA / 21 EMA BEARISH CROSS\n\n"
            f"Stock: {signal['symbol']}\n"
            f"Date: {signal['date']}\n"
            f"Price: ₹{signal['price']:.2f}\n"
            f"9 EMA: ₹{signal['ema9']:.2f}\n"
            f"21 EMA: ₹{signal['ema21']:.2f}\n\n"
            "Timeframe: 1 DAY\n"
            "Signal: FRESH CROSSOVER"
        )

        send_telegram(message)

    print()
    print("======================================")
    print(f"Bullish crosses: {len(bullish)}")
    print(f"Bearish crosses: {len(bearish)}")
    print("Scanner completed.")
    print("======================================")


if __name__ == "__main__":
    main()
