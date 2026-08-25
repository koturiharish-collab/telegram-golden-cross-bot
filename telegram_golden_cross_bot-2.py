"""
NSE 9 EMA / 21 EMA INTRADAY CROSSOVER SCANNER

- NSE equity stocks
- EMA calculation is DAILY
- Current trading price is used to detect an intraday crossover
- Checks during NSE market hours
- Sends Telegram alert only for a FRESH crossover
- Prevents duplicate alerts for the same stock/day/direction
"""

import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf


# ============================================================
# TELEGRAM
# ============================================================

BOT_TOKEN = os.environ["TG_BOT_TOKEN"]
CHAT_ID = os.environ["TG_DESTINATION"]

TELEGRAM_URL = (
    f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
)


# ============================================================
# NSE
# ============================================================

NSE_LIST_URL = (
    "https://archives.nseindia.com/content/equities/"
    "EQUITY_L.csv"
)

IST = ZoneInfo("Asia/Kolkata")


# ============================================================
# SETTINGS
# ============================================================

CHECK_INTERVAL_SECONDS = 300       # 5 minutes
MAX_SYMBOLS_PER_BATCH = 75

# Yahoo Finance can occasionally fail for some symbols.
# We continue scanning the remaining symbols.

MARKET_START = (9, 15)
MARKET_END = (15, 30)


# ============================================================
# DUPLICATE CONTROL
# ============================================================

sent_alerts = set()


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):

    payload = {
        "chat_id": CHAT_ID,
        "text": message
    }

    response = requests.post(
        TELEGRAM_URL,
        data=payload,
        timeout=20
    )

    response.raise_for_status()


# ============================================================
# GET NSE SYMBOLS
# ============================================================

def get_nse_symbols():

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(
        NSE_LIST_URL,
        headers=headers,
        timeout=30
    )

    response.raise_for_status()

    from io import StringIO

    df = pd.read_csv(StringIO(response.text))

    df.columns = [
        str(c).strip().upper()
        for c in df.columns
    ]

    # Only equity series
    if "SERIES" in df.columns:
        df = df[
            df["SERIES"]
            .astype(str)
            .str.upper()
            .eq("EQ")
        ]

    symbols = (
        df["SYMBOL"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    symbols = [
        s for s in symbols
        if s and s != "NAN"
    ]

    return sorted(set(symbols))


# ============================================================
# DAILY EMA
# ============================================================

def calculate_previous_emas(daily):

    if daily is None or daily.empty:
        return None

    if "Close" not in daily.columns:
        return None

    close = daily["Close"].dropna()

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

    return (
        float(ema9.iloc[-1]),
        float(ema21.iloc[-1])
    )


# ============================================================
# CHECK ONE BATCH
# ============================================================

def scan_batch(symbols):

    tickers = [
        f"{symbol}.NS"
        for symbol in symbols
    ]

    try:

        # Daily history.
        # We need enough completed candles to calculate EMA21.
        daily = yf.download(
            tickers=tickers,
            period="3mo",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True
        )

    except Exception as e:

        print(
            "Daily download error:",
            e
        )

        return []

    results = []

    # --------------------------------------------------------
    # Current intraday price
    # --------------------------------------------------------

    try:

        intraday = yf.download(
            tickers=tickers,
            period="1d",
            interval="5m",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True
        )

    except Exception as e:

        print(
            "Intraday download error:",
            e
        )

        return []

    # --------------------------------------------------------
    # Process each stock
    # --------------------------------------------------------

    for symbol in symbols:

        ticker = f"{symbol}.NS"

        try:

            # ==============================
            # DAILY DATA
            # ==============================

            if len(tickers) == 1:

                d = daily.copy()

            else:

                if ticker not in daily.columns.levels[0]:
                    continue

                d = daily[ticker].copy()

            d = d.dropna(
                subset=["Close"]
            )

            if len(d) < 30:
                continue

            # Last COMPLETED daily candle
            completed_daily = d.iloc[:-1]

            previous_emas = calculate_previous_emas(
                completed_daily
            )

            if previous_emas is None:
                continue

            previous_ema9, previous_ema21 = previous_emas

            # ==============================
            # CURRENT PRICE
            # ==============================

            if len(tickers) == 1:

                i = intraday.copy()

            else:

                if ticker not in intraday.columns.levels[0]:
                    continue

                i = intraday[ticker].copy()

            i = i.dropna(
                subset=["Close"]
            )

            if i.empty:
                continue

            current_price = float(
                i["Close"].iloc[-1]
            )

            # ==============================
            # CURRENT DAILY EMA
            # ==============================

            # Daily EMA constants
            k9 = 2 / (9 + 1)
            k21 = 2 / (21 + 1)

            current_ema9 = (
                current_price * k9
                + previous_ema9 * (1 - k9)
            )

            current_ema21 = (
                current_price * k21
                + previous_ema21 * (1 - k21)
            )

            # ==============================
            # CROSSOVER
            # ==============================

            bullish = (
                previous_ema9 <= previous_ema21
                and current_ema9 > current_ema21
            )

            bearish = (
                previous_ema9 >= previous_ema21
                and current_ema9 < current_ema21
            )

            if not bullish and not bearish:
                continue

            direction = (
                "BULLISH"
                if bullish
                else "BEARISH"
            )

            today = datetime.now(
                IST
            ).strftime("%d-%m-%Y")

            current_time = datetime.now(
                IST
            ).strftime("%H:%M")

            # One alert per stock/direction/day
            alert_key = (
                today,
                symbol,
                direction
            )

            if alert_key in sent_alerts:
                continue

            sent_alerts.add(alert_key)

            results.append({
                "symbol": symbol,
                "direction": direction,
                "price": current_price,
                "ema9": current_ema9,
                "ema21": current_ema21,
                "date": today,
                "time": current_time
            })

        except Exception as e:

            print(
                f"{symbol}: error -> {e}"
            )

    return results


# ============================================================
# SEND ALERT
# ============================================================

def send_alert(result):

    emoji = (
        "🟢"
        if result["direction"] == "BULLISH"
        else "🔴"
    )

    message = (
        f"{emoji} 9 EMA / 21 EMA "
        f"{result['direction']} CROSS\n\n"
        f"Stock: {result['symbol']}\n"
        f"Date: {result['date']}\n"
        f"Time: {result['time']} IST\n"
        f"Price: ₹{result['price']:.2f}\n\n"
        f"9 EMA: ₹{result['ema9']:.2f}\n"
        f"21 EMA: ₹{result['ema21']:.2f}\n\n"
        f"Timeframe: 1 DAY\n"
        f"Signal: FRESH INTRADAY CROSSOVER"
    )

    send_telegram(message)

    print(
        f"ALERT SENT: "
        f"{result['symbol']} "
        f"{result['direction']}"
    )


# ============================================================
# MARKET TIME
# ============================================================

def market_is_open():

    now = datetime.now(IST)

    if now.weekday() >= 5:
        return False

    current = (
        now.hour,
        now.minute
    )

    return (
        MARKET_START <= current <= MARKET_END
    )


# ============================================================
# MAIN SCANNER
# ============================================================

def main():

    print("=" * 50)
    print("NSE 9 EMA / 21 EMA INTRADAY SCANNER")
    print("=" * 50)

    symbols = get_nse_symbols()

    print(
        f"NSE stocks found: {len(symbols)}"
    )

    print(
        "Checking every 5 minutes during market hours."
    )

    while True:

        now = datetime.now(IST)

        print(
            f"\nScan time: "
            f"{now.strftime('%d-%m-%Y %H:%M:%S')} IST"
        )

        if not market_is_open():

            print(
                "Market closed. Waiting..."
            )

            time.sleep(60)

            continue

        total = len(symbols)

        for start in range(
            0,
            total,
            MAX_SYMBOLS_PER_BATCH
        ):

            batch = symbols[
                start:
                start + MAX_SYMBOLS_PER_BATCH
            ]

            print(
                f"Checking "
                f"{start + 1}-"
                f"{min(start + len(batch), total)}"
                f"/{total}"
            )

            results = scan_batch(
                batch
            )

            for result in results:

                try:
                    send_alert(result)

                except Exception as e:

                    print(
                        "Telegram error:",
                        e
                    )

        print(
            "Scan complete."
        )

        # Wait five minutes before next scan.
        time.sleep(
            CHECK_INTERVAL_SECONDS
        )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()
