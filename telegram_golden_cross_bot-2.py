import io
import os
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf


# ============================================================
# SETTINGS
# ============================================================

BOT_TOKEN = os.environ["TG_BOT_TOKEN"]
CHAT_ID = os.environ["TG_DESTINATION"]

TELEGRAM_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

IST = ZoneInfo("Asia/Kolkata")

MARKET_START = (9, 15)
MARKET_END = (15, 30)

SCAN_INTERVAL_SECONDS = 300       # 5 minutes
BATCH_SIZE = 100                  # 100 stocks per Yahoo request

MIN_CROSS_GAP_PERCENT = 0.01

# Prevent the same stock/direction from alerting repeatedly
# during the same trading day.
sent_today = set()


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):
    response = requests.post(
        TELEGRAM_URL,
        data={
            "chat_id": CHAT_ID,
            "text": message,
        },
        timeout=20,
    )

    response.raise_for_status()


# ============================================================
# NSE STOCK LIST
# ============================================================

def get_nse_symbols():

    url = (
        "https://archives.nseindia.com/content/equities/"
        "EQUITY_L.csv"
    )

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=30,
    )

    response.raise_for_status()

    df = pd.read_csv(
        io.StringIO(response.text)
    )

    df.columns = [
        str(c).strip().upper()
        for c in df.columns
    ]

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
        .dropna()
        .unique()
        .tolist()
    )

    return sorted(symbols)


# ============================================================
# MARKET TIME
# ============================================================

def market_is_open():

    now = datetime.now(IST)

    if now.weekday() >= 5:
        return False

    current = (
        now.hour,
        now.minute,
    )

    return (
        MARKET_START
        <= current
        <= MARKET_END
    )


# ============================================================
# BATCH DOWNLOAD
# ============================================================

def download_batch(symbols):

    tickers = [
        symbol + ".NS"
        for symbol in symbols
    ]

    try:

        data = yf.download(
            tickers=tickers,
            period="5d",
            interval="5m",
            auto_adjust=False,
            group_by="ticker",
            threads=True,
            progress=False,
            timeout=30,
        )

        if data is None or data.empty:
            return None

        return data

    except Exception as e:

        print(
            "Batch download error:",
            e
        )

        return None


# ============================================================
# GET ONE STOCK FROM BATCH
# ============================================================

def get_stock_data(data, symbol):

    ticker = symbol + ".NS"

    try:

        if isinstance(data.columns, pd.MultiIndex):

            if ticker not in data.columns.get_level_values(0):
                return None

            stock = data[ticker]

        else:

            stock = data

        if "Close" not in stock.columns:
            return None

        close = (
            pd.to_numeric(
                stock["Close"],
                errors="coerce",
            )
            .dropna()
        )

        if close.empty:
            return None

        return close

    except Exception as e:

        print(
            f"{symbol}: data error: {e}"
        )

        return None


# ============================================================
# CHECK FRESH 9 / 21 EMA CROSS
# ============================================================

def check_cross(symbol, close):

    try:

        if len(close) < 30:
            return None

        # ----------------------------------------------------
        # Latest intraday price
        # ----------------------------------------------------

        current_price = float(
            close.iloc[-1]
        )

        # ----------------------------------------------------
        # Build today's evolving daily EMA
        #
        # We use completed 5-minute prices to represent the
        # current market price and calculate the evolving
        # daily EMA from the previous completed daily EMA.
        # ----------------------------------------------------

        daily = yf.download(
            symbol + ".NS",
            period="4mo",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
            timeout=20,
        )

        if daily is None or daily.empty:
            return None

        if isinstance(
            daily.columns,
            pd.MultiIndex
        ):
            daily.columns = (
                daily.columns
                .get_level_values(0)
            )

        if "Close" not in daily.columns:
            return None

        daily_close = (
            pd.to_numeric(
                daily["Close"],
                errors="coerce",
            )
            .dropna()
        )

        if len(daily_close) < 30:
            return None

        # Previous completed daily candle
        completed = daily_close.iloc[:-1]

        ema9 = completed.ewm(
            span=9,
            adjust=False,
        ).mean()

        ema21 = completed.ewm(
            span=21,
            adjust=False,
        ).mean()

        prev_ema9 = float(
            ema9.iloc[-1]
        )

        prev_ema21 = float(
            ema21.iloc[-1]
        )

        # ----------------------------------------------------
        # Evolving today's EMA
        # ----------------------------------------------------

        k9 = 2 / 10
        k21 = 2 / 22

        current_ema9 = (
            current_price * k9
            +
            prev_ema9 * (1 - k9)
        )

        current_ema21 = (
            current_price * k21
            +
            prev_ema21 * (1 - k21)
        )

        difference_percent = (
            abs(
                current_ema9
                -
                current_ema21
            )
            /
            abs(current_ema21)
            *
            100
        )

        if (
            difference_percent
            < MIN_CROSS_GAP_PERCENT
        ):
            return None

        # ----------------------------------------------------
        # FRESH CROSS
        # ----------------------------------------------------

        bullish = (
            prev_ema9 <= prev_ema21
            and
            current_ema9 > current_ema21
        )

        bearish = (
            prev_ema9 >= prev_ema21
            and
            current_ema9 < current_ema21
        )

        if not bullish and not bearish:
            return None

        direction = (
            "BULLISH"
            if bullish
            else "BEARISH"
        )

        now = datetime.now(IST)

        return {
            "symbol": symbol,
            "direction": direction,
            "price": current_price,
            "ema9": current_ema9,
            "ema21": current_ema21,
            "date": now.strftime(
                "%d-%m-%Y"
            ),
            "time": now.strftime(
                "%H:%M"
            ),
        }

    except Exception as e:

        print(
            f"{symbol}: {e}"
        )

        return None


# ============================================================
# ONE MARKET SCAN
# ============================================================

def scan_market(symbols):

    global sent_today

    today = datetime.now(IST).strftime(
        "%Y-%m-%d"
    )

    # Clear previous trading-day memory
    sent_today = {
        item
        for item in sent_today
        if item.startswith(today)
    }

    total = len(symbols)

    alerts = 0

    print()
    print(
        "================================================"
    )

    print(
        "SCAN START:",
        datetime.now(IST).strftime(
            "%H:%M:%S"
        ),
        "IST"
    )

    print(
        "Stocks:",
        total
    )

    print(
        "================================================"
    )

    for start in range(
        0,
        total,
        BATCH_SIZE
    ):

        batch = symbols[
            start:start + BATCH_SIZE
        ]

        print(
            f"Batch "
            f"{start + 1}-"
            f"{min(start + BATCH_SIZE, total)}"
        )

        data = download_batch(
            batch
        )

        if data is None:
            print(
                "Batch failed. Continuing..."
            )
            continue

        for symbol in batch:

            try:

                close = get_stock_data(
                    data,
                    symbol
                )

                if close is None:
                    continue

                result = check_cross(
                    symbol,
                    close
                )

                if result is None:
                    continue

                alert_key = (
                    f"{today}|"
                    f"{result['symbol']}|"
                    f"{result['direction']}"
                )

                if alert_key in sent_today:
                    continue

                emoji = (
                    "🟢"
                    if result["direction"]
                    == "BULLISH"
                    else "🔴"
                )

                message = (
                    f"{emoji} 9 EMA / 21 EMA "
                    f"{result['direction']} CROSS\n\n"
                    f"Stock: {result['symbol']}\n"
                    f"Time: {result['time']} IST\n"
                    f"Price: ₹{result['price']:.2f}\n"
                    f"9 EMA: ₹{result['ema9']:.2f}\n"
                    f"21 EMA: ₹{result['ema21']:.2f}\n\n"
                    f"Timeframe: 1 DAY\n"
                    f"Signal: FRESH INTRADAY CROSSOVER"
                )

                try:

                    send_telegram(
                        message
                    )

                    sent_today.add(
                        alert_key
                    )

                    alerts += 1

                    print(
                        "🚨 ALERT:",
                        result["symbol"],
                        result["direction"],
                        result["time"]
                    )

                except Exception as e:

                    print(
                        "Telegram error:",
                        e
                    )

            except Exception as e:

                print(
                    f"{symbol}: {e}"
                )

    print()
    print(
        "SCAN COMPLETE:",
        datetime.now(IST).strftime(
            "%H:%M:%S"
        ),
        "IST"
    )

    print(
        "Alerts:",
        alerts
    )

    print(
        "================================================"
    )


# ============================================================
# MAIN — CONTINUOUS MARKET SCANNER
# ============================================================

def main():

    print(
        "NSE 9/21 EMA INTRADAY SCANNER"
    )

    print(
        "Started:",
        datetime.now(IST).strftime(
            "%d-%m-%Y %H:%M:%S"
        ),
        "IST"
    )

    symbols = get_nse_symbols()

    print(
        f"Loaded {len(symbols)} NSE EQ stocks."
    )

    while True:

        now = datetime.now(IST)

        # ----------------------------------------------------
        # Weekend
        # ----------------------------------------------------

        if now.weekday() >= 5:

            print(
                "Weekend. Exiting."
            )

            break

        current = (
            now.hour,
            now.minute,
        )

        # ----------------------------------------------------
        # Before market
        # ----------------------------------------------------

        if current < MARKET_START:

            target = now.replace(
                hour=9,
                minute=15,
                second=0,
                microsecond=0,
            )

            wait_seconds = (
                target - now
            ).total_seconds()

            print(
                f"Before market. "
                f"Waiting {int(wait_seconds)} seconds."
            )

            time.sleep(
                max(
                    30,
                    wait_seconds
                )
            )

            continue

        # ----------------------------------------------------
        # After market
        # ----------------------------------------------------

        if current > MARKET_END:

            print(
                "NSE market closed."
            )

            break

        # ----------------------------------------------------
        # Scan
        # ----------------------------------------------------

        scan_market(
            symbols
        )

        # ----------------------------------------------------
        # Wait for next 5-minute scan
        # ----------------------------------------------------

        now = datetime.now(IST)

        seconds_into_minute = (
            now.second
            +
            now.microsecond / 1_000_000
        )

        minutes_to_next = (
            5
            -
            (now.minute % 5)
        )

        wait_seconds = (
            minutes_to_next * 60
            -
            seconds_into_minute
        )

        print(
            f"Next scan in "
            f"{int(wait_seconds)} seconds."
        )

        time.sleep(
            max(
                30,
                wait_seconds
            )
        )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()
