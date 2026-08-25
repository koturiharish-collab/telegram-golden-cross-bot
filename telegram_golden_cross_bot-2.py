import io
import os
import time
from datetime import datetime
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

# Check every 5 minutes
CHECK_INTERVAL_SECONDS = 300

# Number of symbols per Yahoo request
BATCH_SIZE = 75

# Ignore extremely tiny EMA differences
MIN_CROSS_GAP_PERCENT = 0.01

# Keep track of alerts during this run
ALERTED = set()


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):

    try:
        response = requests.post(
            TELEGRAM_URL,
            data={
                "chat_id": CHAT_ID,
                "text": message
            },
            timeout=15
        )

        response.raise_for_status()

        print("Telegram alert sent")

    except Exception as e:

        print("Telegram error:", e)


# ============================================================
# NSE SYMBOL LIST
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
        timeout=20
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
# BATCH DOWNLOAD
# ============================================================

def download_batch(symbols, interval, period):

    tickers = [
        symbol + ".NS"
        for symbol in symbols
    ]

    try:

        data = yf.download(
            tickers=tickers,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=True,
            group_by="ticker",
            timeout=15
        )

        return data

    except Exception as e:

        print(
            f"Yahoo batch error ({interval}):",
            e
        )

        return pd.DataFrame()


# ============================================================
# GET SERIES FROM BATCH DATA
# ============================================================

def get_close_series(data, symbol):

    ticker = symbol + ".NS"

    try:

        if data.empty:
            return None

        # MultiTicker download
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
                errors="coerce"
            )
            .dropna()
        )

        if close.empty:
            return None

        return close

    except Exception as e:

        print(
            f"{symbol}: data extraction error:",
            e
        )

        return None


# ============================================================
# PREVIOUS COMPLETED DAILY EMA
# ============================================================

def get_previous_emas(daily_data, symbol):

    close = get_close_series(
        daily_data,
        symbol
    )

    if close is None:
        return None

    # Remove today's incomplete candle.
    completed = close.iloc[:-1]

    if len(completed) < 30:
        return None

    ema9 = (
        completed
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    ema21 = (
        completed
        .ewm(
            span=21,
            adjust=False
        )
        .mean()
    )

    return (
        float(ema9.iloc[-1]),
        float(ema21.iloc[-1])
    )


# ============================================================
# FIND FRESH INTRADAY CROSS
# ============================================================

def find_cross(
    symbol,
    daily_data,
    intraday_data
):

    previous = get_previous_emas(
        daily_data,
        symbol
    )

    if previous is None:
        return None

    prev_ema9, prev_ema21 = previous

    prices = get_close_series(
        intraday_data,
        symbol
    )

    if prices is None:
        return None

    if prices.empty:
        return None

    # Only today's candles
    prices = prices.tail(100)

    k9 = 2 / 10
    k21 = 2 / 22

    ema9 = prev_ema9
    ema21 = prev_ema21

    previous_difference = ema9 - ema21

    for timestamp, price in prices.items():

        try:
            price = float(price)
        except Exception:
            continue

        # Evolving today's daily EMA
        ema9 = (
            price * k9
            +
            ema9 * (1 - k9)
        )

        ema21 = (
            price * k21
            +
            ema21 * (1 - k21)
        )

        difference = ema9 - ema21

        difference_percent = (
            abs(difference)
            / abs(ema21)
            * 100
        )

        if difference_percent < MIN_CROSS_GAP_PERCENT:
            previous_difference = difference
            continue

        bullish = (
            previous_difference <= 0
            and
            difference > 0
        )

        bearish = (
            previous_difference >= 0
            and
            difference < 0
        )

        if bullish or bearish:

            direction = (
                "BULLISH"
                if bullish
                else "BEARISH"
            )

            # Convert Yahoo timestamp to IST
            try:

                if timestamp.tzinfo is None:
                    signal_time = timestamp.replace(
                        tzinfo=ZoneInfo("UTC")
                    ).astimezone(IST)

                else:
                    signal_time = timestamp.astimezone(IST)

            except Exception:

                signal_time = datetime.now(IST)

            return {
                "symbol": symbol,
                "direction": direction,
                "price": price,
                "ema9": ema9,
                "ema21": ema21,
                "time": signal_time.strftime(
                    "%H:%M"
                )
            }

        previous_difference = difference

    return None


# ============================================================
# SEND ALERT
# ============================================================

def send_alert(result):

    today = datetime.now(IST).strftime(
        "%d-%m-%Y"
    )

    key = (
        today,
        result["symbol"],
        result["direction"]
    )

    # Prevent duplicate alerts
    if key in ALERTED:

        return

    ALERTED.add(key)

    emoji = (
        "🟢"
        if result["direction"] == "BULLISH"
        else "🔴"
    )

    message = (
        f"{emoji} 9 EMA / 21 EMA "
        f"{result['direction']} CROSS\n\n"

        f"Stock: {result['symbol']}\n"
        f"Date: {today}\n"
        f"Time: {result['time']} IST\n"
        f"Price: ₹{result['price']:.2f}\n\n"

        f"9 EMA: ₹{result['ema9']:.2f}\n"
        f"21 EMA: ₹{result['ema21']:.2f}\n\n"

        f"Timeframe: 1 DAY\n"
        f"Signal: FRESH INTRADAY CROSSOVER"
    )

    send_telegram(message)

    print(
        "ALERT:",
        result["symbol"],
        result["direction"],
        result["time"]
    )


# ============================================================
# ONE MONITORING CYCLE
# ============================================================

def run_scan(symbols):

    print()
    print("=" * 60)

    now = datetime.now(IST)

    print(
        "SCAN:",
        now.strftime(
            "%d-%m-%Y %H:%M:%S"
        ),
        "IST"
    )

    print(
        "Stocks:",
        len(symbols)
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Download daily data in batches
    # --------------------------------------------------------

    daily_batches = {}

    for start in range(
        0,
        len(symbols),
        BATCH_SIZE
    ):

        batch = symbols[
            start:start + BATCH_SIZE
        ]

        print(
            f"Daily data "
            f"{start + 1}-"
            f"{min(start + BATCH_SIZE, len(symbols))}"
        )

        data = download_batch(
            batch,
            "1d",
            "4mo"
        )

        daily_batches[start] = (
            batch,
            data
        )

    # --------------------------------------------------------
    # Download today's 5-minute data
    # --------------------------------------------------------

    for start in range(
        0,
        len(symbols),
        BATCH_SIZE
    ):

        batch = symbols[
            start:start + BATCH_SIZE
        ]

        daily_batch, daily_data = (
            daily_batches[start]
        )

        print(
            f"Intraday data "
            f"{start + 1}-"
            f"{min(start + BATCH_SIZE, len(symbols))}"
        )

        intraday_data = download_batch(
            batch,
            "5m",
            "1d"
        )

        if intraday_data.empty:
            continue

        # ----------------------------------------------------
        # Check each stock
        # ----------------------------------------------------

        for symbol in batch:

            try:

                result = find_cross(
                    symbol,
                    daily_data,
                    intraday_data
                )

                if result is not None:

                    send_alert(result)

            except Exception as e:

                print(
                    f"{symbol}: scan error:",
                    e
                )


# ============================================================
# MARKET STATUS
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
        MARKET_START
        <= current
        <= MARKET_END
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("NSE 9 EMA / 21 EMA")
    print("INTRADAY FRESH CROSSOVER MONITOR")
    print()

    symbols = get_nse_symbols()

    print(
        f"Loaded {len(symbols)} NSE EQ stocks"
    )

    # --------------------------------------------------------
    # Continuous monitoring
    # --------------------------------------------------------

    while True:

        now = datetime.now(IST)

        if not market_is_open():

            print(
                "Market closed:",
                now.strftime(
                    "%H:%M:%S"
                ),
                "IST"
            )

            break

        try:

            run_scan(symbols)

        except Exception as e:

            print(
                "SCAN ERROR:",
                e
            )

        now = datetime.now(IST)

        if (
            now.hour > MARKET_END[0]
            or
            (
                now.hour == MARKET_END[0]
                and
                now.minute >= MARKET_END[1]
            )
        ):

            print(
                "NSE market closed."
            )

            break

        print()
        print(
            "Waiting 5 minutes..."
        )

        time.sleep(
            CHECK_INTERVAL_SECONDS
        )

    print()
    print(
        "Scanner stopped."
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()
