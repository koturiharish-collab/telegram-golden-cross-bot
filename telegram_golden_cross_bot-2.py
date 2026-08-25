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

TELEGRAM_URL = (
    f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
)

IST = ZoneInfo("Asia/Kolkata")

MARKET_START_HOUR = 9
MARKET_START_MINUTE = 45

MARKET_END_HOUR = 15
MARKET_END_MINUTE = 30

CHECK_INTERVAL_SECONDS = 30 * 60

BATCH_SIZE = 100

MIN_CROSS_GAP_PERCENT = 0.01


# Keeps the EMA relationship from the previous scan.
previous_state = {}

# Prevents duplicate alerts.
sent_alerts = set()


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
# NSE SYMBOLS
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

def market_time_ok():

    now = datetime.now(IST)

    if now.weekday() >= 5:
        return False

    current_minutes = (
        now.hour * 60
        + now.minute
    )

    start_minutes = (
        MARKET_START_HOUR * 60
        + MARKET_START_MINUTE
    )

    end_minutes = (
        MARKET_END_HOUR * 60
        + MARKET_END_MINUTE
    )

    return (
        start_minutes
        <= current_minutes
        <= end_minutes
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
# DAILY DATA — BATCH
# ============================================================

def download_daily_batch(symbols):

    tickers = [
        symbol + ".NS"
        for symbol in symbols
    ]

    try:

        data = yf.download(
            tickers=tickers,
            period="4mo",
            interval="1d",
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
            "Daily batch error:",
            e
        )

        return None


# ============================================================
# EXTRACT CLOSE
# ============================================================

def extract_close(data, symbol):

    ticker = symbol + ".NS"

    try:

        if isinstance(
            data.columns,
            pd.MultiIndex
        ):

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

    except Exception:

        return None


# ============================================================
# CALCULATE CURRENT DAILY EMA
# ============================================================

def calculate_emas(
    daily_close,
    current_price
):

    if daily_close is None:
        return None

    if len(daily_close) < 30:
        return None

    # Last completed daily candle.
    completed = daily_close.iloc[:-1]

    if len(completed) < 30:
        return None

    ema9 = completed.ewm(
        span=9,
        adjust=False
    ).mean()

    ema21 = completed.ewm(
        span=21,
        adjust=False
    ).mean()

    previous_ema9 = float(
        ema9.iloc[-1]
    )

    previous_ema21 = float(
        ema21.iloc[-1]
    )

    # Today's evolving daily EMA.
    k9 = 2 / (9 + 1)
    k21 = 2 / (21 + 1)

    current_ema9 = (
        current_price * k9
        +
        previous_ema9 * (1 - k9)
    )

    current_ema21 = (
        current_price * k21
        +
        previous_ema21 * (1 - k21)
    )

    return (
        previous_ema9,
        previous_ema21,
        current_ema9,
        current_ema21,
    )


# ============================================================
# PROCESS ONE STOCK
# ============================================================

def process_stock(
    symbol,
    intraday_data,
    daily_data
):

    global previous_state

    try:

        intraday_close = extract_close(
            intraday_data,
            symbol
        )

        daily_close = extract_close(
            daily_data,
            symbol
        )

        if intraday_close is None:
            return None

        if daily_close is None:
            return None

        current_price = float(
            intraday_close.iloc[-1]
        )

        emas = calculate_emas(
            daily_close,
            current_price
        )

        if emas is None:
            return None

        (
            previous_ema9,
            previous_ema21,
            current_ema9,
            current_ema21,
        ) = emas

        gap_percent = (
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

        if gap_percent < MIN_CROSS_GAP_PERCENT:

            relationship = "NEAR"

        elif current_ema9 > current_ema21:

            relationship = "ABOVE"

        else:

            relationship = "BELOW"

        old_relationship = previous_state.get(
            symbol
        )

        # First observation of the day:
        # establish state, but DON'T alert.
        if old_relationship is None:

            previous_state[symbol] = (
                relationship
            )

            return None

        # ----------------------------------------------------
        # FRESH CROSS
        # ----------------------------------------------------

        bullish = (
            old_relationship == "BELOW"
            and
            relationship == "ABOVE"
        )

        bearish = (
            old_relationship == "ABOVE"
            and
            relationship == "BELOW"
        )

        # Update state every scan.
        previous_state[symbol] = (
            relationship
        )

        if not bullish and not bearish:
            return None

        direction = (
            "BULLISH"
            if bullish
            else "BEARISH"
        )

        now = datetime.now(IST)

        date_key = now.strftime(
            "%Y-%m-%d"
        )

        alert_key = (
            f"{date_key}|"
            f"{symbol}|"
            f"{direction}"
        )

        if alert_key in sent_alerts:
            return None

        sent_alerts.add(
            alert_key
        )

        return {
            "symbol": symbol,
            "direction": direction,
            "price": current_price,
            "ema9": current_ema9,
            "ema21": current_ema21,
            "time": now.strftime(
                "%H:%M:%S"
            ),
        }

    except Exception as e:

        print(
            f"{symbol}: {e}"
        )

        return None


# ============================================================
# ONE COMPLETE SCAN
# ============================================================

def scan_market(symbols):

    print()
    print(
        "=============================================="
    )

    print(
        "SCAN:",
        datetime.now(IST).strftime(
            "%d-%m-%Y %H:%M:%S"
        ),
        "IST"
    )

    print(
        "Stocks:",
        len(symbols)
    )

    print(
        "=============================================="
    )

    alerts = 0

    for start in range(
        0,
        len(symbols),
        BATCH_SIZE
    ):

        batch = symbols[
            start:start + BATCH_SIZE
        ]

        end = min(
            start + BATCH_SIZE,
            len(symbols)
        )

        print(
            f"Batch {start + 1}-{end}"
        )

        # One batch for intraday.
        intraday_data = download_batch(
            batch
        )

        # One batch for daily history.
        daily_data = download_daily_batch(
            batch
        )

        if (
            intraday_data is None
            or
            daily_data is None
        ):

            print(
                "Batch data unavailable."
            )

            continue

        for symbol in batch:

            result = process_stock(
                symbol,
                intraday_data,
                daily_data
            )

            if result is None:
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

                alerts += 1

                print(
                    "🚨 TELEGRAM ALERT:",
                    result["symbol"],
                    result["direction"]
                )

            except Exception as e:

                print(
                    "Telegram error:",
                    e
                )

    print()
    print(
        "SCAN COMPLETE"
    )

    print(
        "Alerts:",
        alerts
    )

    print(
        "Next scan: approximately 30 minutes"
    )

    print(
        "=============================================="
    )


# ============================================================
# MAIN
# ============================================================

def main():

    global previous_state
    global sent_alerts

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

    current_date = (
        datetime.now(IST).date()
    )

    # --------------------------------------------------------
    # Wait until 9:45 AM
    # --------------------------------------------------------

    while True:

        now = datetime.now(IST)

        if now.date() != current_date:
            current_date = now.date()
            previous_state.clear()
            sent_alerts.clear()

        if now.weekday() >= 5:

            print(
                "Weekend. Exiting."
            )

            return

        current_minutes = (
            now.hour * 60
            + now.minute
        )

        start_minutes = (
            MARKET_START_HOUR * 60
            + MARKET_START_MINUTE
        )

        end_minutes = (
            MARKET_END_HOUR * 60
            + MARKET_END_MINUTE
        )

        if current_minutes < start_minutes:

            target = now.replace(
                hour=MARKET_START_HOUR,
                minute=MARKET_START_MINUTE,
                second=0,
                microsecond=0,
            )

            wait_seconds = (
                target - now
            ).total_seconds()

            print(
                "Waiting until 09:45 IST..."
            )

            time.sleep(
                max(
                    30,
                    wait_seconds
                )
            )

            continue

        if current_minutes > end_minutes:

            print(
                "Market closed. Exiting."
            )

            return

        break

    # --------------------------------------------------------
    # Continuous 30-minute scanning
    # --------------------------------------------------------

    while True:

        now = datetime.now(IST)

        current_minutes = (
            now.hour * 60
            + now.minute
        )

        end_minutes = (
            MARKET_END_HOUR * 60
            + MARKET_END_MINUTE
        )

        if current_minutes > end_minutes:

            print(
                "NSE market closed."
            )

            break

        scan_market(
            symbols
        )

        # ----------------------------------------------------
        # Wait exactly approximately 30 minutes from the
        # completion of this scan.
        # ----------------------------------------------------

        print(
            "Sleeping 30 minutes..."
        )

        time.sleep(
            CHECK_INTERVAL_SECONDS
        )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()
