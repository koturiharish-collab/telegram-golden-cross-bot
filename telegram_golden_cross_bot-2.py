import io
import os
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

MARKET_START = (9, 15)
MARKET_END = (15, 30)

# Keep a small minimum gap between price and crossover
# to reduce noisy near-cross signals.
MIN_CROSS_GAP_PERCENT = 0.01


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):

    response = requests.post(
        TELEGRAM_URL,
        data={
            "chat_id": CHAT_ID,
            "text": message
        },
        timeout=20
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
        timeout=30
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

    return sorted(
        df["SYMBOL"]
        .astype(str)
        .str.strip()
        .dropna()
        .unique()
        .tolist()
    )


# ============================================================
# ONE STOCK
# ============================================================

def check_stock(symbol):

    ticker = symbol + ".NS"

    try:

        # Daily history for completed candles
        daily = yf.download(
            ticker,
            period="4mo",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False
        )

        if daily.empty:
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

        close = (
            pd.to_numeric(
                daily["Close"],
                errors="coerce"
            )
            .dropna()
        )

        if len(close) < 30:
            return None

        # ----------------------------------------------------
        # Previous COMPLETED daily candle
        # ----------------------------------------------------

        completed = close.iloc[:-1]

        ema9 = completed.ewm(
            span=9,
            adjust=False
        ).mean()

        ema21 = completed.ewm(
            span=21,
            adjust=False
        ).mean()

        prev_ema9 = float(
            ema9.iloc[-1]
        )

        prev_ema21 = float(
            ema21.iloc[-1]
        )

        # ----------------------------------------------------
        # Current intraday price
        # ----------------------------------------------------

        intraday = yf.download(
            ticker,
            period="1d",
            interval="5m",
            auto_adjust=False,
            progress=False,
            threads=False
        )

        if intraday.empty:
            return None

        if isinstance(
            intraday.columns,
            pd.MultiIndex
        ):
            intraday.columns = (
                intraday.columns
                .get_level_values(0)
            )

        if "Close" not in intraday.columns:
            return None

        current_prices = (
            pd.to_numeric(
                intraday["Close"],
                errors="coerce"
            )
            .dropna()
        )

        if current_prices.empty:
            return None

        current_price = float(
            current_prices.iloc[-1]
        )

        # ----------------------------------------------------
        # Evolving TODAY daily EMA
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
            abs(current_ema9 - current_ema21)
            / current_ema21
            * 100
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
            )
        }

    except Exception as e:

        print(
            f"{symbol}: {e}"
        )

        return None


# ============================================================
# MAIN — ONE SCAN ONLY
# ============================================================

def main():

    now = datetime.now(IST)

    print(
        "NSE 9/21 EMA scanner"
    )

    print(
        "Time:",
        now.strftime(
            "%d-%m-%Y %H:%M:%S"
        ),
        "IST"
    )

    # --------------------------------------------------------
    # Market-hours check
    # --------------------------------------------------------

    if now.weekday() >= 5:

        print(
            "Weekend. Exiting."
        )

        return

    current = (
        now.hour,
        now.minute
    )

    if not (
        MARKET_START
        <= current
        <= MARKET_END
    ):

        print(
            "Outside NSE market hours."
        )

        return

    symbols = get_nse_symbols()

    print(
        f"Stocks to scan: {len(symbols)}"
    )

    alerts = 0

    # --------------------------------------------------------
    # Scan stocks
    # --------------------------------------------------------

    for number, symbol in enumerate(
        symbols,
        start=1
    ):

        print(
            f"{number}/{len(symbols)} "
            f"{symbol}"
        )

        result = check_stock(
            symbol
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

        send_telegram(
            message
        )

        alerts += 1

        print(
            "TELEGRAM ALERT:",
            result["symbol"],
            result["direction"]
        )

    print()
    print(
        f"Scan completed. "
        f"Alerts: {alerts}"
    )


if __name__ == "__main__":
    main()
