"""
Telegram Golden Cross Forwarder
Reads posts from a Telegram source channel and forwards only
fresh-looking Golden Cross messages to your private chat.

IMPORTANT:
- Put credentials in environment variables, not in this file.
- This filters Telegram MESSAGE TEXT; it does not calculate a Golden Cross
  from market prices.
"""

import os
import re
from telethon import TelegramClient, events

API_ID = int(os.environ["TG_API_ID"])
API_HASH = os.environ["TG_API_HASH"]
BOT_TOKEN = os.environ["TG_BOT_TOKEN"]

SOURCE = os.environ.get("TG_SOURCE", "@haris8977")
DESTINATION = os.environ["TG_DESTINATION"]  # your personal chat/user ID

client = TelegramClient("golden_cross_bot", API_ID, API_HASH).start(
    bot_token=BOT_TOKEN
)

GOLDEN = re.compile(
    r"\b(golden\s*cross|50\s*(?:day|dma|sma)?\s*(?:above|cross(?:es|ed)?\s*above)\s*200|"
    r"50\s*(?:dma|sma)\s*(?:cross(?:es|ed)?\s*)?(?:above|over)\s*200)\b",
    re.I,
)

seen = set()

@client.on(events.NewMessage(chats=SOURCE))
async def handler(event):
    text = event.raw_text or ""
    if not GOLDEN.search(text):
        return

    # Prevent duplicate forwarding of the same Telegram message.
    key = (event.chat_id, event.id)
    if key in seen:
        return
    seen.add(key)

    await client.send_message(
        DESTINATION,
        "🟢 FRESH GOLDEN CROSS\n\n" + text
    )

print("Golden Cross Telegram bot is running...")
client.run_until_disconnected()
