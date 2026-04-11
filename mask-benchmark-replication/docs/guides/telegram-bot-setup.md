# Telegram bot setup

This guide shows one simple way to receive batch-job notifications on your phone or desktop.

## What you need

- A Telegram account
- About five minutes
- A shell where you can set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`

## Step 1: Create a bot with BotFather

1. Open Telegram.
2. Search for `@BotFather`.
3. Start a chat.
4. Send `/newbot`.
5. Follow the prompts for the bot name and username.
6. Copy the token BotFather gives you.

That token becomes `TELEGRAM_BOT_TOKEN`.

## Step 2: Send one message to your new bot

Open your bot chat and send any short message, such as `hello`.

This creates a chat record that Telegram can return through the Bot API.

## Step 3: Find your chat ID

In PowerShell, replace `<BOT_TOKEN>` with the value you received:

```powershell
Invoke-RestMethod "https://api.telegram.org/bot<BOT_TOKEN>/getUpdates"
```

Look for the `chat` object in the response. The `id` field is your `TELEGRAM_CHAT_ID`.

## Step 4: Set the variables in your shell

```powershell
$env:TELEGRAM_BOT_TOKEN = "your-bot-token"
$env:TELEGRAM_CHAT_ID = "your-chat-id"
```

## Step 5: Start the watcher

```powershell
.\scripts\run_google_batch_monitor.ps1
```

If both environment variables are present, the watcher will try to send one message per batch job when the job reaches a terminal state.

## Basic troubleshooting

- If `getUpdates` returns an empty list, send a fresh message to the bot and try again.
- If the watcher prints notification failures, double-check the token and chat id.
- If you only want console polling, skip the Telegram variables entirely.
