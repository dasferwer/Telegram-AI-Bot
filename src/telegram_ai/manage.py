import argparse
import asyncio

import aiohttp
from aiogram import Bot
from aiogram.client.session.aiohttp import AiohttpSession

from .model import ModelClient
from .settings import Settings


async def set_webhook(bot: Bot, settings: Settings) -> None:
    await bot.set_webhook(
        f"{settings.webhook_base_url}/telegram/webhook",
        secret_token=settings.webhook_secret,
        max_connections=1,
        allowed_updates=["message"],
        drop_pending_updates=False,
    )


async def execute(command: str) -> None:
    settings = Settings.from_env()
    settings.validate_runtime()
    if command == "check-model":
        async with aiohttp.ClientSession() as session:
            client = ModelClient(session, settings)
            await client.verify_free_model()
            await client.complete([{"role": "user", "content": "Ответь одним словом: работает"}])
            print(
                "Реальный запрос к настроенной модели выполнен успешно. Использован 1 запрос квоты."
            )
        return
    async with Bot(settings.bot_token, session=AiohttpSession(timeout=15)) as bot:
        if command == "set-webhook":
            await set_webhook(bot, settings)
            print("Webhook установлен; ожидающие обновления сохранены.")
        else:
            info = await bot.get_webhook_info()
            print(
                f"URL: {info.url}; ожидающих обновлений: {info.pending_update_count}; "
                f"есть ошибка доставки: {bool(info.last_error_date)}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["check-model", "set-webhook", "webhook-info"])
    args = parser.parse_args()
    try:
        asyncio.run(execute(args.command))
    except Exception as exc:
        # Provider and database exceptions may contain secrets; only print their class.
        print(
            f"Проверка не выполнена: {type(exc).__name__}. "
            "Проверьте настройки и доступность сервиса."
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
