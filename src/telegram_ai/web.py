import hmac
import json
import logging

from aiogram import Bot, Dispatcher
from aiogram.types import Update
from aiohttp import web
from pydantic import ValidationError

from .dialogue import access_denied
from .settings import Settings
from .storage import Store
from .worker import Workers

logger = logging.getLogger("telegram_ai")


def create_app(
    settings: Settings,
    store: Store,
    bot: Bot,
    dispatcher: Dispatcher,
    workers: Workers,
) -> web.Application:
    app = web.Application(client_max_size=65536)

    async def health(request: web.Request) -> web.Response:
        if any(task.done() for task in workers.tasks):
            return web.json_response({"ok": False}, status=503)
        return web.json_response({"ok": True})

    async def webhook(request: web.Request) -> web.Response:
        supplied = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
        if not settings.webhook_secret or not hmac.compare_digest(
            supplied.encode(),
            settings.webhook_secret.encode(),
        ):
            return web.Response(status=403)
        try:
            update = Update.model_validate(await request.json(), context={"bot": bot})
        except (ValidationError, json.JSONDecodeError, UnicodeError, TypeError, ValueError):
            return web.Response(status=400)
        message = update.message
        if not message or message.chat.type != "private" or not message.from_user:
            return web.json_response({"ok": True})
        if message.from_user.id not in settings.allowed_user_ids:
            return web.json_response(
                {
                    "method": "sendMessage",
                    "chat_id": message.chat.id,
                    "text": access_denied(message.from_user.id),
                }
            )
        if message.chat.id != message.from_user.id:
            return web.Response(status=400)
        try:
            result = await dispatcher.feed_update(
                bot,
                update,
                store=store,
                workers=workers,
                settings=settings,
            )
        except Exception as exc:
            logger.error(
                "webhook_error update_id=%s error=%s", update.update_id, type(exc).__name__
            )
            # Do not acknowledge an update whose durable admission has failed.
            return web.Response(status=503)
        return web.json_response(result or {"ok": True})

    app.router.add_get("/healthz", health)
    app.router.add_post("/telegram/webhook", webhook)
    return app
