import asyncio
import logging
import logging.config
import signal

import aiohttp
from aiogram import Bot
from aiogram.client.session.aiohttp import AiohttpSession
from aiohttp import web

from .handlers import create_dispatcher
from .manage import set_webhook
from .model import ModelClient
from .settings import Settings
from .storage import Store
from .web import create_app
from .worker import Workers


def configure_logging() -> None:
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": True,
            "formatters": {"plain": {"format": "%(levelname)s %(message)s"}},
            "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "plain"}},
            "root": {"handlers": [], "level": "CRITICAL"},
            "loggers": {
                "telegram_ai": {"handlers": ["console"], "level": "INFO", "propagate": False}
            },
        }
    )


async def serve() -> None:
    settings = Settings.from_env()
    settings.validate_runtime()
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, stopped.set)
    async with Bot(
        settings.bot_token, session=AiohttpSession(timeout=settings.telegram_timeout)
    ) as bot:
        async with aiohttp.ClientSession() as session:
            model = ModelClient(session, settings)
            await model.verify_free_model()
            store = await Store.open(settings.database_url, settings, bot.id)
            try:
                if settings.database_url_unpooled:
                    migrations = await Store.open(settings.database_url_unpooled, settings, bot.id)
                    try:
                        await migrations.migrate()
                    finally:
                        await migrations.close()
                else:
                    await store.migrate()
                workers = Workers(store, model, bot, settings)
                app = create_app(settings, store, bot, create_dispatcher(), workers)
                runner = web.AppRunner(app, access_log=None, shutdown_timeout=20)
                await runner.setup()
                workers.start()
                try:
                    await web.TCPSite(runner, "0.0.0.0", settings.port).start()
                    await set_webhook(bot, settings)
                    logging.getLogger("telegram_ai").info("service_ready")
                    await stopped.wait()
                finally:
                    # Close admission before stopping workers; leases retain unfinished jobs.
                    await runner.cleanup()
                    await workers.close()
            finally:
                await store.close()


def main() -> None:
    configure_logging()
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        logging.getLogger("telegram_ai").error("startup_failed error=%s", type(exc).__name__)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
