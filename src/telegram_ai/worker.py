import asyncio
import logging
from contextlib import suppress

from aiogram import Bot
from aiogram.exceptions import (
    TelegramAPIError,
    TelegramBadRequest,
    TelegramForbiddenError,
    TelegramNetworkError,
    TelegramRetryAfter,
    TelegramServerError,
    TelegramUnauthorizedError,
)
from aiogram.types import LinkPreviewOptions

from .dialogue import GENERATION_FAILED, build_messages
from .model import ModelClient, ModelError
from .settings import Settings
from .storage import Job, Store

logger = logging.getLogger("telegram_ai")


class Workers:
    def __init__(self, store: Store, model: ModelClient, bot: Bot, settings: Settings):
        self.store, self.model, self.bot, self.settings = store, model, bot, settings
        self.events = [asyncio.Event() for _ in range(settings.workers)]
        self.tasks: list[asyncio.Task] = []

    def wake(self) -> None:
        for event in self.events:
            event.set()

    def start(self) -> None:
        self.tasks = [asyncio.create_task(self._run(event)) for event in self.events]

    async def close(self) -> None:
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    async def _heartbeat(self, job: Job, work: asyncio.Task) -> None:
        try:
            while True:
                await asyncio.sleep(min(30, self.settings.lease_seconds / 3))
                if not await self.store.renew(job):
                    work.cancel()
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "lease_error update_id=%s error=%s", job["update_id"], type(exc).__name__
            )
            work.cancel()

    async def _run(self, event: asyncio.Event) -> None:
        while True:
            event.clear()
            try:
                job = await self.store.claim()
                if job is not None:
                    work = asyncio.create_task(self.process(job))
                    heartbeat = asyncio.create_task(self._heartbeat(job, work))
                    try:
                        await work
                    except asyncio.CancelledError:
                        if asyncio.current_task().cancelling():
                            raise
                    finally:
                        heartbeat.cancel()
                        await asyncio.gather(heartbeat, return_exceptions=True)
                    self.wake()
                    continue
                delay = await self.store.next_delay()
                # No database polling when idle, so Neon can scale to zero.
                if delay is None:
                    await event.wait()
                else:
                    with suppress(TimeoutError):
                        await asyncio.wait_for(event.wait(), timeout=delay)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("worker_error error=%s", type(exc).__name__)
                # Persisted leases recover jobs after database/network failures.
                with suppress(TimeoutError):
                    await asyncio.wait_for(event.wait(), timeout=5)

    async def process(self, job: Job) -> None:
        if job["user_id"] not in self.settings.allowed_user_ids:
            await self.store.finish(job, failed=True)
            return
        if job["answer"] is None:
            if not await self.store.begin_generation(job):
                if not await self.store.save_answer(job, GENERATION_FAILED, is_turn=False):
                    return
            else:
                try:
                    messages = build_messages(
                        await self.store.history(job["user_id"]),
                        job["question"],
                        max_turns=self.settings.history_turns,
                        max_chars=self.settings.history_chars,
                    )
                    answer = await self.model.complete(messages)
                except ModelError as exc:
                    logger.info("model_error update_id=%s kind=%s", job["update_id"], exc.kind)
                    if (
                        exc.retryable
                        and job["generation_attempts"] + 1 < self.settings.max_generation_attempts
                    ):
                        await self.store.defer(job, exc.retry_after)
                        return
                    if not await self.store.save_answer(job, exc.message, is_turn=False):
                        return
                else:
                    if not await self.store.save_answer(job, answer, is_turn=True):
                        return
        while True:
            try:
                outcome = await self.store.send_next(job, self._send)
            except TelegramRetryAfter as exc:
                await self.store.defer(job, max(1, exc.retry_after))
                return
            except (TelegramNetworkError, TelegramServerError, TimeoutError):
                await self.store.defer(job, min(30, 2 ** (job["send_attempts"] + 1)))
                return
            except (TelegramForbiddenError, TelegramBadRequest, TelegramUnauthorizedError) as exc:
                await self.store.finish(job, failed=True)
                logger.warning(
                    "delivery_failed update_id=%s error=%s", job["update_id"], type(exc).__name__
                )
                return
            except TelegramAPIError as exc:
                await self.store.finish(job, failed=True)
                logger.warning(
                    "delivery_failed update_id=%s error=%s", job["update_id"], type(exc).__name__
                )
                return
            if outcome == "complete":
                await self.store.finish(job)
            if outcome == "failed":
                logger.warning("delivery_exhausted update_id=%s", job["update_id"])
            if outcome != "sent":
                return

    async def _send(self, user_id: int, text: str) -> None:
        async with asyncio.timeout(self.settings.telegram_timeout + 1):
            await self.bot.send_message(
                user_id,
                text,
                parse_mode=None,
                link_preview_options=LinkPreviewOptions(is_disabled=True),
            )
