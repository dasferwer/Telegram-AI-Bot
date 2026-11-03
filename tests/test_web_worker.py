import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.exceptions import TelegramForbiddenError, TelegramNetworkError
from aiogram.methods import SendMessage
from aiohttp.test_utils import TestClient, TestServer

from telegram_ai.dialogue import NEW, ONLY_TEXT
from telegram_ai.handlers import create_dispatcher
from telegram_ai.model import ModelError
from telegram_ai.web import create_app
from telegram_ai.worker import Workers

SECRET = "a" * 40
# These tests also run against a remote Neon branch, including connection cold starts.
ASYNC_TEST_TIMEOUT = 10


def update(number=1, user_id=101, text="привет", chat_type="private"):
    message = {
        "message_id": number,
        "date": 1793700000,
        "chat": {"id": user_id, "type": chat_type},
        "from": {"id": user_id, "is_bot": False, "first_name": "Test"},
    }
    if text is not None:
        message["text"] = text
    else:
        message["document"] = {"file_id": "test", "file_unique_id": "test"}
    return {"update_id": number, "message": message}


@pytest.fixture
async def runtime(store, settings):
    settings = replace(settings, webhook_secret=SECRET)
    bot = Bot("12345:" + "x" * 35, session=AiohttpSession(timeout=1))
    bot.send_message = AsyncMock()
    model = AsyncMock()
    model.complete.return_value = "ответ"
    workers = Workers(store, model, bot, settings)
    app = create_app(settings, store, bot, create_dispatcher(), workers)
    async with TestClient(TestServer(app)) as client:
        yield client, workers, model, bot
    await workers.close()
    await bot.session.close()


async def post(client, payload, secret=SECRET):
    return await client.post(
        "/telegram/webhook", json=payload, headers={"X-Telegram-Bot-Api-Secret-Token": secret}
    )


async def drain(store, workers):
    for _ in range(20):
        job = await store.claim()
        if job is None:
            return
        await workers.process(job)
    pytest.fail("Очередь не опустела за 20 заданий")


async def test_secret_and_access_are_checked_before_database_and_model(runtime, store):
    client, workers, model, bot = runtime
    assert (await post(client, update(), secret="incorrect")).status == 403
    # Closing the pool proves that rejection does not touch storage.
    await store.close()
    response = await post(client, update(user_id=303))
    assert response.status == 200
    assert (await response.json())["text"] == "Доступ закрыт. Ваш Telegram user ID: 303."
    model.complete.assert_not_called()
    bot.send_message.assert_not_called()


async def test_group_and_empty_allowlist_are_closed(runtime, store):
    client, workers, model, bot = runtime
    assert (await post(client, update(chat_type="group"))).status == 200
    assert await store.claim() is None
    closed = replace(workers.settings, allowed_user_ids=frozenset())
    app = create_app(closed, store, bot, create_dispatcher(), workers)
    async with TestClient(TestServer(app)) as other_client:
        response = await post(other_client, update())
        assert (await response.json())["text"] == "Доступ закрыт. Ваш Telegram user ID: 101."


async def test_webhook_ack_does_not_wait_for_generation_and_is_deduplicated(runtime, store):
    client, workers, model, bot = runtime
    entered, release = asyncio.Event(), asyncio.Event()

    async def slow_model(messages):
        entered.set()
        await release.wait()
        return "готово"

    model.complete.side_effect = slow_model
    workers.start()
    response = await asyncio.wait_for(post(client, update()), timeout=ASYNC_TEST_TIMEOUT)
    assert response.status == 200
    await asyncio.wait_for(entered.wait(), timeout=ASYNC_TEST_TIMEOUT)
    assert (await post(client, update())).status == 200
    model.complete.assert_awaited_once()
    release.set()
    for _ in range(100):
        if await store.history(101):
            break
        await asyncio.sleep(0.01)
    assert await store.history(101) == [("привет", "готово")]
    bot.send_message.assert_awaited_once()


async def test_new_discards_inflight_old_answer(runtime, store):
    client, workers, model, bot = runtime
    entered, release = asyncio.Event(), asyncio.Event()

    async def slow_model(messages):
        entered.set()
        await release.wait()
        return "СТАРЫЙ ОТВЕТ"

    model.complete.side_effect = slow_model
    await post(client, update())
    job = await store.claim()
    running = asyncio.create_task(workers.process(job))
    await entered.wait()
    assert (await post(client, update(2, text="/new"))).status == 200
    release.set()
    await running
    await drain(store, workers)
    assert [call.args[1] for call in bot.send_message.await_args_list] == [NEW]
    assert await store.history(101) == []


async def test_long_answer_uses_plain_text_and_preserves_order(runtime, store):
    client, workers, model, bot = runtime
    long_answer = "🙂<tag>_test_" * 1000
    model.complete.side_effect = [long_answer, "второй ответ"]
    await post(client, update(1, text="первый"))
    await post(client, update(2, text="второй"))
    await drain(store, workers)
    sent = bot.send_message.await_args_list
    assert "".join(call.args[1] for call in sent[:-1]) == long_answer
    assert all(call.kwargs["parse_mode"] is None for call in sent)
    assert all(len(call.args[1].encode("utf-16-le")) // 2 <= 4000 for call in sent)
    assert model.complete.await_args_list[1].args[0][-3:] == [
        {"role": "user", "content": "первый"},
        {"role": "assistant", "content": long_answer},
        {"role": "user", "content": "второй"},
    ]


async def test_attachments_and_commands_do_not_call_model(runtime, store):
    client, workers, model, bot = runtime
    for number, text in enumerate(["/start", "/help", None, "/unknown"], 1):
        await post(client, update(number, text=text))
        await drain(store, workers)
    model.complete.assert_not_called()
    assert bot.send_message.await_args_list[2].args[1] == ONLY_TEXT
    assert await store.history(101) == []


async def test_quota_is_not_retried_in_background(runtime, store):
    client, workers, model, bot = runtime
    model.complete.side_effect = ModelError("quota", "Квота исчерпана")
    await post(client, update())
    await drain(store, workers)
    await drain(store, workers)
    model.complete.assert_awaited_once()
    assert bot.send_message.await_args.args[1] == "Квота исчерпана"
    assert await store.history(101) == []


async def test_model_timeout_is_retried_only_twice(runtime, store):
    client, workers, model, bot = runtime
    model.complete.side_effect = ModelError("timeout", "Время истекло", True, 0)
    await post(client, update())
    await drain(store, workers)
    assert model.complete.await_count == 2
    assert bot.send_message.await_args.args[1] == "Время истекло"


async def test_restart_resumes_saved_answer_without_regeneration(runtime, store):
    client, workers, model, bot = runtime
    await post(client, update())
    job = await store.claim()
    assert await store.begin_generation(job)
    assert await store.save_answer(job, "Сохранённый ответ", is_turn=True)
    async with store.pool.connection() as conn:
        await conn.execute("UPDATE jobs SET lease_until=now()-interval '1 second'")
    restarted = await type(store).open(store.dsn, store.settings, store.bot_id)
    try:
        replacement = Workers(restarted, model, bot, workers.settings)
        await drain(restarted, replacement)
        model.complete.assert_not_called()
        assert bot.send_message.await_args.args[1] == "Сохранённый ответ"
        assert await restarted.history(101) == [("привет", "Сохранённый ответ")]
    finally:
        await restarted.close()


async def test_failed_delivery_retries_saved_chunks_without_regeneration(runtime, store):
    client, workers, model, bot = runtime
    model.complete.return_value = "a" * 5000
    method = SendMessage(chat_id=101, text="test")
    bot.send_message.side_effect = [
        None,
        TelegramNetworkError(method=method, message="network"),
        None,
    ]
    await post(client, update())
    await drain(store, workers)
    async with store.pool.connection() as conn:
        await conn.execute("UPDATE jobs SET available_at=now()")
    await drain(store, workers)
    model.complete.assert_awaited_once()
    assert [len(call.args[1]) for call in bot.send_message.await_args_list] == [4000, 1000, 1000]
    assert await store.history(101) == [("привет", "a" * 5000)]


async def test_database_failure_is_not_acknowledged(runtime, store):
    client, workers, model, bot = runtime
    await store.close()
    assert (await post(client, update())).status == 503
    model.complete.assert_not_called()


async def test_forbidden_delivery_is_recorded_as_failed(runtime, store):
    client, workers, model, bot = runtime
    bot.send_message.side_effect = TelegramForbiddenError(
        method=SendMessage(chat_id=101, text="test"),
        message="forbidden",
    )
    await post(client, update())
    await drain(store, workers)
    async with store.pool.connection() as conn:
        row = await (await conn.execute("SELECT state FROM jobs WHERE update_id=1")).fetchone()
    assert row["state"] == "failed"


async def test_new_serializes_with_an_already_started_send(runtime, store):
    client, workers, model, bot = runtime
    sending, release = asyncio.Event(), asyncio.Event()

    async def delayed_send(*args, **kwargs):
        sending.set()
        await release.wait()

    bot.send_message.side_effect = delayed_send
    await post(client, update())
    job = await store.claim()
    sending_task = asyncio.create_task(workers.process(job))
    await asyncio.wait_for(sending.wait(), timeout=ASYNC_TEST_TIMEOUT)
    reset_task = asyncio.create_task(post(client, update(2, text="/new")))
    await asyncio.sleep(0.02)
    assert not reset_task.done()
    release.set()
    await sending_task
    assert (await reset_task).status == 200
    bot.send_message.side_effect = None
    await drain(store, workers)
    assert bot.send_message.await_args.args[1] == NEW
    assert await store.history(101) == []


async def test_worker_concurrency_is_bounded_across_users(runtime, store):
    client, workers, model, bot = runtime
    workers.settings = replace(workers.settings, allowed_user_ids=frozenset({101, 202, 303}))
    active = maximum = 0
    two_running, third_started, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    third_finished = asyncio.Event()
    process = workers.process

    async def process_and_notify(job):
        await process(job)
        if job["user_id"] == 303:
            third_finished.set()

    workers.process = process_and_notify

    async def slow(messages):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        if active == 2:
            two_running.set()
        if model.complete.await_count == 3:
            third_started.set()
        await release.wait()
        active -= 1
        return "готово"

    model.complete.side_effect = slow
    for number, user_id in enumerate([101, 202, 303], 1):
        await store.accept(number, user_id, question="test")
    workers.start()
    await asyncio.wait_for(two_running.wait(), timeout=ASYNC_TEST_TIMEOUT)
    await asyncio.sleep(0.02)
    assert maximum == 2
    release.set()
    await asyncio.wait_for(third_started.wait(), timeout=ASYNC_TEST_TIMEOUT)
    await asyncio.wait_for(third_finished.wait(), timeout=ASYNC_TEST_TIMEOUT)
    assert len(await store.history(303)) == 1
    assert model.complete.await_count == 3
    assert maximum == 2
