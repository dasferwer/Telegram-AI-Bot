import asyncio
import os
import sys
from dataclasses import replace

import pytest

from telegram_ai import storage
from telegram_ai.storage import Store


async def test_duplicate_webhook_only_creates_one_job(store):
    results = await asyncio.gather(*(store.accept(1, 101, question="привет") for _ in range(5)))
    assert results.count("accepted") == 1
    assert results.count("duplicate") == 4


async def test_claim_preserves_user_order_but_allows_another_user(store):
    await store.accept(1, 101, question="первый")
    await store.accept(2, 101, question="второй")
    await store.accept(3, 202, question="чужой")
    first, second = await asyncio.gather(store.claim(), store.claim())
    assert {first["update_id"], second["update_id"]} == {1, 3}
    assert await store.claim() is None


async def test_many_competing_claims_never_steal_an_active_lease(store):
    for number in range(1, 31):
        await store.accept(number, 101, question="concurrent")
        results = await asyncio.gather(*(store.claim() for _ in range(12)))
        claimed = [job for job in results if job is not None]
        assert len(claimed) == 1
        assert await store.finish(claimed[0], failed=True)


async def test_new_cancels_old_results_and_clears_history(store):
    await store.accept(1, 101, question="старое")
    job = await store.claim()
    await store.accept(2, 101, answer="История очищена", reset=True)
    assert not await store.save_answer(job, "поздний ответ", is_turn=True)
    assert await store.history(101) == []
    reset_job = await store.claim()
    assert reset_job["update_id"] == 2
    assert reset_job["epoch"] == 1


async def test_history_is_isolated_bounded_and_survives_reconnect(store):
    store.settings = replace(store.settings, history_turns=2)
    for number in range(1, 4):
        await store.accept(number, 101, question=f"вопрос {number}")
        job = await store.claim()
        assert await store.save_answer(job, f"ответ {number}", is_turn=True)
        assert await store.finish(job)
    assert len(await store.history(101)) == 2
    assert await store.history(202) == []
    another = await Store.open(store.dsn, store.settings, store.bot_id)
    try:
        assert await another.history(101) == [("вопрос 2", "ответ 2"), ("вопрос 3", "ответ 3")]
    finally:
        await another.close()


async def test_expired_lease_is_recovered_with_new_owner(store):
    await store.accept(1, 101, question="восстановление")
    abandoned = await store.claim()
    async with store.pool.connection() as conn:
        await conn.execute("UPDATE jobs SET lease_until=now()-interval '1 second'")
    replacement = await Store.open(store.dsn, store.settings, store.bot_id)
    try:
        recovered = await replacement.claim()
        assert recovered["update_id"] == abandoned["update_id"]
        assert recovered["owner"] != abandoned["owner"]
        assert not await store.save_answer(abandoned, "устаревшее", is_turn=True)
        assert await replacement.save_answer(recovered, "готово", is_turn=True)
    finally:
        await replacement.close()


async def test_pending_limit_and_rate_limit_persist(store):
    store.settings = replace(store.settings, max_pending_per_user=1)
    assert await store.accept(1, 101, question="a") == "accepted"
    assert await store.accept(2, 101, question="b") == "busy"
    assert await store.accept(2, 101, question="b") == "duplicate"
    await store.accept(3, 101, answer="reset", reset=True)
    store.settings = replace(store.settings, min_request_interval=60, max_pending_per_user=3)
    assert await store.accept(4, 101, question="c") == "rate"


async def test_job_survives_real_handler_process_exit(store):
    await store.accept(77, 101, question="сохранить до падения")
    source = """
import asyncio, os
from telegram_ai.settings import Settings
from telegram_ai.storage import Store
async def main():
    store = await Store.open(os.environ['TEST_DATABASE_URL'], Settings(), 12345)
    job = await store.claim()
    assert job['update_id'] == 77
    assert await store.begin_generation(job)
    os._exit(0)
asyncio.run(main())
"""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        source,
        env=os.environ.copy(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await asyncio.wait_for(process.communicate(), timeout=15)
    assert process.returncode == 0, stderr.decode()
    assert await store.claim() is None
    async with store.pool.connection() as conn:
        await conn.execute("UPDATE jobs SET lease_until=now()-interval '1 second'")
    recovered = await store.claim()
    assert recovered["update_id"] == 77
    assert recovered["question"] == "сохранить до падения"
    assert recovered["generation_attempts"] == 1


async def test_database_deadline_cancels_slow_query_and_pool_recovers(store, monkeypatch):
    monkeypatch.setattr(storage, "_DB_TIMEOUT_SECONDS", 0.05)
    with pytest.raises(TimeoutError):
        async with store.connection() as conn:
            await conn.execute("SELECT pg_sleep(1)")
    monkeypatch.setattr(storage, "_DB_TIMEOUT_SECONDS", 25)
    assert await store.accept(1, 101, question="после таймаута") == "accepted"
