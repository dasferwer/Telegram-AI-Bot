import os
from dataclasses import replace

import pytest

from telegram_ai.settings import Settings
from telegram_ai.storage import Store


@pytest.fixture
def settings():
    return replace(Settings(), allowed_user_ids=frozenset({101, 202}), min_request_interval=0)


@pytest.fixture
async def store(settings):
    dsn = os.environ.get("TEST_DATABASE_URL")
    if not dsn:
        pytest.fail("Задайте TEST_DATABASE_URL отдельной тестовой PostgreSQL, см. README")
    database = await Store.open(dsn, settings, bot_id=12345)
    async with database.pool.connection() as conn:
        row = await (await conn.execute("SELECT current_database() AS name")).fetchone()
        if not row["name"].endswith("_test"):
            await database.close()
            pytest.fail("Имя тестовой БД должно оканчиваться на _test")
    await database.migrate()
    async with database.pool.connection() as conn:
        await conn.execute("TRUNCATE turns, jobs, conversations")
    yield database
    await database.close()
