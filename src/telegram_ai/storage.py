import asyncio
import hashlib
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from importlib.resources import files
from typing import Any
from uuid import uuid4

import certifi
from psycopg import AsyncConnection
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from .dialogue import split_text
from .settings import Settings

Job = dict[str, Any]
Sender = Callable[[int, str], Awaitable[None]]
_DB_TIMEOUT_SECONDS = 25


class Store:
    def __init__(self, pool: AsyncConnectionPool, dsn: str, settings: Settings, bot_id: int):
        self.pool, self.dsn, self.settings, self.bot_id = pool, dsn, settings, bot_id

    @classmethod
    async def open(cls, dsn: str, settings: Settings, bot_id: int) -> "Store":
        options: dict[str, Any] = {
            "autocommit": True,
            "row_factory": dict_row,
            "prepare_threshold": None,
            "connect_timeout": 15,
        }
        host = conninfo_to_dict(dsn).get("host", "")
        if host and host not in {"localhost", "127.0.0.1", "::1"} and not host.startswith("/"):
            options.update(sslmode="verify-full", sslrootcert=certifi.where())
        pool = AsyncConnectionPool(
            dsn,
            kwargs=options,
            min_size=0,
            max_size=settings.workers * 2 + 2,
            max_idle=60,
            timeout=20,
            reconnect_timeout=30,
            open=False,
            check=AsyncConnectionPool.check_connection,
        )
        await pool.open()
        return cls(pool, dsn, settings, bot_id)

    async def close(self) -> None:
        await self.pool.close()

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[AsyncConnection]:
        # Client deadlines also work with PgBouncer, without unsupported startup options.
        async with asyncio.timeout(_DB_TIMEOUT_SECONDS), self.pool.connection() as conn:
            yield conn

    async def migrate(self) -> None:
        async with self.connection() as conn, conn.transaction():
            await conn.execute("SELECT pg_advisory_xact_lock(684902143)")
            await conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations "
                "(version TEXT PRIMARY KEY, checksum TEXT NOT NULL)"
            )
            for path in sorted(
                files("telegram_ai").joinpath("migrations").iterdir(), key=lambda entry: entry.name
            ):
                if not path.name.endswith(".sql"):
                    continue
                source = path.read_text(encoding="utf-8")
                checksum = hashlib.sha256(source.encode()).hexdigest()
                row = await (
                    await conn.execute(
                        "SELECT checksum FROM schema_migrations WHERE version=%s",
                        (path.name,),
                    )
                ).fetchone()
                if row:
                    if row["checksum"] != checksum:
                        raise RuntimeError("Изменена уже применённая миграция")
                    continue
                await conn.execute(source, prepare=False)
                await conn.execute(
                    "INSERT INTO schema_migrations VALUES (%s,%s)", (path.name, checksum)
                )

    async def accept(
        self,
        update_id: int,
        user_id: int,
        *,
        question: str | None = None,
        answer: str | None = None,
        reset: bool = False,
    ) -> str:
        async with self.connection() as conn, conn.transaction():
            # Serializes admission and the shared queue limit, not model generation.
            await conn.execute("SELECT pg_advisory_xact_lock(%s)", (self.bot_id,))
            if await (
                await conn.execute(
                    "SELECT 1 FROM jobs WHERE bot_id=%s AND update_id=%s",
                    (self.bot_id, update_id),
                )
            ).fetchone():
                return "duplicate"
            await conn.execute(
                "INSERT INTO conversations(bot_id,user_id) VALUES (%s,%s) ON CONFLICT DO NOTHING",
                (self.bot_id, user_id),
            )
            conversation = await (
                await conn.execute(
                    "SELECT *, EXTRACT(EPOCH FROM (now()-last_request)) AS elapsed "
                    "FROM conversations WHERE bot_id=%s AND user_id=%s FOR UPDATE",
                    (self.bot_id, user_id),
                )
            ).fetchone()
            epoch = conversation["epoch"]
            result = "accepted"
            if reset:
                epoch += 1
                await conn.execute(
                    "UPDATE conversations SET epoch=%s WHERE bot_id=%s AND user_id=%s",
                    (epoch, self.bot_id, user_id),
                )
                await conn.execute(
                    "DELETE FROM turns WHERE bot_id=%s AND user_id=%s", (self.bot_id, user_id)
                )
                await conn.execute(
                    "UPDATE jobs SET question=NULL,answer=NULL,is_turn=FALSE, "
                    "state=CASE WHEN state IN ('pending','running') "
                    "THEN 'cancelled' ELSE state END, "
                    "finished_at=COALESCE(finished_at,now()),owner=NULL,lease_until=NULL "
                    "WHERE bot_id=%s AND user_id=%s",
                    (self.bot_id, user_id),
                )
            else:
                counts = await (
                    await conn.execute(
                        "SELECT count(*) AS total, count(*) FILTER (WHERE user_id=%s) AS personal "
                        "FROM jobs WHERE bot_id=%s AND state IN ('pending','running')",
                        (user_id, self.bot_id),
                    )
                ).fetchone()
                if (
                    counts["personal"] >= self.settings.max_pending_per_user
                    or counts["total"] >= self.settings.max_pending_total
                ):
                    result = "busy"
                elif (
                    question is not None
                    and conversation["elapsed"] is not None
                    and conversation["elapsed"] < self.settings.min_request_interval
                ):
                    result = "rate"
            if result == "accepted" and question is not None:
                await conn.execute(
                    "UPDATE conversations SET last_request=now() WHERE bot_id=%s AND user_id=%s",
                    (self.bot_id, user_id),
                )
            await conn.execute(
                "INSERT INTO jobs("
                "bot_id,update_id,user_id,epoch,question,answer,state,finished_at) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,CASE WHEN %s THEN now() ELSE NULL END)",
                (
                    self.bot_id,
                    update_id,
                    user_id,
                    epoch,
                    question if result == "accepted" else None,
                    answer if result == "accepted" else None,
                    "pending" if result == "accepted" else "done",
                    result != "accepted",
                ),
            )
            # Telegram only retains updates for 24 hours; keep tombstones for seven days.
            await conn.execute(
                "DELETE FROM jobs WHERE bot_id=%s AND finished_at < now()-interval '7 days'",
                (self.bot_id,),
            )
            return result

    async def claim(self) -> Job | None:
        async with self.connection() as conn, conn.transaction():
            row = await (
                await conn.execute(
                    "SELECT j.* FROM conversations c JOIN LATERAL ("
                    " SELECT * FROM jobs j WHERE j.bot_id=c.bot_id AND j.user_id=c.user_id "
                    " AND j.state IN ('pending','running') ORDER BY update_id LIMIT 1"
                    ") j ON TRUE WHERE c.bot_id=%s AND j.available_at<=now() "
                    "AND (j.state='pending' OR j.lease_until<=now()) "
                    "ORDER BY j.update_id LIMIT 1 FOR UPDATE OF c SKIP LOCKED",
                    (self.bot_id,),
                )
            ).fetchone()
            if row is None:
                return None
            # Re-read after acquiring the conversation lock. A competing transaction
            # may have committed a lease after the candidate query took its snapshot.
            head = await (
                await conn.execute(
                    "SELECT *, available_at<=clock_timestamp() AND "
                    "(state='pending' OR lease_until<=clock_timestamp()) AS ready "
                    "FROM jobs WHERE bot_id=%s AND user_id=%s AND state IN ('pending','running') "
                    "ORDER BY update_id LIMIT 1",
                    (self.bot_id, row["user_id"]),
                )
            ).fetchone()
            if head is None or not head["ready"]:
                return None
            return await (
                await conn.execute(
                    "UPDATE jobs SET state='running',owner=%s,"
                    "lease_until=now()+%s*interval '1 second' "
                    "WHERE bot_id=%s AND update_id=%s RETURNING *",
                    (uuid4(), self.settings.lease_seconds, self.bot_id, head["update_id"]),
                )
            ).fetchone()

    async def next_delay(self) -> float | None:
        async with self.connection() as conn:
            row = await (
                await conn.execute(
                    "SELECT EXTRACT(EPOCH FROM (min(due)-now())) AS seconds FROM ("
                    " SELECT DISTINCT ON (user_id) CASE WHEN state='running' "
                    " THEN GREATEST(lease_until,available_at) ELSE available_at END AS due "
                    " FROM jobs WHERE bot_id=%s AND state IN ('pending','running') "
                    " ORDER BY user_id,update_id) heads",
                    (self.bot_id,),
                )
            ).fetchone()
            return None if row["seconds"] is None else max(0.1, float(row["seconds"]))

    async def renew(self, job: Job) -> bool:
        async with self.connection() as conn:
            result = await conn.execute(
                "UPDATE jobs SET lease_until=now()+%s*interval '1 second' "
                "WHERE bot_id=%s AND update_id=%s AND owner=%s AND state='running' "
                "AND lease_until>now()",
                (self.settings.lease_seconds, self.bot_id, job["update_id"], job["owner"]),
            )
            return result.rowcount == 1

    async def _owned(self, conn: Any, job: Job) -> Job | None:
        await conn.execute(
            "SELECT 1 FROM conversations WHERE bot_id=%s AND user_id=%s FOR UPDATE",
            (self.bot_id, job["user_id"]),
        )
        return await (
            await conn.execute(
                "SELECT j.* FROM jobs j JOIN conversations c USING(bot_id,user_id) "
                "WHERE j.bot_id=%s AND j.update_id=%s AND j.owner=%s AND j.state='running' "
                "AND j.epoch=c.epoch AND j.lease_until>now()",
                (self.bot_id, job["update_id"], job["owner"]),
            )
        ).fetchone()

    async def begin_generation(self, job: Job) -> bool:
        async with self.connection() as conn, conn.transaction():
            current = await self._owned(conn, job)
            if (
                not current
                or current["generation_attempts"] >= self.settings.max_generation_attempts
            ):
                return False
            await conn.execute(
                "UPDATE jobs SET generation_attempts=generation_attempts+1 "
                "WHERE bot_id=%s AND update_id=%s",
                (self.bot_id, job["update_id"]),
            )
            return True

    async def save_answer(self, job: Job, answer: str, *, is_turn: bool) -> bool:
        async with self.connection() as conn, conn.transaction():
            if not await self._owned(conn, job):
                return False
            await conn.execute(
                "UPDATE jobs SET answer=%s,is_turn=%s WHERE bot_id=%s AND update_id=%s",
                (answer, is_turn, self.bot_id, job["update_id"]),
            )
            return True

    async def history(self, user_id: int) -> list[tuple[str, str]]:
        async with self.connection() as conn:
            rows = await (
                await conn.execute(
                    "SELECT question,answer FROM turns WHERE bot_id=%s AND user_id=%s "
                    "ORDER BY update_id DESC LIMIT %s",
                    (self.bot_id, user_id, self.settings.history_turns),
                )
            ).fetchall()
            return [(row["question"], row["answer"]) for row in reversed(rows)]

    async def send_next(self, job: Job, sender: Sender) -> str:
        error = None
        outcome = "stale"
        async with self.connection() as conn, conn.transaction():
            current = await self._owned(conn, job)
            if not current:
                return outcome
            chunks = split_text(current["answer"] or "")
            if current["sent_chunks"] >= len(chunks):
                return "complete"
            if current["send_attempts"] >= self.settings.max_send_attempts:
                await self._finish(conn, current, "failed")
                return "failed"
            await conn.execute(
                "UPDATE jobs SET send_attempts=send_attempts+1 WHERE bot_id=%s AND update_id=%s",
                (self.bot_id, job["update_id"]),
            )
            # The conversation lock prevents a send from crossing a committed /new.
            try:
                await sender(current["user_id"], chunks[current["sent_chunks"]])
            except Exception as exc:
                error = exc
            else:
                await conn.execute(
                    "UPDATE jobs SET sent_chunks=sent_chunks+1,send_attempts=0 "
                    "WHERE bot_id=%s AND update_id=%s",
                    (self.bot_id, job["update_id"]),
                )
                outcome = "sent"
        if error is not None:
            raise error
        return outcome

    async def _finish(self, conn: Any, current: Job, state: str) -> None:
        if state == "done" and current["is_turn"]:
            await conn.execute(
                "INSERT INTO turns(bot_id,user_id,update_id,question,answer) "
                "VALUES (%s,%s,%s,%s,%s) "
                "ON CONFLICT DO NOTHING",
                (
                    self.bot_id,
                    current["user_id"],
                    current["update_id"],
                    current["question"],
                    current["answer"],
                ),
            )
            await conn.execute(
                "DELETE FROM turns WHERE bot_id=%s AND user_id=%s AND update_id IN ("
                " SELECT update_id FROM turns WHERE bot_id=%s AND user_id=%s "
                " ORDER BY update_id DESC OFFSET %s)",
                (
                    self.bot_id,
                    current["user_id"],
                    self.bot_id,
                    current["user_id"],
                    self.settings.history_turns,
                ),
            )
        await conn.execute(
            "UPDATE jobs SET state=%s,question=NULL,answer=NULL,owner=NULL,lease_until=NULL, "
            "finished_at=now() WHERE bot_id=%s AND update_id=%s",
            (state, self.bot_id, current["update_id"]),
        )

    async def finish(self, job: Job, *, failed: bool = False) -> bool:
        async with self.connection() as conn, conn.transaction():
            current = await self._owned(conn, job)
            if not current:
                return False
            await self._finish(conn, current, "failed" if failed else "done")
            return True

    async def defer(self, job: Job, delay: float) -> None:
        async with self.connection() as conn:
            await conn.execute(
                "UPDATE jobs SET state='pending',owner=NULL,lease_until=NULL, "
                "available_at=now()+%s*interval '1 second' "
                "WHERE bot_id=%s AND update_id=%s AND owner=%s AND state='running'",
                (max(0, delay), self.bot_id, job["update_id"], job["owner"]),
            )
