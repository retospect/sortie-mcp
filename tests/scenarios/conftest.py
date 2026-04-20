"""Shared fixtures for scenario tests that require a real PostgreSQL database.

Set TEST_DATABASE_URL to point at a Postgres instance, e.g.:
    export TEST_DATABASE_URL="postgresql://localhost/test_sortie"

Every test gets its own schema (sortie_test_XXXX) which is dropped on
teardown, so tests are fully isolated and can run in parallel.
"""

from __future__ import annotations

import os
import uuid

import asyncpg
import pytest

from sortie_mcp.db import DB


def _dsn() -> str:
    return os.environ.get("TEST_DATABASE_URL", "postgresql://localhost/test_sortie")


@pytest.fixture(scope="session")
def pg_dsn() -> str:
    """The Postgres DSN used for scenario tests."""
    return _dsn()


@pytest.fixture(scope="session")
async def _ensure_pg(pg_dsn: str) -> None:
    """Verify we can connect to Postgres. Skip the whole session if not."""
    try:
        conn = await asyncpg.connect(pg_dsn)
        # Ensure pgvector extension exists (needed for notes embedding column)
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except asyncpg.InsufficientPrivilegeError:
            pass  # fine — tests that need it will skip
        await conn.close()
    except (OSError, asyncpg.PostgresError) as exc:
        pytest.skip(f"Postgres not available at {pg_dsn}: {exc}")


@pytest.fixture
async def db(_ensure_pg: None, pg_dsn: str) -> DB:
    """A DB instance with a unique, isolated schema. Dropped on teardown."""
    schema = f"sortie_test_{uuid.uuid4().hex[:8]}"
    instance = DB(pg_dsn, schema=schema)
    await instance.connect()
    await instance.migrate()
    yield instance
    # Teardown: drop the schema entirely
    async with instance.pool.acquire() as conn:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    await instance.close()
