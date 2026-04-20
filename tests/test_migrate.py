"""Tests for the yoyo-based migration layer."""

from __future__ import annotations

import os
import subprocess
import uuid

import pytest

pytestmark = pytest.mark.postgres

PG_USER = "bots"
TEST_DB_NAME = "sortie_migrate_test"
DSN = f"postgresql://{PG_USER}@localhost/{TEST_DB_NAME}"

# Total migrations in src/sortie_mcp/migrations/. Bump when adding new ones.
N_MIGRATIONS = 4


def _run(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(list(args), capture_output=True, check=check, text=True)


def _pg_available() -> bool:
    try:
        _run("createdb", "-U", PG_USER, TEST_DB_NAME, check=False)
        _run(
            "psql",
            "-U",
            PG_USER,
            "-d",
            TEST_DB_NAME,
            "-c",
            "CREATE EXTENSION IF NOT EXISTS vector;",
            check=False,
        )
        _run("dropdb", "-U", PG_USER, "--if-exists", TEST_DB_NAME, check=False)
        return True
    except FileNotFoundError:
        return False


if not _pg_available():
    pytestmark = pytest.mark.skip(reason="PostgreSQL client tools not available")


def _force_drop() -> None:
    """Terminate stragglers (yoyo's psycopg2 conns) then drop.

    yoyo doesn't close its bootstrap connection pool between calls in
    the same process, so ``dropdb`` can return "database is being
    accessed by other users". Kill the backends first.
    """
    _run(
        "psql",
        "-U",
        PG_USER,
        "-d",
        "postgres",
        "-c",
        f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
        f"WHERE datname='{TEST_DB_NAME}' AND pid <> pg_backend_pid();",
        check=False,
    )
    _run("dropdb", "-U", PG_USER, "--if-exists", TEST_DB_NAME, check=False)


@pytest.fixture
def fresh_db():
    """Per-test fresh database with pgvector installed."""
    _force_drop()
    _run("createdb", "-U", PG_USER, TEST_DB_NAME)
    _run(
        "psql",
        "-U",
        PG_USER,
        "-d",
        TEST_DB_NAME,
        "-c",
        "CREATE EXTENSION IF NOT EXISTS vector;",
    )
    yield DSN
    _force_drop()


def _count_sortie_tables(schema: str) -> int:
    result = _run(
        "psql",
        "-U",
        PG_USER,
        "-d",
        TEST_DB_NAME,
        "-tAc",
        f"SELECT count(*) FROM information_schema.tables "
        f"WHERE table_schema='{schema}' "
        f"AND table_name IN ('campaigns','campaign_steps','campaign_notes','notifications')",
    )
    return int(result.stdout.strip())


class TestMigrateCLI:
    def test_apply_creates_all_four_tables(self, fresh_db: str) -> None:
        from sortie_mcp._migrate_impl import apply_migrations

        n = apply_migrations(fresh_db, schema="sortie")
        assert n == N_MIGRATIONS
        assert _count_sortie_tables("sortie") == 4

    def test_apply_is_idempotent(self, fresh_db: str) -> None:
        from sortie_mcp._migrate_impl import apply_migrations

        assert apply_migrations(fresh_db, schema="sortie") == N_MIGRATIONS
        # Second apply: nothing to do
        assert apply_migrations(fresh_db, schema="sortie") == 0

    def test_rollback_drops_tables(self, fresh_db: str) -> None:
        from sortie_mcp._migrate_impl import apply_migrations

        apply_migrations(fresh_db, schema="sortie")
        assert _count_sortie_tables("sortie") == 4

        apply_migrations(fresh_db, schema="sortie", rollback=True)
        assert _count_sortie_tables("sortie") == 0

    def test_dry_run_does_not_apply(self, fresh_db: str) -> None:
        from sortie_mcp._migrate_impl import apply_migrations

        n = apply_migrations(fresh_db, schema="sortie", dry_run=True)
        assert n == N_MIGRATIONS  # reported pending, but not applied
        assert _count_sortie_tables("sortie") == 0

    def test_isolated_schemas_track_history_independently(self, fresh_db: str) -> None:
        """Two schemas in the same DB must each track their own migration state."""
        from sortie_mcp._migrate_impl import apply_migrations

        schema_a = f"sortie_a_{uuid.uuid4().hex[:6]}"
        schema_b = f"sortie_b_{uuid.uuid4().hex[:6]}"

        apply_migrations(fresh_db, schema=schema_a)
        # Rolling back A must not affect B, and applying to B must still work.
        assert apply_migrations(fresh_db, schema=schema_b) == N_MIGRATIONS
        assert _count_sortie_tables(schema_a) == 4
        assert _count_sortie_tables(schema_b) == 4

        apply_migrations(fresh_db, schema=schema_a, rollback=True)
        assert _count_sortie_tables(schema_a) == 0
        assert _count_sortie_tables(schema_b) == 4  # B untouched

    def test_inject_schema_preserves_other_query_params(self) -> None:
        from sortie_mcp._migrate_impl import _inject_schema

        out = _inject_schema("postgresql://u:p@h:5432/d?sslmode=require", "myschema")
        assert "schema=myschema" in out
        assert "sslmode=require" in out

    def test_cli_entry_point_runs(self, fresh_db: str) -> None:
        """The ``sortie-migrate`` script runs end-to-end."""
        result = subprocess.run(
            [
                os.environ.get("VIRTUAL_ENV", "/usr/local") + "/bin/python"
                if os.environ.get("VIRTUAL_ENV")
                else "python",
                "-m",
                "sortie_mcp.migrate",
                "--schema",
                "sortie_cli",
            ],
            env={**os.environ, "DATABASE_URL": fresh_db},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert _count_sortie_tables("sortie_cli") == 4
