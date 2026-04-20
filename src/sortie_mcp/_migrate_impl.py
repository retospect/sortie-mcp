"""Internal migration helper — re-exported from :mod:`sortie_mcp.migrate`.

Split out so the same logic is callable from:
- ``DB.migrate()``              (via the async wrapper)
- ``sortie-migrate`` CLI
- tests

Core idea: yoyo's PostgreSQL backend understands ``?schema=<name>`` in
the DSN and issues ``SET search_path TO <name>, public`` on every
connection. That puts yoyo's own tracking tables (``_yoyo_log``,
``_yoyo_version``, ``yoyo_lock``) inside the target schema, so
``DROP SCHEMA ... CASCADE`` in tests wipes the history cleanly. Migration
``.sql`` files use unqualified table names and are identical across
dev / test / prod.
"""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

log = logging.getLogger(__name__)


def _migrations_dir() -> Path:
    return Path(__file__).parent / "migrations"


def _inject_schema(dsn: str, schema: str) -> str:
    """Return ``dsn`` with ``schema=<schema>`` set in the query string.

    yoyo's PostgreSQL backend consumes this and sets ``search_path``
    on every connection it opens.
    """
    parts = urlparse(dsn)
    query = dict(parse_qsl(parts.query))
    query["schema"] = schema
    return urlunparse(parts._replace(query=urlencode(query)))


def apply_migrations(
    dsn: str,
    schema: str = "sortie",
    *,
    dry_run: bool = False,
    rollback: bool = False,
) -> int:
    """Apply (or roll back) sortie-mcp migrations. Returns the number applied."""
    import psycopg2
    from yoyo import get_backend, read_migrations

    # Ensure the schema exists before yoyo opens its first connection —
    # yoyo will SET search_path immediately and its bootstrap DDL must
    # find the schema there.
    bootstrap = psycopg2.connect(dsn)
    try:
        bootstrap.autocommit = True
        with bootstrap.cursor() as cur:
            cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
    finally:
        bootstrap.close()

    scoped_dsn = _inject_schema(dsn, schema)
    backend = get_backend(scoped_dsn)
    migrations = read_migrations(str(_migrations_dir()))

    with backend.lock():
        if rollback:
            pending = backend.to_rollback(migrations)
            action = "rollback"
        else:
            pending = backend.to_apply(migrations)
            action = "apply"

        if not pending:
            log.info("No migrations to %s (schema=%s)", action, schema)
            return 0

        for m in pending:
            log.info("  %s %s", action, m.id)

        if dry_run:
            log.info("Dry run — %d migrations would be %sd", len(pending), action)
            return len(pending)

        if rollback:
            backend.rollback_migrations(pending)
            verb = "Rolled back"
        else:
            backend.apply_migrations(pending)
            verb = "Applied"
        log.info("%s %d migrations (schema=%s)", verb, len(pending), schema)
        return len(pending)
