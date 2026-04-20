"""sortie-migrate — CLI to apply pending schema migrations.

Usage::

    sortie-migrate                       # apply pending migrations
    sortie-migrate --dry-run             # list pending without applying
    sortie-migrate --rollback            # roll back the most recent migration
    sortie-migrate --schema sortie_test  # override target schema

Environment:
    DATABASE_URL    PostgreSQL DSN (required, no default)
    SORTIE_SCHEMA   Target schema (default: sortie)

Called by the Ansible sortie role post-install, and safe to re-run: yoyo
acquires an advisory lock and skips already-applied migrations.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from ._migrate_impl import apply_migrations

log = logging.getLogger(__name__)


def _resolve_dsn(cli_dsn: str | None) -> str:
    dsn = cli_dsn or os.environ.get("DATABASE_URL")
    if not dsn:
        print(
            "error: DATABASE_URL must be set (or pass --dsn).",
            file=sys.stderr,
        )
        sys.exit(2)
    return dsn


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Apply sortie-mcp schema migrations")
    parser.add_argument(
        "--dsn",
        help="PostgreSQL DSN (overrides DATABASE_URL)",
    )
    parser.add_argument(
        "--schema",
        default=os.environ.get("SORTIE_SCHEMA", "sortie"),
        help="Target schema name (default: env SORTIE_SCHEMA or 'sortie')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List pending migrations without applying them",
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Roll back the most recent applied migration",
    )
    args = parser.parse_args()

    dsn = _resolve_dsn(args.dsn)
    apply_migrations(
        dsn,
        schema=args.schema,
        dry_run=args.dry_run,
        rollback=args.rollback,
    )


if __name__ == "__main__":
    main()
