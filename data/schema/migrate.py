"""
Applies every .sql file in this directory, in filename order, against the
configured database. Idempotent — safe to re-run (all DDL uses IF NOT EXISTS).

Usage:
    python -m data.schema.migrate
"""
from __future__ import annotations

import pathlib

from sqlalchemy import create_engine, text

from config.settings import settings

SCHEMA_DIR = pathlib.Path(__file__).parent


def migrate() -> None:
    engine = create_engine(settings.db_url)
    sql_files = sorted(SCHEMA_DIR.glob("*.sql"))
    if not sql_files:
        print("No .sql migration files found.")
        return

    with engine.begin() as conn:
        for path in sql_files:
            print(f"Applying {path.name} ...")
            sql = path.read_text()
            conn.execute(text(sql))
    print(f"Applied {len(sql_files)} migration file(s).")


if __name__ == "__main__":
    migrate()
