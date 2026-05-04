"""
SQLite database connection and initialisation for the Meta-Agent.

Provides a single connection helper used by all modules in core/.
Schema is defined in db/schema.sql and applied on first run.
"""

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "meta_agent.db"
SCHEMA_PATH = PROJECT_ROOT / "db" / "schema.sql"


def initialise_database(db_path: Path = DEFAULT_DB_PATH) -> None:
    """
    Create the database file and apply the schema if it doesn't exist.

    Idempotent — safe to call multiple times. The schema uses
    IF NOT EXISTS clauses so re-running it does not destroy data.

    Args:
        db_path: Where to create the SQLite file.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema file not found at {SCHEMA_PATH}")

    schema = SCHEMA_PATH.read_text()

    with sqlite3.connect(db_path) as conn:
        conn.executescript(schema)
        conn.commit()

    logger.info(f"Database initialised at {db_path}")


@contextmanager
def get_connection(db_path: Path = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    """
    Yield a SQLite connection with sensible defaults.

    Use as a context manager:

        with get_connection() as conn:
            cursor = conn.execute("SELECT * FROM students")
            ...

    The connection is automatically committed on success and rolled back
    on exception. Foreign keys are enforced.

    Args:
        db_path: Path to the SQLite database file.

    Yields:
        An open sqlite3.Connection with row factory enabled.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # access columns by name
    conn.execute("PRAGMA foreign_keys = ON")  # enforce FK constraints

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()