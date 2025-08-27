"""Common database utilities for BSC AI Apps."""

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from contextlib import contextmanager
from .logging_config import get_logger
from .error_handling import ConfigurationError

logger = get_logger(__name__)

class DatabaseConnection:
    """SQLite database connection manager."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row  # Enable column access by name
            yield conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def execute_query(self, query: str, params: Tuple = ()) -> List[Dict[str, Any]]:
        """Execute a SELECT query and return results as list of dicts."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(query, params)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
            except Exception as e:
                logger.error(f"Query execution error: {e}")
                raise

    def execute_update(self, query: str, params: Tuple = ()) -> int:
        """Execute an INSERT, UPDATE, or DELETE query and return affected rows."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(query, params)
                conn.commit()
                return cursor.rowcount
            except Exception as e:
                conn.rollback()
                logger.error(f"Update execution error: {e}")
                raise

    def create_table(self, table_name: str, schema: Dict[str, str]):
        """Create a table with the given schema."""
        columns = ", ".join([f"{col} {col_type}" for col, col_type in schema.items()])
        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(query)
                conn.commit()
                logger.info(f"Created table {table_name}")
            except Exception as e:
                logger.error(f"Table creation error: {e}")
                raise

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
        result = self.execute_query(query, (table_name,))
        return len(result) > 0

def init_database(db_path: Path, schema: Dict[str, Dict[str, str]]) -> DatabaseConnection:
    """Initialize database with given schema."""
    db = DatabaseConnection(db_path)

    for table_name, table_schema in schema.items():
        db.create_table(table_name, table_schema)

    return db

def get_db_connection(db_path: Optional[Path] = None) -> DatabaseConnection:
    """Get a database connection with default path."""
    if db_path is None:
        db_path = Path("./data/app.db")
    return DatabaseConnection(db_path)

# Example schema for common use cases
DEFAULT_SCHEMAS = {
    "sessions": {
        "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "session_id": "TEXT UNIQUE NOT NULL",
        "study_name": "TEXT NOT NULL",
        "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        "status": "TEXT DEFAULT 'active'"
    },
    "files": {
        "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "filename": "TEXT NOT NULL",
        "filepath": "TEXT NOT NULL",
        "upload_date": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        "file_type": "TEXT",
        "file_size": "INTEGER"
    },
    "results": {
        "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "session_id": "TEXT NOT NULL",
        "result_type": "TEXT NOT NULL",
        "result_data": "TEXT NOT NULL",  # JSON string
        "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
    }
}