"""Database connection and session management for DeviceGuard AI."""

import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.config import settings
from src.database.models import Base

# Ensure SQLite directory exists if using local sqlite file
if settings.DATABASE_URL.startswith("sqlite"):
    db_file_path = settings.DATABASE_URL.replace("sqlite:///", "").replace("sqlite://", "")
    if db_file_path and not db_file_path.startswith(":memory:"):
        Path(db_file_path).resolve().parent.mkdir(parents=True, exist_ok=True)
    connect_args = {"check_same_thread": False}
else:
    connect_args = {}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    echo=settings.DEBUG,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Dependency that provides a transactional database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables defined in SQLAlchemy Base and perform non-destructive schema updates."""
    Base.metadata.create_all(bind=engine)
    if settings.DATABASE_URL.startswith("sqlite"):
        try:
            with engine.connect() as conn:
                from sqlalchemy import text
                res = conn.execute(text("PRAGMA table_info(alert_records)"))
                cols = [row[1] for row in res.fetchall()]
                if cols and "acknowledged_at" not in cols:
                    conn.execute(text("ALTER TABLE alert_records ADD COLUMN acknowledged_at DATETIME"))
                    conn.commit()
        except Exception:
            pass
