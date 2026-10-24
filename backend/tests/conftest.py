"""Shared test fixtures and database isolation for DeviceGuard AI test suite."""

import os
from pathlib import Path
import sys
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from src.database.connection import get_db
from src.database.models import Base
from src.main import app
from src.services.inference_service import get_inference_service

# Dedicated isolated SQLite test database file
TEST_DB_PATH = BACKEND_DIR / "data" / "test_isolated.db"
TEST_DATABASE_URL = f"sqlite:///{TEST_DB_PATH}"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
)
TestingSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=test_engine,
)


@pytest.fixture(scope="session", autouse=True)
def setup_isolated_test_database():
    """Initializes tables in an isolated test database and cleans it up after tests."""
    # Ensure inference service is warmed up
    service = get_inference_service()
    assert service.lstm_model is not None, "LSTM model must be loaded for tests"

    # Create tables in isolated test database
    Base.metadata.create_all(bind=test_engine)
    yield
    # Teardown
    Base.metadata.drop_all(bind=test_engine)
    test_engine.dispose()
    if TEST_DB_PATH.exists():
        try:
            TEST_DB_PATH.unlink()
        except Exception:
            pass


@pytest.fixture
def db_session():
    """Provides a fresh isolated SQLAlchemy session for testing."""
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client():
    """FastAPI TestClient wired to the isolated test database."""
    def override_get_db():
        session = TestingSessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
