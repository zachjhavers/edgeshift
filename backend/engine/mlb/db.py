import os
from pathlib import Path
from sqlalchemy import create_engine

MLB_DB_PATH = os.getenv("MLB_DB_PATH", str(Path(__file__).parent / "mlb_predictor.db"))


def get_engine():
    """Creates and returns a SQLAlchemy engine for the MLB SQLite database."""
    return create_engine(f"sqlite:///{MLB_DB_PATH}", connect_args={"check_same_thread": False})
