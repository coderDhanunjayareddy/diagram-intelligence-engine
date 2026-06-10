import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Retrieve database connection string (default: SQLite fallback for local-dev)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///C:/Work/PPT Generation application/backend/storage/diagram_v2.db")

# Ensure storage directory exists for SQLite
if DATABASE_URL.startswith("sqlite"):
    db_dir = os.path.dirname(DATABASE_URL.replace("sqlite:///", ""))
    os.makedirs(db_dir, exist_ok=True)

# Create engine (PostgreSQL or SQLite)
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
