"""
Database layer. Same names as before (Base, engine, send_movie, create_table,
Movies, Booking), so agent.py imports work unchanged.

Seats:
  total_seats      = hall capacity (never changes)
  available_seats  = seats still free (goes down on booking, up on cancel)

DATABASE_URL: defaults to a local SQLite file. For real customers use Postgres:
  DATABASE_URL=postgresql+psycopg2://user:pass@host/dbname
NOTE: default file is cinema.db (new), so your old db file does not clash with
the new columns. Delete cinema.db if you change the models later.
"""
import os
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, create_engine, func
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///cinema.db")
_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=_args, pool_pre_ping=True)


class Base(DeclarativeBase):
    pass


class Movies(Base):
    __tablename__ = "movies"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200), unique=True)
    genre: Mapped[str] = mapped_column(String(50))
    show_time: Mapped[str] = mapped_column(String(20))
    price: Mapped[float] = mapped_column(Float)
    total_seats: Mapped[int] = mapped_column(Integer)
    available_seats: Mapped[int] = mapped_column(Integer)


class Booking(Base):
    __tablename__ = "bookings"

    id: Mapped[int] = mapped_column(primary_key=True)
    movie_id: Mapped[int] = mapped_column(ForeignKey("movies.id"), index=True)
    customer_name: Mapped[str] = mapped_column(String(100))
    seats: Mapped[int] = mapped_column(Integer)
    total_amount: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(20), default="confirmed")  # confirmed | cancelled
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


MOVIES = [
    {"title": "Inception",       "genre": "Sci-Fi", "show_time": "7:00 PM", "price": 15.0, "total_seats": 50},
    {"title": "Interstellar",    "genre": "Sci-Fi", "show_time": "8:30 PM", "price": 15.0, "total_seats": 40},
    {"title": "The Dark Knight", "genre": "Action", "show_time": "6:00 PM", "price": 12.0, "total_seats": 60},
    {"title": "Avengers",        "genre": "Action", "show_time": "5:00 PM", "price": 13.0, "total_seats": 80},
    {"title": "The Godfather",   "genre": "Drama",  "show_time": "9:00 PM", "price": 10.0, "total_seats": 30},
    {"title": "RRR",             "genre": "Action", "show_time": "7:30 PM", "price": 14.0, "total_seats": 70},
]


def create_table():
    Base.metadata.create_all(engine)


def send_movie():
    """Insert only the movies that are missing (safe to run on every start)."""
    with Session(engine) as db:
        added = 0
        for m in MOVIES:
            if not db.query(Movies).filter(Movies.title == m["title"]).first():
                db.add(Movies(**m, available_seats=m["total_seats"]))
                added += 1
        db.commit()
        print(f"Added {added} new movies" if added else "Data Already Exists...")