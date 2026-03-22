from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy import (
    create_engine,
    String,
    Integer,
    DateTime,
    Text,
    select,
    update,
    JSON,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session


class Base(DeclarativeBase):
    pass


class QueueItem(Base):
    __tablename__ = "queue"

    id = mapped_column(Integer, primary_key=True)
    url = mapped_column(String(500), index=True)
    platform = mapped_column(String(50), index=True)
    status: Mapped[str] = mapped_column(String(50), default="DISCOVERED", index=True)
    discovered_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )
    last_update_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )
    error_msg: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    audio_path: Mapped[Optional[str]] = mapped_column(
        String(500), nullable=True
    )
    duration_seconds: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )
    rejection_reason: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True
    )
    source_metadata: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True
    )


class DB:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url, echo=False, future=True)
        Base.metadata.create_all(self.engine)
        self._migrate_source_metadata()

    def _migrate_source_metadata(self) -> None:
        """Add source_metadata column if missing (for existing DBs)."""
        with self.engine.connect() as conn:
            if "sqlite" in str(self.engine.url):
                r = conn.execute(text("PRAGMA table_info(queue)"))
                cols = [row[1] for row in r]
                if "source_metadata" not in cols:
                    conn.execute(text("ALTER TABLE queue ADD COLUMN source_metadata TEXT"))
                    conn.commit()

    # -----------------------------
    # INSERT
    # -----------------------------
    def add_to_queue(
        self,
        url: str,
        platform: str,
        source_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Insert a new URL into the queue if it does not already exist.
        Deduplication is URL-based. Returns True if inserted, False if duplicate.
        """
        with Session(self.engine) as session:
            existing = session.scalar(
                select(QueueItem).where(QueueItem.url == url)
            )
            if existing:
                return False

            item = QueueItem(
                url=url,
                platform=platform,
                status="DISCOVERED",
                source_metadata=source_metadata,
            )
            session.add(item)
            session.commit()
            return True

    # -----------------------------
    # FETCH (FIXED: NO HARD LIMIT)
    # -----------------------------
    def fetch_screened_with_metadata(self, limit: Optional[int] = None) -> List[QueueItem]:
        """Fetch SCREENED items that have source_metadata (for weak labeling)."""
        with Session(self.engine) as session:
            stmt = (
                select(QueueItem)
                .where(
                    QueueItem.status == "SCREENED",
                    QueueItem.source_metadata.isnot(None),
                )
                .order_by(QueueItem.id.asc())
            )
            if limit is not None:
                stmt = stmt.limit(limit)
            return list(session.scalars(stmt))

    def fetch_queue(
        self,
        status: str,
        limit: Optional[int] = None,
    ) -> List[QueueItem]:
        """
        Fetch queue items by status.

        IMPORTANT:
        - If limit is None → fetch ALL matching rows
        - If limit is provided → apply SQL LIMIT

        This removes the hidden 50-item throttle.
        """
        with Session(self.engine) as session:
            stmt = (
                select(QueueItem)
                .where(QueueItem.status == status)
                .order_by(QueueItem.discovered_at.asc())
            )

            if limit is not None:
                stmt = stmt.limit(limit)

            return list(session.scalars(stmt))

    # -----------------------------
    # STATUS UPDATES
    # -----------------------------
    def update_status(
        self,
        item_id: int,
        status: str,
        error_msg: Optional[str] = None,
    ) -> None:
        with Session(self.engine) as session:
            stmt = (
                update(QueueItem)
                .where(QueueItem.id == item_id)
                .values(
                    status=status,
                    last_update_at=datetime.utcnow(),
                    error_msg=error_msg,
                )
            )
            session.execute(stmt)
            session.commit()

    def mark_downloaded(
        self,
        item_id: int,
        audio_path: str,
        duration_seconds: int,
    ) -> None:
        """
        Mark item as successfully downloaded.
        """
        with Session(self.engine) as session:
            stmt = (
                update(QueueItem)
                .where(QueueItem.id == item_id)
                .values(
                    audio_path=audio_path,
                    duration_seconds=duration_seconds,
                    status="DOWNLOADED",
                    last_update_at=datetime.utcnow(),
                    error_msg=None,
                )
            )
            session.execute(stmt)
            session.commit()

    def mark_rejected(self, item_id: int, reason: dict) -> None:
        """
        Mark item as rejected with structured reason.
        """
        with Session(self.engine) as session:
            stmt = (
                update(QueueItem)
                .where(QueueItem.id == item_id)
                .values(
                    status="REJECTED",
                    rejection_reason=reason,
                    last_update_at=datetime.utcnow(),
                )
            )
            session.execute(stmt)
            session.commit()

    def update_source_metadata(self, item_id: int, source_metadata: dict) -> None:
        """Update source_metadata for an item (e.g. backfill)."""
        with Session(self.engine) as session:
            stmt = (
                update(QueueItem)
                .where(QueueItem.id == item_id)
                .values(
                    source_metadata=source_metadata,
                    last_update_at=datetime.utcnow(),
                )
            )
            session.execute(stmt)
            session.commit()

    def fetch_podcast_needing_backfill(self, limit: Optional[int] = None) -> List[QueueItem]:
        """Fetch podcast/podcast_rss items with missing source_metadata for backfill."""
        with Session(self.engine) as session:
            stmt = (
                select(QueueItem)
                .where(
                    QueueItem.platform.in_(["podcast", "podcast_rss"]),
                    QueueItem.source_metadata.is_(None),
                )
                .order_by(QueueItem.id.asc())
            )
            if limit is not None:
                stmt = stmt.limit(limit)
            return list(session.scalars(stmt))

    def fetch_youtube_needing_backfill(self, limit: Optional[int] = None) -> List[QueueItem]:
        """Fetch YouTube items with missing source_metadata for backfill."""
        with Session(self.engine) as session:
            stmt = (
                select(QueueItem)
                .where(
                    QueueItem.platform == "youtube",
                    QueueItem.source_metadata.is_(None),
                )
                .order_by(QueueItem.id.asc())
            )
            if limit is not None:
                stmt = stmt.limit(limit)
            return list(session.scalars(stmt))

    def fetch_queue_by_platform(
        self,
        status: str,
        platform: str,
        limit: int,
    ):
        """
        Fetch a deterministic batch filtered at DB level.
        Oldest first to avoid starvation.
        """
        with Session(self.engine) as session:
            stmt = (
                select(QueueItem)
                .where(
                    QueueItem.status == status,
                    QueueItem.platform == platform,
                )
                .order_by(QueueItem.discovered_at.asc())
                .limit(limit)
            )
            return list(session.scalars(stmt))

