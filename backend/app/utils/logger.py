import logging
import sys
from app.config import settings


def setup_logging():
    level = logging.DEBUG if settings.DEBUG else logging.INFO
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Silence noisy libraries
    for noisy in ("urllib3", "PIL", "httpx", "httpcore", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


async def log_to_db(db, level: str, source: str, message: str, payload=None):
    """Persist a log entry to system_logs table."""
    try:
        from app.models.database import SystemLog
        entry = SystemLog(level=level, source=source, message=message, payload_json=payload)
        db.add(entry)
        await db.commit()
    except Exception:
        pass  # Never let logging break the app
