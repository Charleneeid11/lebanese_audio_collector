import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.db import DB
from src.cfg import Settings
from sqlalchemy import update, text
settings = Settings.load()
db = DB(settings.db_url)
with db.engine.connect() as conn:
    r = conn.execute(text("""
        UPDATE queue SET status = 'SCREENED' 
        WHERE platform = 'youtube' 
        AND status IN ('WEAK_POSITIVE', 'WEAK_NEGATIVE', 'SCREENED_NO_LABEL')
    """))
    conn.commit()
    print(f"Reset {r.rowcount} YouTube items to SCREENED")