import sqlite3

eol_root_dir = "/fs/ess/PAS2136/eol/data/interim/media_cargo_archive"
inat21_root_dir = "/fs/ess/PAS2136/foundation_model/inat21/raw/train"
bioscan_root_dir = "/fs/scratch/PAS2136/bioscan/cropped_256"


def get_outdir(tag):
    return f"/fs/ess/PAS2136/open_clip/data/evobio10m-{tag}"


schema = """
CREATE TABLE IF NOT EXISTS eol (
    content_id INT NOT NULL,
    page_id INT NOT NULL,
    evobio10m_id TEXT NOT NULL PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS inat21 (
    filename TEXT NOT NULL,
    cls_name TEXT NOT NULL,
    cls_num INT NOT NULL,
    evobio10m_id TEXT NOT NULL PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS bioscan (
    part INT NOT NULL,
    filename TEXT NOT NULL,
    evobio10m_id TEXT NOT NULL PRIMARY KEY
);

-- evobio10m_id is a foreign key for one of the three other tables.
CREATE TABLE IF NOT EXISTS split (
    evobio10m_id TEXT NOT NULL PRIMARY KEY,
    is_val INTEGER NOT NULL,
    is_train_small INTEGER NOT NULL
);

PRAGMA journal_mode=WAL;  -- write-ahead log
"""


def get_db(path):
    try:
        db = sqlite3.connect(path, timeout=120)
    except sqlite3.OperationalError as err:
        print(f"Could not connect to {path} ({err}).")
        raise

    db.execute("PRAGMA busy_timeout = 120000;")  # 120 second timeout
    db.commit()
    db.executescript(schema)
    db.commit()
    return db
