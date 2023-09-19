import logging
import os
import random
import sqlite3
import time

logger = logging.getLogger(f"helpers(p{os.getpid()})")


def executerobustly(db, stmt, values, *, max_attempts=-1):
    attempt = 0
    while max_attempts < 0 or attempt < max_attempts:
        attempt += 1
        try:
            db.executemany(stmt, values)
            db.commit()
            logger.debug("Successful commit. [attempt: %d]", attempt)
            return
        except sqlite3.OperationalError as err:
            logger.warning("Error. [attempt: %d, err: %s]", attempt, err)

            if attempt == max_attempts:
                raise err

            sleep = attempt**2 + random.randrange(10)
            logger.info("Sleeping. [attempt: %d, sleep: %d]", attempt, sleep)
            time.sleep(sleep)


if __name__ == "__main__":
    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)

    test_db_path = "helpers-test.sqlite"
    db = sqlite3.connect(test_db_path)
    executerobustly(
        db, "INSERT INTO missing (name) VALUES (?)", [("a",), ("b",)], max_attempts=4
    )
    os.remove(test_db_path)
